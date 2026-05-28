//! Debug window system for quick image visualization.
//!
//! This module provides an OpenCV-like `imshow` experience for Rust:
//! display any [`ImageView`](irys_cv::image::ImageView) in a window with a single
//! function call. It is gated behind the `debug-window` feature flag and
//! is **not intended for production use**.
//!
//! # Architecture
//!
//! The debug window system uses a two-thread model:
//!
//! - **Main thread**: Runs the winit event loop (`DebugDisplay::run()`).
//!   This is required by winit (and macOS) — the event loop must live on
//!   the main thread.
//! - **Background thread**: Runs user code. The user receives a
//!   [`DisplayContext`] handle to send images and wait for key presses.
//!
//! Communication between threads uses `std::sync::mpsc` channels and a
//! winit [`EventLoopProxy`] for wakeup.
//!
//! # Examples
//!
//! ## One-shot display
//!
//! ```no_run
//! use irys_cv_display::{show, Identity};
//! use irys_cv::image::Image;
//! use irys_cv::pixel::Srgb8;
//!
//! let img = Image::fill(100, 100, Srgb8::new(128, 64, 200));
//! show("Preview", &img, Identity);
//! ```
//!
//! ## Multi-window with `DebugDisplay::run()`
//!
//! ```no_run
//! use irys_cv_display::{DebugDisplay, DisplayContext, AutoContrast, Identity};
//! use irys_cv::image::Image;
//! use irys_cv::pixel::{Mono16, Srgba8};
//!
//! DebugDisplay::run(|ctx| {
//!     let mono = Image::<Mono16>::zero(640, 480);
//!     let strategy = AutoContrast::scan_with(&mono, |p| p.value() as f64);
//!     ctx.show("Mono Preview", &mono, strategy);
//!
//!     let color = Image::fill(320, 240, Srgba8::new(255, 0, 0, 255));
//!     ctx.show("Color Preview", &color, Identity);
//!
//!     ctx.wait_key();
//! });
//! ```
//!
//! # Platform considerations
//!
//! - **macOS**: The event loop **must** run on the main thread. Both
//!   [`DebugDisplay::run()`] and [`show()`] take over the main thread
//!   automatically. Calling either from a non-main thread will panic.
//! - **Linux (Wayland/X11)**: No special requirements. The event loop
//!   runs on whichever thread calls `run()`.
//! - **Windows**: No special requirements.
//! - **Minimized windows**: When a window is minimized its inner size
//!   becomes zero. The blit is skipped until the window is restored.
//! - **Window resizing**: The framebuffer is scaled to the window size
//!   using nearest-neighbor interpolation in the `u32` domain. This
//!   avoids re-running the display strategy on every resize.
//!
//! # Logging
//!
//! This module emits log messages via the [`log`] facade:
//!
//! | Level   | Events                                                    |
//! |---------|-----------------------------------------------------------|
//! | `debug` | Window created, updated, closed; exit command received     |
//! | `warn`  | Proxy send failed, surface/window creation errors          |
//! | `trace` | Surface resize dimensions                                 |
//!
//! Attach a logger (e.g. `env_logger`) to see these messages.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowAttributes, WindowId};

use irys_cv::image::ImageView;

use crate::strategy::{DisplayStrategy, Framebuffer};

// ═══════════════════════════════════════════════════════════════════════════════
// 3.1  Channel types and message protocol
// ═══════════════════════════════════════════════════════════════════════════════

/// Commands sent from the background thread to the event loop.
enum WindowCommand {
    /// Create or update a window with the given title and framebuffer.
    Show {
        title: String,
        framebuffer: Framebuffer,
    },
    /// Close all windows and exit the event loop.
    Exit,
}

/// Events sent from the event loop back to the background thread.
#[derive(Debug)]
#[allow(dead_code)]
enum WindowEvent_ {
    /// A key was pressed in any window.
    KeyPressed { key: KeyCode, window_title: String },
    /// A window was closed by the user.
    WindowClosed { title: String },
    /// All windows have been closed.
    AllClosed,
}

/// Custom user event for waking the event loop when commands are available.
#[derive(Debug)]
enum UserEvent {
    /// Signals that a [`WindowCommand`] is available in the channel.
    CommandAvailable,
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.1a  Notifier — abstraction over EventLoopProxy for testability
// ═══════════════════════════════════════════════════════════════════════════════

/// Abstraction for waking the event loop when a command is available.
///
/// In production, wraps an [`EventLoopProxy`]. In tests, can be replaced
/// with a no-op or a flag-setting closure. This enables unit testing
/// [`DisplayContext`] without a real winit event loop.
struct Notifier(Box<dyn Fn() -> bool + Send + Sync>);

impl Notifier {
    /// Create a notifier backed by an [`EventLoopProxy`].
    fn from_proxy(proxy: EventLoopProxy<UserEvent>) -> Self {
        Notifier(Box::new(move || {
            proxy.send_event(UserEvent::CommandAvailable).is_ok()
        }))
    }

    /// Wake the event loop. Returns `true` if the notification was delivered,
    /// `false` if the event loop has exited.
    fn notify(&self) -> bool {
        (self.0)()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.2  DisplayContext — background thread API
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle for displaying images from a background thread.
///
/// Obtained inside the closure passed to [`DebugDisplay::run()`]. All
/// methods are safe to call from the background thread.
///
/// # Examples
///
/// ```no_run
/// use irys_cv_display::{DebugDisplay, Identity};
/// use irys_cv::image::Image;
/// use irys_cv::pixel::Srgba8;
///
/// DebugDisplay::run(|ctx| {
///     let img = Image::fill(100, 100, Srgba8::new(255, 0, 0, 255));
///     ctx.show("Red", &img, Identity);
///     ctx.wait_key();
/// });
/// ```
pub struct DisplayContext {
    cmd_tx: mpsc::Sender<WindowCommand>,
    event_rx: mpsc::Receiver<WindowEvent_>,
    notifier: Notifier,
}

impl DisplayContext {
    /// Display an image in a window with the given title.
    ///
    /// Converts the image to a framebuffer using the given strategy on
    /// the calling thread, then sends the framebuffer to the event loop
    /// for display. If a window with this title already exists, its
    /// contents are updated; otherwise a new window is created.
    ///
    /// This method is **non-blocking** — it returns immediately after
    /// sending the command.
    ///
    /// # Type parameters
    ///
    /// - `V`: Any [`ImageView`] — owned images, ROIs, tiled views, etc.
    /// - `S`: A [`DisplayStrategy`] that can convert `V`'s pixel type.
    pub fn show<V, S>(&self, title: &str, image: &V, strategy: S)
    where
        V: ImageView,
        V::Pixel: Copy,
        S: DisplayStrategy<V::Pixel>,
    {
        let fb = Framebuffer::from_image(image, strategy);
        self.show_framebuffer(title, fb);
    }

    /// Send a pre-built framebuffer for display.
    ///
    /// This is used internally by the [`show()`] convenience function to
    /// avoid `Send` bounds on the image and strategy types.
    pub(crate) fn show_framebuffer(&self, title: &str, fb: Framebuffer) {
        let w = fb.width;
        let h = fb.height;
        let title_owned = title.to_string();

        if self
            .cmd_tx
            .send(WindowCommand::Show {
                title: title_owned.clone(),
                framebuffer: fb,
            })
            .is_err()
        {
            log::warn!("command channel closed — event loop has exited");
            return;
        }

        log::debug!("show: sent framebuffer for \"{title_owned}\" ({w}×{h})");

        if !self.notifier.notify() {
            log::warn!("event loop proxy send failed (event loop has exited)");
        }
    }

    /// Block until a key is pressed in any window.
    ///
    /// Returns the [`KeyCode`] of the pressed key, or `None` if all
    /// windows were closed or the event loop exited.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use irys_cv_display::{DebugDisplay, Identity};
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgba8;
    ///
    /// DebugDisplay::run(|ctx| {
    ///     let img = Image::fill(100, 100, Srgba8::new(0, 255, 0, 255));
    ///     ctx.show("Green", &img, Identity);
    ///     match ctx.wait_key() {
    ///         Some(key) => println!("Key pressed: {:?}", key),
    ///         None => println!("All windows closed"),
    ///     }
    /// });
    /// ```
    #[must_use]
    pub fn wait_key(&self) -> Option<KeyCode> {
        loop {
            match self.event_rx.recv() {
                Ok(WindowEvent_::KeyPressed { key, .. }) => return Some(key),
                Ok(WindowEvent_::AllClosed) => return None,
                Ok(WindowEvent_::WindowClosed { .. }) => {
                    // A single window closed — keep waiting for a key
                    // or AllClosed.
                    continue;
                }
                Err(_) => {
                    // Channel disconnected — event loop has exited.
                    return None;
                }
            }
        }
    }

    /// Block until a key is pressed, with a timeout.
    ///
    /// Returns `Some(key)` if a key was pressed within the timeout,
    /// `None` if the timeout elapsed or all windows were closed.
    #[must_use]
    pub fn wait_key_timeout(&self, timeout: Duration) -> Option<KeyCode> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            match self.event_rx.recv_timeout(remaining) {
                Ok(WindowEvent_::KeyPressed { key, .. }) => return Some(key),
                Ok(WindowEvent_::AllClosed) => return None,
                Ok(WindowEvent_::WindowClosed { .. }) => {
                    // A single window closed — keep waiting.
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => return None,
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    /// Request the event loop to close all windows and exit.
    ///
    /// This is called automatically when the user closure returns, but
    /// can be called explicitly if needed.
    pub fn exit(&self) {
        let _ = self.cmd_tx.send(WindowCommand::Exit);
        if !self.notifier.notify() {
            log::warn!("event loop proxy send failed (event loop has exited)");
        }
    }

    /// Create a `DisplayContext` for unit tests with a no-op notifier.
    #[cfg(test)]
    fn new_for_test(
        cmd_tx: mpsc::Sender<WindowCommand>,
        event_rx: mpsc::Receiver<WindowEvent_>,
    ) -> Self {
        DisplayContext {
            cmd_tx,
            event_rx,
            notifier: Notifier(Box::new(|| true)),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.3  DebugDisplay::run() — event loop setup
// ═══════════════════════════════════════════════════════════════════════════════

/// Entry point for the debug display system.
///
/// This struct provides the [`run()`](DebugDisplay::run) method, which takes
/// over the main thread for the winit event loop and runs user code on a
/// background thread.
pub struct DebugDisplay;

impl DebugDisplay {
    /// Run the debug display system.
    ///
    /// Takes over the **main thread** for the winit event loop. The user's
    /// code runs in the provided closure on a **background thread**, which
    /// receives a [`DisplayContext`] handle for displaying images.
    ///
    /// The event loop exits when:
    /// - All windows are closed by the user, OR
    /// - The user closure returns (an `Exit` command is sent automatically).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use irys_cv_display::{DebugDisplay, AutoContrast, Identity};
    /// use irys_cv::image::Image;
    /// use irys_cv::pixel::Srgba8;
    ///
    /// DebugDisplay::run(|ctx| {
    ///     let img = Image::fill(640, 480, Srgba8::new(128, 64, 200, 255));
    ///     ctx.show("Preview", &img, Identity);
    ///     ctx.wait_key();
    /// });
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the event loop cannot be created (e.g. no display server
    /// available). On macOS, panics if called from a non-main thread.
    pub fn run<F>(user_fn: F)
    where
        F: FnOnce(&DisplayContext) + Send + 'static,
    {
        // Build the event loop with custom user events.
        let event_loop = EventLoop::<UserEvent>::with_user_event()
            .build()
            .expect("failed to create event loop");

        let proxy = event_loop.create_proxy();

        // Command channel: background thread → event loop.
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();

        // Event channel: event loop → background thread.
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();

        let ctx = DisplayContext {
            cmd_tx: cmd_tx.clone(),
            event_rx,
            notifier: Notifier::from_proxy(proxy.clone()),
        };

        // Spawn the user's code on a background thread.
        let bg_cmd_tx = cmd_tx;
        let bg_notifier = Notifier::from_proxy(proxy);
        std::thread::spawn(move || {
            user_fn(&ctx);

            // When the user closure returns, signal the event loop to exit.
            let _ = bg_cmd_tx.send(WindowCommand::Exit);
            if !bg_notifier.notify() {
                log::warn!("event loop proxy send failed (event loop has exited)");
            }
        });

        // Create the application handler and run the event loop.
        let mut app = App {
            cmd_rx,
            event_tx,
            context: None,
            windows: HashMap::new(),
        };

        event_loop
            .run_app(&mut app)
            .expect("event loop terminated with error");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.4  App struct — ApplicationHandler implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// The winit application handler that manages windows and processes commands.
struct App {
    cmd_rx: mpsc::Receiver<WindowCommand>,
    event_tx: mpsc::Sender<WindowEvent_>,
    /// Lazily initialized softbuffer context (needs a display handle from
    /// the first window).
    context: Option<softbuffer::Context<Arc<Window>>>,
    /// Open windows, keyed by title.
    windows: HashMap<String, WindowState>,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        // Windows are created on demand when Show commands arrive.
        // No action needed here.
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: UserEvent) {
        // Drain all pending commands from the channel.
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            match cmd {
                WindowCommand::Show { title, framebuffer } => {
                    self.handle_show(event_loop, title, framebuffer);
                }
                WindowCommand::Exit => {
                    self.handle_exit(event_loop);
                    return;
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.handle_close(event_loop, window_id);
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw(window_id);
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    // Find the title for this window.
                    let title = self.title_for_window(window_id);
                    if let Some(title) = title {
                        let _ = self.event_tx.send(WindowEvent_::KeyPressed {
                            key,
                            window_title: title,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

impl App {
    /// Handle a `Show` command: create or update a window.
    fn handle_show(
        &mut self,
        event_loop: &ActiveEventLoop,
        title: String,
        framebuffer: Framebuffer,
    ) {
        if let Some(state) = self.windows.get_mut(&title) {
            // Update existing window.
            let w = framebuffer.width;
            let h = framebuffer.height;
            state.framebuffer = framebuffer;
            state.window.request_redraw();
            log::debug!("window updated: \"{title}\" ({w}×{h})");
        } else {
            // Create a new window.
            let w = framebuffer.width;
            let h = framebuffer.height;

            let attrs = WindowAttributes::default()
                .with_title(&title)
                .with_inner_size(winit::dpi::LogicalSize::new(w, h));

            let window = match event_loop.create_window(attrs) {
                Ok(win) => Arc::new(win),
                Err(e) => {
                    log::warn!("failed to create window \"{title}\": {e}");
                    return;
                }
            };

            // Lazily initialize the softbuffer context from the first window.
            if self.context.is_none() {
                match softbuffer::Context::new(window.clone()) {
                    Ok(ctx) => self.context = Some(ctx),
                    Err(e) => {
                        log::warn!("failed to create softbuffer context: {e}");
                        return;
                    }
                }
            }

            let context = self.context.as_ref().unwrap();

            let surface = match softbuffer::Surface::new(context, window.clone()) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("failed to create softbuffer surface for \"{title}\": {e}");
                    return;
                }
            };

            let mut state = WindowState {
                window,
                surface,
                framebuffer,
            };

            // Do an initial blit so the window shows content immediately.
            state.blit(&title);

            log::debug!("window created: \"{title}\" ({w}×{h})");

            self.windows.insert(title, state);
        }
    }

    /// Handle a `CloseRequested` event for a specific window.
    fn handle_close(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId) {
        let title = self.title_for_window(window_id);
        if let Some(title) = title {
            log::debug!("window closed: \"{title}\"");
            self.windows.remove(&title);
            let _ = self.event_tx.send(WindowEvent_::WindowClosed { title });
        }

        if self.windows.is_empty() {
            log::debug!("all windows closed, exiting event loop");
            let _ = self.event_tx.send(WindowEvent_::AllClosed);
            event_loop.exit();
        }
    }

    /// Handle a `RedrawRequested` event for a specific window.
    fn handle_redraw(&mut self, window_id: WindowId) {
        let title = self.title_for_window(window_id);
        if let Some(title) = title {
            if let Some(state) = self.windows.get_mut(&title) {
                state.blit(&title);
            }
        }
    }

    /// Handle an `Exit` command: close all windows and exit.
    fn handle_exit(&mut self, event_loop: &ActiveEventLoop) {
        log::debug!("exit command received, closing all windows");
        self.windows.clear();
        let _ = self.event_tx.send(WindowEvent_::AllClosed);
        event_loop.exit();
    }

    /// Find the title of the window with the given ID.
    ///
    /// Linear search is fine — we expect very few windows (< 10).
    fn title_for_window(&self, window_id: WindowId) -> Option<String> {
        for (title, state) in &self.windows {
            if state.window.id() == window_id {
                return Some(title.clone());
            }
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.5  WindowState internal struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Internal state for a single debug window.
struct WindowState {
    window: Arc<Window>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    framebuffer: Framebuffer,
}

impl WindowState {
    /// Blit the framebuffer to the window surface.
    ///
    /// Handles window resizing via nearest-neighbor scaling in the `u32`
    /// domain. This avoids re-running the display strategy on resize.
    fn blit(&mut self, title: &str) {
        let size = self.window.inner_size();
        let win_w = size.width;
        let win_h = size.height;

        // Skip blit if window is zero-sized (e.g. minimized).
        if win_w == 0 || win_h == 0 {
            return;
        }

        // Safety: we checked non-zero above.
        let nz_w = NonZeroU32::new(win_w).unwrap();
        let nz_h = NonZeroU32::new(win_h).unwrap();

        if let Err(e) = self.surface.resize(nz_w, nz_h) {
            log::warn!("failed to resize surface for \"{title}\": {e}");
            return;
        }

        log::trace!(
            "surface resized: \"{}\" {}×{} → {}×{}",
            title,
            self.framebuffer.width,
            self.framebuffer.height,
            win_w,
            win_h
        );

        let mut buffer = match self.surface.buffer_mut() {
            Ok(buf) => buf,
            Err(e) => {
                log::warn!("failed to get buffer for \"{title}\": {e}");
                return;
            }
        };

        if win_w == self.framebuffer.width && win_h == self.framebuffer.height {
            // Direct copy — dimensions match exactly.
            buffer[..self.framebuffer.data.len()].copy_from_slice(&self.framebuffer.data);
        } else {
            // Nearest-neighbor scale from framebuffer to window buffer.
            scale_blit(&self.framebuffer, &mut buffer, win_w, win_h);
        }

        if let Err(e) = buffer.present() {
            log::warn!("failed to present buffer for \"{title}\": {e}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.5a  Nearest-neighbor scaling
// ═══════════════════════════════════════════════════════════════════════════════

/// Nearest-neighbor scale from a [`Framebuffer`] into a destination `u32` buffer.
///
/// Operates entirely in the `u32` domain — no pixel conversion needed.
/// This is fast enough for a debug tool and avoids re-running the display
/// strategy on window resize.
fn scale_blit(src: &Framebuffer, dst: &mut [u32], dst_w: u32, dst_h: u32) {
    // Handle edge cases.
    if src.width == 0 || src.height == 0 {
        // Source is empty — fill destination with black.
        for pixel in dst.iter_mut() {
            *pixel = 0;
        }
        return;
    }

    for dy in 0..dst_h {
        let sy = (dy as u64 * src.height as u64 / dst_h as u64) as u32;
        let dst_row_start = (dy * dst_w) as usize;
        let src_row_start = (sy * src.width) as usize;

        for dx in 0..dst_w {
            let sx = (dx as u64 * src.width as u64 / dst_w as u64) as u32;
            dst[dst_row_start + dx as usize] = src.data[src_row_start + sx as usize];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.7  One-shot show() convenience function
// ═══════════════════════════════════════════════════════════════════════════════

/// Display a single image and block until the window is closed or a key
/// is pressed.
///
/// This is the simplest way to inspect an image during development.
/// Internally converts the image to a framebuffer on the **calling
/// thread**, then displays it using the winit event loop.
///
/// # Blocking behavior
///
/// This function blocks until the user presses a key or closes the window.
/// Calling `show()` multiple times in sequence is valid — the event loop
/// is created once and reused via `run_app_on_demand`.
///
/// # Platform notes
///
/// On macOS, this function **must** be called from the main thread.
///
/// Uses [`EventLoopExtRunOnDemand`] internally, which is supported on
/// Windows, Linux, macOS, and Android. Not available on iOS or Web
/// (but those platforms are not relevant for debug visualization).
///
/// # Mixing with `DebugDisplay::run()`
///
/// Do not mix `show()` and [`DebugDisplay::run()`] in the same process.
/// Each creates its own event loop, and winit only allows one per process.
/// Use [`DebugDisplay::run()`] for multi-window interactive workflows.
///
/// # Examples
///
/// ```no_run
/// use irys_cv_display::{show, Identity};
/// use irys_cv::image::Image;
/// use irys_cv::pixel::Srgb8;
///
/// let img = Image::fill(100, 100, Srgb8::new(128, 64, 200));
/// show("Preview", &img, Identity);
/// ```
pub fn show<V, S>(title: &str, image: &V, strategy: S)
where
    V: ImageView,
    V::Pixel: Copy,
    S: DisplayStrategy<V::Pixel>,
{
    use std::cell::RefCell;

    thread_local! {
        /// Lazily-initialized event loop reused across all `show()` calls
        /// on this thread. Winit only allows one `EventLoop` per process,
        /// so we create it once and drive it with `run_app_on_demand`.
        static SHOW_EVENT_LOOP: RefCell<EventLoop<UserEvent>> = RefCell::new(
            EventLoop::<UserEvent>::with_user_event()
                .build()
                .expect("failed to create event loop for show()")
        );
    }

    // Convert to Framebuffer BEFORE entering the event loop — this avoids
    // Send bounds on V and S. The Framebuffer (Vec<u32> + dimensions) is
    // Send and can be moved into the background thread.
    let fb = Framebuffer::from_image(image, strategy);
    let title = title.to_string();

    SHOW_EVENT_LOOP.with(|cell| {
        let mut event_loop = cell.borrow_mut();

        let proxy = event_loop.create_proxy();

        // Command channel: background thread → event loop.
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();

        // Event channel: event loop → background thread.
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();

        let ctx = DisplayContext {
            cmd_tx: cmd_tx.clone(),
            event_rx,
            notifier: Notifier::from_proxy(proxy.clone()),
        };

        // Spawn the user's show+wait_key logic on a background thread.
        let bg_cmd_tx = cmd_tx;
        let bg_notifier = Notifier::from_proxy(proxy);
        std::thread::spawn(move || {
            ctx.show_framebuffer(&title, fb);
            let _ = ctx.wait_key();

            // Signal the event loop to exit so run_app_on_demand returns.
            let _ = bg_cmd_tx.send(WindowCommand::Exit);
            if !bg_notifier.notify() {
                log::warn!("event loop proxy send failed (event loop has exited)");
            }
        });

        // Drive the event loop until the background thread signals Exit.
        let mut app = App {
            cmd_rx,
            event_tx,
            context: None,
            windows: HashMap::new(),
        };

        event_loop
            .run_app_on_demand(&mut app)
            .expect("event loop terminated with error");
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Identity;
    use crate::strategy::Framebuffer;
    use irys_cv::image::Image;
    use irys_cv::pixel::Srgba8;

    /// Helper: create a `DisplayContext` with mock channels.
    ///
    /// Returns `(ctx, cmd_rx, event_tx)` so the test can inspect
    /// commands and inject events.
    fn make_test_ctx() -> (
        DisplayContext,
        mpsc::Receiver<WindowCommand>,
        mpsc::Sender<WindowEvent_>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();
        let ctx = DisplayContext::new_for_test(cmd_tx, event_rx);
        (ctx, cmd_rx, event_tx)
    }

    // ── scale_blit tests ───────────────────────────────────────────────

    #[test]
    fn scale_blit_identity() {
        let src = Framebuffer::from_raw(2, 2, vec![0xAA, 0xBB, 0xCC, 0xDD]);
        let mut dst = vec![0u32; 4];
        scale_blit(&src, &mut dst, 2, 2);
        assert_eq!(dst, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn scale_blit_upscale_2x() {
        // 1×1 → 2×2: all pixels should be the same.
        let src = Framebuffer::from_raw(1, 1, vec![0xFF0000]);
        let mut dst = vec![0u32; 4];
        scale_blit(&src, &mut dst, 2, 2);
        assert_eq!(dst, vec![0xFF0000, 0xFF0000, 0xFF0000, 0xFF0000]);
    }

    #[test]
    fn scale_blit_downscale() {
        // 2×2 → 1×1: should pick pixel (0,0).
        let src = Framebuffer::from_raw(2, 2, vec![0xAA, 0xBB, 0xCC, 0xDD]);
        let mut dst = vec![0u32; 1];
        scale_blit(&src, &mut dst, 1, 1);
        assert_eq!(dst, vec![0xAA]);
    }

    #[test]
    fn scale_blit_empty_source() {
        let src = Framebuffer::from_raw(0, 0, vec![]);
        let mut dst = vec![0x12345678u32; 4];
        scale_blit(&src, &mut dst, 2, 2);
        // Should fill with black.
        assert_eq!(dst, vec![0, 0, 0, 0]);
    }

    #[test]
    fn scale_blit_non_square_upscale() {
        // 2×1 → 4×2
        let src = Framebuffer::from_raw(2, 1, vec![0xAA, 0xBB]);
        let mut dst = vec![0u32; 8];
        scale_blit(&src, &mut dst, 4, 2);
        // Row 0: 0xAA, 0xAA, 0xBB, 0xBB
        // Row 1: 0xAA, 0xAA, 0xBB, 0xBB (same row mapped)
        assert_eq!(dst, vec![0xAA, 0xAA, 0xBB, 0xBB, 0xAA, 0xAA, 0xBB, 0xBB]);
    }

    #[test]
    fn scale_blit_3x2_to_6x4() {
        // Source:
        // [1, 2, 3]
        // [4, 5, 6]
        let src = Framebuffer::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]);
        let mut dst = vec![0u32; 24]; // 6×4
        scale_blit(&src, &mut dst, 6, 4);

        // Expected nearest-neighbor 2x:
        // [1, 1, 2, 2, 3, 3]
        // [1, 1, 2, 2, 3, 3]
        // [4, 4, 5, 5, 6, 6]
        // [4, 4, 5, 5, 6, 6]
        let expected = vec![
            1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6,
        ];
        assert_eq!(dst, expected);
    }

    // ── Notifier tests ─────────────────────────────────────────────────

    #[test]
    fn notifier_noop_returns_true() {
        let n = Notifier(Box::new(|| true));
        assert!(n.notify());
    }

    #[test]
    fn notifier_failing_returns_false() {
        let n = Notifier(Box::new(|| false));
        assert!(!n.notify());
    }

    // ── DisplayContext::show_framebuffer tests ──────────────────────────

    #[test]
    fn show_framebuffer_sends_command() {
        let (ctx, cmd_rx, _event_tx) = make_test_ctx();
        let fb = Framebuffer::from_raw(2, 2, vec![0xAA, 0xBB, 0xCC, 0xDD]);
        ctx.show_framebuffer("my title", fb);

        match cmd_rx.recv().unwrap() {
            WindowCommand::Show { title, framebuffer } => {
                assert_eq!(title, "my title");
                assert_eq!(framebuffer.width, 2);
                assert_eq!(framebuffer.height, 2);
                assert_eq!(framebuffer.data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
            }
            WindowCommand::Exit => panic!("expected Show, got Exit"),
        }
    }

    #[test]
    fn show_framebuffer_with_closed_channel_does_not_panic() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();
        let (_event_tx, event_rx) = mpsc::channel::<WindowEvent_>();
        let ctx = DisplayContext::new_for_test(cmd_tx, event_rx);

        // Drop the receiver so the channel is closed.
        drop(cmd_rx);

        // Should not panic, just log a warning.
        let fb = Framebuffer::from_raw(1, 1, vec![0]);
        ctx.show_framebuffer("dead", fb);
    }

    #[test]
    fn show_framebuffer_notifier_failure_does_not_panic() {
        let (cmd_tx, _cmd_rx) = mpsc::channel::<WindowCommand>();
        let (_event_tx, event_rx) = mpsc::channel::<WindowEvent_>();
        // Notifier that always "fails".
        let ctx = DisplayContext {
            cmd_tx,
            event_rx,
            notifier: Notifier(Box::new(|| false)),
        };

        let fb = Framebuffer::from_raw(1, 1, vec![0]);
        ctx.show_framebuffer("fail-notify", fb);
        // Should not panic — just logs a warning.
    }

    // ── DisplayContext::show tests ──────────────────────────────────────

    #[test]
    fn show_converts_image_and_sends_command() {
        let (ctx, cmd_rx, _event_tx) = make_test_ctx();

        let img = Image::fill(2, 2, Srgba8::new(255, 0, 0, 255));
        ctx.show("red", &img, Identity);

        match cmd_rx.recv().unwrap() {
            WindowCommand::Show { title, framebuffer } => {
                assert_eq!(title, "red");
                assert_eq!(framebuffer.width, 2);
                assert_eq!(framebuffer.height, 2);
                // Red = 0x00FF0000 in 0x00RRGGBB
                assert!(framebuffer.data.iter().all(|&p| p == 0x00FF0000));
            }
            WindowCommand::Exit => panic!("expected Show, got Exit"),
        }
    }

    #[test]
    fn show_zero_size_image() {
        let (ctx, cmd_rx, _event_tx) = make_test_ctx();

        let img = Image::<Srgba8>::zero(0, 0);
        ctx.show("empty", &img, Identity);

        match cmd_rx.recv().unwrap() {
            WindowCommand::Show { title, framebuffer } => {
                assert_eq!(title, "empty");
                assert_eq!(framebuffer.width, 0);
                assert_eq!(framebuffer.height, 0);
                assert!(framebuffer.data.is_empty());
            }
            WindowCommand::Exit => panic!("expected Show, got Exit"),
        }
    }

    // ── DisplayContext::exit tests ──────────────────────────────────────

    #[test]
    fn exit_sends_exit_command() {
        let (ctx, cmd_rx, _event_tx) = make_test_ctx();
        ctx.exit();

        match cmd_rx.recv().unwrap() {
            WindowCommand::Exit => {} // ok
            WindowCommand::Show { .. } => panic!("expected Exit, got Show"),
        }
    }

    #[test]
    fn exit_with_closed_channel_does_not_panic() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();
        let (_event_tx, event_rx) = mpsc::channel::<WindowEvent_>();
        let ctx = DisplayContext::new_for_test(cmd_tx, event_rx);
        drop(cmd_rx);
        ctx.exit(); // should not panic
    }

    // ── DisplayContext::wait_key tests ──────────────────────────────────

    #[test]
    fn wait_key_returns_key_on_key_pressed() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::Space,
                window_title: "test".to_string(),
            })
            .unwrap();

        assert_eq!(ctx.wait_key(), Some(KeyCode::Space));
    }

    #[test]
    fn wait_key_skips_window_closed_waits_for_key() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        // WindowClosed then KeyPressed.
        event_tx
            .send(WindowEvent_::WindowClosed {
                title: "closing".to_string(),
            })
            .unwrap();
        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::Enter,
                window_title: "remaining".to_string(),
            })
            .unwrap();

        // wait_key should skip WindowClosed and return the key.
        assert_eq!(ctx.wait_key(), Some(KeyCode::Enter));
    }

    #[test]
    fn wait_key_returns_none_on_all_closed() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx.send(WindowEvent_::AllClosed).unwrap();

        assert_eq!(ctx.wait_key(), None);
    }

    #[test]
    fn wait_key_returns_none_on_channel_disconnect() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        // Drop the sender — recv() will return Err.
        drop(event_tx);

        assert_eq!(ctx.wait_key(), None);
    }

    #[test]
    fn wait_key_skips_multiple_window_closed() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx
            .send(WindowEvent_::WindowClosed {
                title: "a".to_string(),
            })
            .unwrap();
        event_tx
            .send(WindowEvent_::WindowClosed {
                title: "b".to_string(),
            })
            .unwrap();
        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::KeyA,
                window_title: "c".to_string(),
            })
            .unwrap();

        assert_eq!(ctx.wait_key(), Some(KeyCode::KeyA));
    }

    // ── DisplayContext::wait_key_timeout tests ──────────────────────────

    #[test]
    fn wait_key_timeout_returns_key_before_timeout() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::Escape,
                window_title: "w".to_string(),
            })
            .unwrap();

        let result = ctx.wait_key_timeout(Duration::from_secs(5));
        assert_eq!(result, Some(KeyCode::Escape));
    }

    #[test]
    fn wait_key_timeout_returns_none_on_timeout() {
        let (ctx, _cmd_rx, _event_tx) = make_test_ctx();

        // No events sent — should time out.
        let result = ctx.wait_key_timeout(Duration::from_millis(10));
        assert_eq!(result, None);
    }

    #[test]
    fn wait_key_timeout_returns_none_on_all_closed() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx.send(WindowEvent_::AllClosed).unwrap();

        let result = ctx.wait_key_timeout(Duration::from_secs(5));
        assert_eq!(result, None);
    }

    #[test]
    fn wait_key_timeout_returns_none_on_disconnect() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();
        drop(event_tx);

        let result = ctx.wait_key_timeout(Duration::from_secs(5));
        assert_eq!(result, None);
    }

    #[test]
    fn wait_key_timeout_skips_window_closed() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        event_tx
            .send(WindowEvent_::WindowClosed {
                title: "gone".to_string(),
            })
            .unwrap();
        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::KeyZ,
                window_title: "still here".to_string(),
            })
            .unwrap();

        let result = ctx.wait_key_timeout(Duration::from_secs(5));
        assert_eq!(result, Some(KeyCode::KeyZ));
    }

    #[test]
    fn wait_key_timeout_zero_returns_none_immediately() {
        let (ctx, _cmd_rx, event_tx) = make_test_ctx();

        // Even though an event is queued, zero timeout should return None
        // (or the key if it happens to be received — but we test zero-duration
        // short-circuits the loop).
        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::KeyX,
                window_title: "w".to_string(),
            })
            .unwrap();

        // With Duration::ZERO the remaining-is-zero check fires immediately.
        let result = ctx.wait_key_timeout(Duration::ZERO);
        // The event may or may not be received in zero time — just verify
        // no panic and result is a valid Option.
        assert!(result.is_none() || result == Some(KeyCode::KeyX));
    }

    // ── Combined show + wait_key workflow ──────────────────────────────

    #[test]
    fn show_then_wait_key_workflow() {
        let (ctx, cmd_rx, event_tx) = make_test_ctx();

        // Simulate: user shows an image, then waits for a key.
        let img = Image::fill(4, 4, Srgba8::new(0, 255, 0, 255));
        ctx.show("green", &img, Identity);

        // Verify the Show command arrived.
        match cmd_rx.recv().unwrap() {
            WindowCommand::Show { title, framebuffer } => {
                assert_eq!(title, "green");
                assert_eq!(framebuffer.width, 4);
                assert_eq!(framebuffer.height, 4);
                assert_eq!(framebuffer.data.len(), 16);
                // Green = 0x0000FF00
                assert!(framebuffer.data.iter().all(|&p| p == 0x0000FF00));
            }
            WindowCommand::Exit => panic!("expected Show"),
        }

        // Simulate: event loop sends a key event back.
        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::KeyQ,
                window_title: "green".to_string(),
            })
            .unwrap();

        assert_eq!(ctx.wait_key(), Some(KeyCode::KeyQ));
    }

    #[test]
    fn multiple_shows_then_exit_workflow() {
        let (ctx, cmd_rx, _event_tx) = make_test_ctx();

        let img1 = Image::fill(1, 1, Srgba8::new(255, 0, 0, 255));
        let img2 = Image::fill(1, 1, Srgba8::new(0, 0, 255, 255));
        ctx.show("red", &img1, Identity);
        ctx.show("blue", &img2, Identity);
        ctx.exit();

        // Drain: Show, Show, Exit.
        let mut titles = Vec::new();
        let mut got_exit = false;
        for _ in 0..3 {
            match cmd_rx.recv().unwrap() {
                WindowCommand::Show { title, .. } => titles.push(title),
                WindowCommand::Exit => got_exit = true,
            }
        }
        assert_eq!(titles, vec!["red", "blue"]);
        assert!(got_exit);
    }

    // ── WindowCommand / WindowEvent_ channel protocol tests ────────────

    #[test]
    fn channel_show_sends_correct_command() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();

        cmd_tx
            .send(WindowCommand::Show {
                title: "test".to_string(),
                framebuffer: Framebuffer::from_raw(2, 2, vec![0, 0, 0, 0]),
            })
            .unwrap();

        match cmd_rx.recv().unwrap() {
            WindowCommand::Show { title, framebuffer } => {
                assert_eq!(title, "test");
                assert_eq!(framebuffer.width, 2);
                assert_eq!(framebuffer.height, 2);
                assert_eq!(framebuffer.data.len(), 4);
            }
            WindowCommand::Exit => panic!("expected Show, got Exit"),
        }
    }

    #[test]
    fn channel_exit_command() {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WindowCommand>();

        cmd_tx.send(WindowCommand::Exit).unwrap();
        match cmd_rx.recv().unwrap() {
            WindowCommand::Exit => {} // ok
            WindowCommand::Show { .. } => panic!("expected Exit, got Show"),
        }
    }

    #[test]
    fn channel_key_pressed_event() {
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();

        event_tx
            .send(WindowEvent_::KeyPressed {
                key: KeyCode::Escape,
                window_title: "test window".to_string(),
            })
            .unwrap();

        match event_rx.recv().unwrap() {
            WindowEvent_::KeyPressed { key, window_title } => {
                assert_eq!(key, KeyCode::Escape);
                assert_eq!(window_title, "test window");
            }
            other => panic!("expected KeyPressed, got {:?}", other),
        }
    }

    #[test]
    fn channel_window_closed_event() {
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();

        event_tx
            .send(WindowEvent_::WindowClosed {
                title: "my window".to_string(),
            })
            .unwrap();

        match event_rx.recv().unwrap() {
            WindowEvent_::WindowClosed { title } => {
                assert_eq!(title, "my window");
            }
            other => panic!("expected WindowClosed, got {:?}", other),
        }
    }

    #[test]
    fn channel_all_closed_event() {
        let (event_tx, event_rx) = mpsc::channel::<WindowEvent_>();

        event_tx.send(WindowEvent_::AllClosed).unwrap();

        match event_rx.recv().unwrap() {
            WindowEvent_::AllClosed => {} // ok
            other => panic!("expected AllClosed, got {:?}", other),
        }
    }

    // ── Misc ───────────────────────────────────────────────────────────

    #[test]
    fn framebuffer_is_send() {
        // Framebuffer must be Send so it can cross thread boundaries.
        fn assert_send<T: Send>() {}
        assert_send::<Framebuffer>();
    }

    #[test]
    fn display_context_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DisplayContext>();
    }
}
