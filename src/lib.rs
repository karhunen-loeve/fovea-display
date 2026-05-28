//! # fovea-display
//!
//! Display and debug visualization for fovea images.
//!
//! This crate provides:
//! - **Display traits** — strategy-based conversion of any [`ImageView`](fovea::image::ImageView)
//!   to a displayable format.
//! - **GPU integration traits** — texture format descriptors and byte access for rendering
//!   pipelines (always available, no GPU dependency).
//! - **Debug window** (feature `debug-window`) — an OpenCV-like `imshow` window for quick
//!   visualization during development. Not intended for production use.
//!
//! ## Feature flags
//!
//! | Feature        | Dependencies          | Description                                |
//! |----------------|-----------------------|--------------------------------------------|
//! | `debug-window` | `winit`, `softbuffer`  | [`DebugDisplay`] / [`show()`] support      |
//!
//! ## Quick start (debug window)
//!
//! The simplest way to inspect an image during development:
//!
//! ```no_run
//! use fovea::image::Image;
//! use fovea::pixel::Srgb8;
//! use fovea_display::{show, Identity};
//!
//! let img = Image::fill(320, 240, Srgb8::new(128, 64, 200));
//! show("Preview", &img, Identity);
//! ```
//!
//! For multiple windows or interactive workflows, use [`DebugDisplay::run()`]:
//!
//! ```no_run
//! use fovea::image::Image;
//! use fovea::pixel::Srgba8;
//! use fovea_display::{DebugDisplay, Identity};
//!
//! DebugDisplay::run(|ctx| {
//!     let img = Image::fill(320, 240, Srgba8::new(255, 0, 0, 255));
//!     ctx.show("Red", &img, Identity);
//!
//!     let img2 = Image::fill(320, 240, Srgba8::new(0, 255, 0, 255));
//!     ctx.show("Green", &img2, Identity);
//!
//!     ctx.wait_key(); // blocks until a key is pressed or all windows close
//! });
//! ```
//!
//! ## Display strategies
//!
//! Every pixel type requires an explicit [`DisplayStrategy`] to become
//! displayable. There is no silent default — `Mono16` cannot be shown
//! without choosing [`AutoContrast`], [`FixedRange`], or another strategy.
//!
//! | Strategy            | Use case                                         |
//! |---------------------|--------------------------------------------------|
//! | [`Identity`]        | sRGB types that are already display-ready         |
//! | [`LinearToDisplay`] | Linear-light types (applies sRGB gamma encoding)  |
//! | [`AutoContrast`]    | Scan-based contrast stretching for HDR / deep     |
//! | [`FixedRange`]      | User-specified value range mapping                |
//!
//! ## Logging
//!
//! This crate uses the [`log`] facade for diagnostic messages. Attach a
//! logger (e.g. [`env_logger`](https://docs.rs/env_logger)) in your
//! application to see them. When no logger is attached, all log calls
//! compile to no-ops.
//!
//! ## Design principles
//!
//! - **Conversions are named.** Every pixel type requires an explicit
//!   [`DisplayStrategy`] to become displayable. This follows fovea's
//!   core philosophy.
//!
//! - **`ImageView` is the display bound.** The [`show()`] function accepts
//!   any `impl ImageView` — owned images, ROIs, tiled views, custom
//!   backends. This is the minimum trait bound needed for pixel-by-pixel
//!   conversion.
//!
//! - **GPU upload uses `PlainImage`.** The [`TextureSource`] trait requires
//!   contiguous byte access, a stronger bound than display. An ROI can be
//!   displayed but not GPU-uploaded without copying to contiguous storage
//!   first.
//!
//! - **sRGB correctness via types.** [`Identity`] only accepts `Srgb*`
//!   types. Linear [`Rgb8`](fovea::pixel::Rgb8) requires
//!   [`LinearToDisplay`]. The compiler catches double-gamma errors.

mod pixel;
mod strategy;
mod texture;

#[cfg(feature = "debug-window")]
mod debug_window;

#[cfg(feature = "debug-window")]
mod debug_histogram;

// ── Public re-exports ───────────────────────────────────────────────────────

pub use pixel::{DisplayPixel, GpuPixel, TextureFormat};
pub use strategy::{AutoContrast, DisplayStrategy, FixedRange, Identity, LinearToDisplay};
pub use texture::TextureSource;

#[cfg(feature = "debug-window")]
pub use debug_window::{DebugDisplay, DisplayContext, show};

#[cfg(feature = "debug-window")]
pub use debug_histogram::{
    HistogramLayer, HistogramPlotOptions, HistogramRenderOptions, debug_histogram,
    debug_histogram_layers, debug_histogram_with, render_histogram, render_histogram_layers,
};
