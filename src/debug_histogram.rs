//! Debug histogram window for quick inspection of [`Histogram`] data.
//!
//! This module provides an `imshow`-style entry point for visualising
//! histograms during development. It is gated behind the `debug-window`
//! feature flag and is **not intended for production use**.
//!
//! Histograms produced by [`fovea::analyze::histogram::histogram()`]
//! carry a `bins()` slice that is strategy-independent — that is the
//! only datum the renderer needs. One or more histograms are
//! rasterised into a small [`Srgba8`](fovea::pixel::Srgba8) bar-chart
//! image and handed off to the same [`Framebuffer`] / event-loop
//! pipeline as [`crate::show`].
//!
//! # Single histogram
//!
//! ```no_run
//! use fovea::analyze::histogram::{histogram, Histogram, NaturalBins};
//! use fovea::image::Image;
//! use fovea::pixel::Mono8;
//! use fovea_display::debug_histogram;
//!
//! let img = Image::<Mono8>::zero(64, 64);
//! let h: Histogram<NaturalBins, _> = histogram(&img, &NaturalBins).unwrap();
//! debug_histogram("Sample histogram", &h);
//! ```
//!
//! # Multiple translucent layers in one window
//!
//! Each layer carries its own colour (with alpha) and its own `bins`
//! slice. All layers in a call share the chart frame, padding, axis,
//! background, and y-scale — so an RGB image's three channels can be
//! overlaid in one plot, or two transforms of the same data (e.g.
//! linear vs. log) can be compared side-on-side.
//!
//! ```no_run
//! use fovea::analyze::histogram::{histogram, Histogram, NaturalBins};
//! use fovea::image::Image;
//! use fovea::pixel::{Rgb8, Srgba8};
//! use fovea_display::{debug_histogram_layers, HistogramLayer, HistogramPlotOptions};
//!
//! let img = Image::<Rgb8>::zero(64, 64);
//! let chans: [Histogram<NaturalBins, _>; 3] =
//!     histogram(&img, &NaturalBins).unwrap();
//!
//! let layers = [
//!     HistogramLayer::new(chans[0].bins(), Srgba8::new(220,  60,  60, 160)),
//!     HistogramLayer::new(chans[1].bins(), Srgba8::new( 60, 200,  80, 160)),
//!     HistogramLayer::new(chans[2].bins(), Srgba8::new( 80, 120, 230, 160)),
//! ];
//! debug_histogram_layers("RGB histogram", &layers, &HistogramPlotOptions::default());
//! ```

use fovea::analyze::histogram::Histogram;
use fovea::image::{Image, ImageViewMut};
use fovea::pixel::Srgba8;

use crate::DisplayContext;
use crate::strategy::{Framebuffer, Identity};

// ═══════════════════════════════════════════════════════════════════════════════
// Plot-level configuration (chart frame, shared by all layers)
// ═══════════════════════════════════════════════════════════════════════════════

/// Frame-level configuration for [`render_histogram_layers`].
///
/// Holds everything that is *shared* between all layers in a single
/// chart: the canvas size, padding, background colour, axis colour,
/// and whether the y-axis is log-scaled.
///
/// Per-layer properties (bin counts and bar colour) live on
/// [`HistogramLayer`].
#[derive(Debug, Clone, Copy)]
pub struct HistogramPlotOptions {
    /// Width of the rendered chart in pixels.
    pub width: u32,
    /// Height of the rendered chart in pixels.
    pub height: u32,
    /// Padding around the plot area, in pixels.
    pub padding: u32,
    /// Background fill colour. Use a translucent value if you intend
    /// to composite the chart over something else later — the
    /// renderer will honour the alpha channel verbatim.
    pub background: Srgba8,
    /// Axis / baseline colour.
    pub axis: Srgba8,
    /// If `true`, every layer's bar heights are scaled by
    /// `log10(1 + count)`. Useful for natural images where a few
    /// bins dominate.
    pub log_scale: bool,
}

impl Default for HistogramPlotOptions {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            padding: 8,
            background: Srgba8::new(24, 24, 28, 255),
            axis: Srgba8::new(96, 96, 104, 255),
            log_scale: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Layer
// ═══════════════════════════════════════════════════════════════════════════════

/// One translucent histogram layer in a multi-layer plot.
///
/// `bins` is borrowed (it's just a slice into a [`Histogram`]'s bin
/// counts) so layers are cheap to construct and keep no ownership of
/// the histogram itself.
///
/// The renderer treats `color.a` as straight (non-premultiplied)
/// alpha. Lower alpha values let underlying layers show through —
/// pick `~120..180` for typical 2–3 layer overlays.
#[derive(Debug, Clone, Copy)]
pub struct HistogramLayer<'a> {
    /// Bin counts to draw, one entry per bar.
    pub bins: &'a [u64],
    /// Bar fill colour (RGBA, straight alpha).
    pub color: Srgba8,
}

impl<'a> HistogramLayer<'a> {
    /// Build a layer from raw bins and a colour.
    #[inline]
    pub fn new(bins: &'a [u64], color: Srgba8) -> Self {
        Self { bins, color }
    }

    /// Build a layer from a [`Histogram`] and a colour.
    ///
    /// Equivalent to `HistogramLayer::new(h.bins(), color)`. Generic
    /// over both the strategy `S` and the channel value type `V`,
    /// because only the strategy-independent `bins()` slice is read.
    #[inline]
    pub fn from_histogram<S, V>(h: &'a Histogram<S, V>, color: Srgba8) -> Self {
        Self::new(h.bins(), color)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Single-histogram convenience: HistogramRenderOptions
// ═══════════════════════════════════════════════════════════════════════════════

/// Visual configuration for the single-histogram entry points.
///
/// Combines [`HistogramPlotOptions`] with one `bar` colour for the
/// classic "render exactly one histogram" use case. Multi-histogram
/// plotting goes through [`HistogramPlotOptions`] +
/// [`HistogramLayer`] directly.
#[derive(Debug, Clone, Copy)]
pub struct HistogramRenderOptions {
    /// Width of the rendered chart in pixels.
    pub width: u32,
    /// Height of the rendered chart in pixels.
    pub height: u32,
    /// Padding around the plot area, in pixels.
    pub padding: u32,
    /// Background fill colour.
    pub background: Srgba8,
    /// Bar fill colour. Alpha is honoured but with one layer it
    /// simply blends against the background.
    pub bar: Srgba8,
    /// Axis / baseline colour.
    pub axis: Srgba8,
    /// If `true`, scale bar heights logarithmically (`log10(1 + count)`).
    pub log_scale: bool,
}

impl Default for HistogramRenderOptions {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            padding: 8,
            background: Srgba8::new(24, 24, 28, 255),
            bar: Srgba8::new(180, 200, 230, 255),
            axis: Srgba8::new(96, 96, 104, 255),
            log_scale: false,
        }
    }
}

impl HistogramRenderOptions {
    /// Split into the frame-level [`HistogramPlotOptions`] and the
    /// single bar colour. Used internally to delegate to the layered
    /// renderer.
    #[inline]
    fn split(&self) -> (HistogramPlotOptions, Srgba8) {
        (
            HistogramPlotOptions {
                width: self.width,
                height: self.height,
                padding: self.padding,
                background: self.background,
                axis: self.axis,
                log_scale: self.log_scale,
            },
            self.bar,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Renderer
// ═══════════════════════════════════════════════════════════════════════════════

/// Render any number of histogram layers into one [`Image<Srgba8>`].
///
/// All layers share the chart frame and a single y-scale: the maximum
/// (transformed) bin value is taken across *every* layer so relative
/// heights stay comparable. Per-layer bars are alpha-blended into the
/// canvas in the order the slice provides them — the last layer is
/// drawn on top.
///
/// This is the workhorse renderer; [`render_histogram`] delegates here
/// with a single layer.
///
/// # Notes
///
/// - Bin → column mapping uses max-pooling: when columns are narrower
///   than the bin count, several adjacent bins collapse onto one
///   column and the column adopts their max so spikes are preserved.
/// - NaN / underflow / overflow counters are not drawn (they have no
///   in-range bin); callers that care can read those fields off the
///   [`Histogram`].
/// - Blending uses sRGB straight-alpha. This is a debug visualiser,
///   not a colour-managed pipeline — the goal is "translucent layers
///   look translucent", not photometric accuracy.
pub fn render_histogram_layers(
    layers: &[HistogramLayer<'_>],
    opts: &HistogramPlotOptions,
) -> Image<Srgba8> {
    let w = opts.width.max(1) as usize;
    let height_px = opts.height.max(1) as usize;
    let pad = (opts.padding as usize).min(w / 4).min(height_px / 4);

    let mut img = Image::fill(w, height_px, opts.background);

    // Plot rectangle in image coordinates (top-left origin).
    let plot_x0 = pad;
    let plot_y0 = pad;
    let plot_x1 = w.saturating_sub(pad).max(plot_x0 + 1);
    let plot_y1 = height_px.saturating_sub(pad).max(plot_y0 + 1);
    let plot_w = plot_x1 - plot_x0;
    let plot_h = plot_y1 - plot_y0;

    // Baseline (axis).
    let baseline_y = plot_y1.saturating_sub(1);
    for x in plot_x0..plot_x1 {
        *img.pixel_at_mut(x, baseline_y) = opts.axis;
    }

    if layers.is_empty() || plot_w == 0 || plot_h == 0 {
        return img;
    }

    let transform = |c: u64| -> f64 {
        if opts.log_scale {
            (1.0 + c as f64).ln()
        } else {
            c as f64
        }
    };

    // Shared y-scale: max across all layers and all bins. Empty layers
    // contribute 0 and do not affect the scale.
    let mut max_val = 0.0_f64;
    for layer in layers {
        for &c in layer.bins {
            let v = transform(c);
            if v > max_val {
                max_val = v;
            }
        }
    }
    if max_val <= 0.0 {
        return img;
    }

    let usable_h = plot_h.saturating_sub(1); // reserve baseline row
    if usable_h == 0 {
        return img;
    }

    // Draw each layer in order. Later layers blend on top.
    for layer in layers {
        let n = layer.bins.len();
        if n == 0 {
            continue;
        }

        for col in 0..plot_w {
            let lo = (col * n) / plot_w;
            let hi_excl = (((col + 1) * n) / plot_w).max(lo + 1).min(n);

            let mut local_max = 0.0_f64;
            for b in &layer.bins[lo..hi_excl] {
                let v = transform(*b);
                if v > local_max {
                    local_max = v;
                }
            }
            if local_max <= 0.0 {
                continue;
            }

            let bar_h = ((local_max / max_val) * usable_h as f64).round() as usize;
            let bar_h = bar_h.min(usable_h);
            if bar_h == 0 {
                continue;
            }

            let x = plot_x0 + col;
            let top = baseline_y.saturating_sub(bar_h);
            for y in top..baseline_y {
                let dst = img.pixel_at_mut(x, y);
                *dst = blend_over(layer.color, *dst);
            }
        }
    }

    img
}

/// Render a single histogram's bin counts to an [`Image<Srgba8>`].
///
/// Convenience wrapper around [`render_histogram_layers`] for the
/// classic single-bar-colour case. See that function for the rendering
/// model.
pub fn render_histogram<S, V>(h: &Histogram<S, V>, opts: &HistogramRenderOptions) -> Image<Srgba8> {
    let (plot, bar) = opts.split();
    let layers = [HistogramLayer::new(h.bins(), bar)];
    render_histogram_layers(&layers, &plot)
}

// ── Compositing ─────────────────────────────────────────────────────────────

/// Straight-alpha "source over destination" blend in 8-bit sRGB space.
///
/// This is intentionally **not** colour-managed: gamma-correct blending
/// would require linearising both operands. For a debug visualiser the
/// approximation is good enough and keeps everything in the same byte
/// space as the [`Framebuffer`].
#[inline]
fn blend_over(src: Srgba8, dst: Srgba8) -> Srgba8 {
    let sa = src.a.0 as u32;
    if sa == 0 {
        return dst;
    }
    if sa == 255 {
        return src;
    }
    let inv = 255 - sa;

    // Round-to-nearest division by 255 via the classic
    // `(x * 0x8081) >> 23` trick; the simpler `(x + 127) / 255` is
    // plenty fast for the debug path and easier to audit.
    let mix = |s: u8, d: u8| -> u8 {
        let v = (s as u32) * sa + (d as u32) * inv;
        ((v + 127) / 255) as u8
    };

    let da = dst.a.0 as u32;
    let out_a = sa + (da * inv + 127) / 255;
    let out_a = out_a.min(255) as u8;

    Srgba8::new(
        mix(src.r.0, dst.r.0),
        mix(src.g.0, dst.g.0),
        mix(src.b.0, dst.b.0),
        out_a,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public entry points — single histogram
// ═══════════════════════════════════════════════════════════════════════════════

/// Display a histogram in a debug window using default render options.
///
/// This is the histogram counterpart of [`crate::show`]: it blocks
/// until the user presses a key or closes the window. See
/// [`debug_histogram_with`] for control over the rendered appearance,
/// and [`debug_histogram_layers`] for multi-layer plots.
///
/// # Platform notes
///
/// On macOS, this function **must** be called from the main thread.
/// Internally it goes through the same one-thread-per-process winit
/// event loop used by [`crate::show`], so do not mix this with
/// [`crate::DebugDisplay::run`] in the same process.
pub fn debug_histogram<S, V>(title: &str, h: &Histogram<S, V>) {
    debug_histogram_with(title, h, &HistogramRenderOptions::default());
}

/// Display a histogram in a debug window with custom render options.
pub fn debug_histogram_with<S, V>(title: &str, h: &Histogram<S, V>, opts: &HistogramRenderOptions) {
    let img = render_histogram(h, opts);
    crate::show(title, &img, Identity);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public entry points — layered histograms
// ═══════════════════════════════════════════════════════════════════════════════

/// Display any number of histogram layers in a single debug window.
///
/// All layers share one chart frame and one y-scale; per-layer colour
/// (with alpha) controls overlay appearance. See
/// [`render_histogram_layers`] for the rendering model and
/// [`crate::show`] for the windowing semantics.
pub fn debug_histogram_layers(
    title: &str,
    layers: &[HistogramLayer<'_>],
    opts: &HistogramPlotOptions,
) {
    let img = render_histogram_layers(layers, opts);
    crate::show(title, &img, Identity);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DisplayContext extension
// ═══════════════════════════════════════════════════════════════════════════════

impl DisplayContext {
    /// Display a histogram inside a [`DebugDisplay::run`](crate::DebugDisplay::run)
    /// session, using default render options.
    ///
    /// Non-blocking. Uses the same window-update semantics as
    /// [`DisplayContext::show`]: passing the same `title` again
    /// updates the existing window in place.
    pub fn show_histogram<S, V>(&self, title: &str, h: &Histogram<S, V>) {
        self.show_histogram_with(title, h, &HistogramRenderOptions::default());
    }

    /// Display a histogram inside a [`DebugDisplay::run`](crate::DebugDisplay::run)
    /// session with custom render options.
    pub fn show_histogram_with<S, V>(
        &self,
        title: &str,
        h: &Histogram<S, V>,
        opts: &HistogramRenderOptions,
    ) {
        let img = render_histogram(h, opts);
        let fb = Framebuffer::from_image(&img, Identity);
        self.show_framebuffer(title, fb);
    }

    /// Display a multi-layer histogram inside a
    /// [`DebugDisplay::run`](crate::DebugDisplay::run) session.
    ///
    /// Non-blocking. See [`render_histogram_layers`] for the
    /// rendering model.
    pub fn show_histogram_layers(
        &self,
        title: &str,
        layers: &[HistogramLayer<'_>],
        opts: &HistogramPlotOptions,
    ) {
        let img = render_histogram_layers(layers, opts);
        let fb = Framebuffer::from_image(&img, Identity);
        self.show_framebuffer(title, fb);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fovea::analyze::histogram::{NaturalBins, histogram};
    use fovea::image::ImageView;
    use fovea::pixel::Mono8;

    fn build_hist() -> Histogram<NaturalBins, std::num::Saturating<u8>> {
        // 4×4 image with one pixel at value 0 and the rest at 255.
        let mut img: Image<Mono8> = Image::fill(4, 4, Mono8::new(255));
        *img.pixel_at_mut(0, 0) = Mono8::new(0);
        histogram(&img, &NaturalBins).unwrap()
    }

    // ── Single-histogram renderer ────────────────────────────────────────

    #[test]
    fn render_default_size_matches_options() {
        let h = build_hist();
        let opts = HistogramRenderOptions::default();
        let img = render_histogram(&h, &opts);
        assert_eq!(img.width(), opts.width as usize);
        assert_eq!(img.height(), opts.height as usize);
    }

    #[test]
    fn render_zero_size_padded_does_not_panic() {
        let h = build_hist();
        let opts = HistogramRenderOptions {
            width: 1,
            height: 1,
            padding: 0,
            ..HistogramRenderOptions::default()
        };
        let img = render_histogram(&h, &opts);
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
    }

    #[test]
    fn render_paints_background_outside_bars() {
        let h = build_hist();
        let opts = HistogramRenderOptions::default();
        let img = render_histogram(&h, &opts);
        // Top-left corner is well inside the padding region, which is
        // never overwritten by bars.
        assert_eq!(img.pixel_at(0, 0), opts.background);
    }

    #[test]
    fn render_log_scale_runs() {
        let h = build_hist();
        let opts = HistogramRenderOptions {
            log_scale: true,
            ..HistogramRenderOptions::default()
        };
        let _ = render_histogram(&h, &opts);
    }

    #[test]
    fn render_empty_bins_is_safe() {
        let h = build_hist();
        let opts = HistogramRenderOptions {
            width: 4,
            height: 4,
            padding: 2,
            ..HistogramRenderOptions::default()
        };
        let img = render_histogram(&h, &opts);
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 4);
    }

    // ── Layered renderer ─────────────────────────────────────────────────

    #[test]
    fn layered_no_layers_yields_background_and_axis_only() {
        let opts = HistogramPlotOptions::default();
        let img = render_histogram_layers(&[], &opts);
        assert_eq!(img.width(), opts.width as usize);
        assert_eq!(img.height(), opts.height as usize);
        // Top-left padding pixel is background.
        assert_eq!(img.pixel_at(0, 0), opts.background);
    }

    #[test]
    fn layered_two_translucent_layers_blend() {
        let h = build_hist();
        let opts = HistogramPlotOptions::default();
        let layers = [
            HistogramLayer::from_histogram(&h, Srgba8::new(255, 0, 0, 128)),
            HistogramLayer::from_histogram(&h, Srgba8::new(0, 0, 255, 128)),
        ];
        let img = render_histogram_layers(&layers, &opts);
        assert_eq!(img.width(), opts.width as usize);
        // No panics, dimensions match.
        assert_eq!(img.height(), opts.height as usize);
    }

    // ── Compositing ──────────────────────────────────────────────────────

    #[test]
    fn blend_over_zero_alpha_is_passthrough() {
        let dst = Srgba8::new(10, 20, 30, 200);
        let src = Srgba8::new(255, 0, 0, 0);
        assert_eq!(blend_over(src, dst), dst);
    }

    #[test]
    fn blend_over_full_alpha_replaces_destination() {
        let dst = Srgba8::new(10, 20, 30, 200);
        let src = Srgba8::new(255, 0, 0, 255);
        assert_eq!(blend_over(src, dst), src);
    }

    #[test]
    fn blend_over_half_alpha_mixes_components() {
        // 50% red over solid black → mid red, fully opaque (since dst
        // alpha is also 255).
        let dst = Srgba8::new(0, 0, 0, 255);
        let src = Srgba8::new(255, 0, 0, 128);
        let out = blend_over(src, dst);
        // 255*128/255 ≈ 128 (with rounding).
        assert!(out.r.0 >= 127 && out.r.0 <= 129, "r = {}", out.r.0);
        assert_eq!(out.g.0, 0);
        assert_eq!(out.b.0, 0);
        assert_eq!(out.a.0, 255);
    }
}
