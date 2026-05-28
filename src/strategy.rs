//! Display strategies: named conversions from pixel types to `Srgba8`.
//!
//! Every pixel type requires an explicit [`DisplayStrategy`] to become
//! displayable. This follows irys-cv's core philosophy: conversions are
//! named, data loss never happens silently.
//!
//! # Strategies
//!
//! | Strategy            | Use case                                        |
//! |---------------------|-------------------------------------------------|
//! | [`Identity`]        | sRGB types that are already display-ready        |
//! | [`LinearToDisplay`] | Linear-light types (applies sRGB gamma)          |
//! | [`AutoContrast`]    | Scan-based contrast stretching for HDR / deep    |
//! | [`FixedRange`]      | User-specified value range mapping               |
//!
//! # Examples
//!
//! ```
//! use irys_cv::pixel::Srgba8;
//! use irys_cv_display::{DisplayStrategy, Identity};
//!
//! let px = Srgba8::new(128, 64, 200, 255);
//! let out = Identity.to_display(&px);
//! assert_eq!(out, px);
//! ```

use irys_cv::image::ImageView;
use irys_cv::pixel::{
    Bgr8, Bgra8, Mono, Mono8, Mono16, Mono32, Mono64, MonoF32, MonoF64, Rgb8, RgbF32, Rgba8,
    RgbaF32, Srgb8, SrgbMono8, SrgbMonoA8, Srgba8,
};
use irys_cv::transform::{ConvertPixel, SrgbGamma};

use crate::pixel::DisplayPixel;

// ═══════════════════════════════════════════════════════════════════════════════
// 2.1  DisplayStrategy trait
// ═══════════════════════════════════════════════════════════════════════════════

/// Named conversion from any pixel type to [`Srgba8`] for display.
///
/// This follows irys-cv Philosophy #4: conversions are named, data loss
/// never happens silently. Every pixel type requires an explicit strategy
/// to become displayable.
///
/// The output is always [`Srgba8`] because displays are sRGB devices.
///
/// # Design note
///
/// This is intentionally separate from [`ConvertPixel<P, Srgba8>`]. See
/// the TODO.md "Why not reuse `ConvertPixel`?" section for rationale.
///
/// [`ConvertPixel<P, Srgba8>`]: irys_cv::transform::ConvertPixel
pub trait DisplayStrategy<P: Copy> {
    /// Convert a single pixel to display-ready [`Srgba8`].
    fn to_display(&self, pixel: &P) -> Srgba8;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.2  Identity strategy
// ═══════════════════════════════════════════════════════════════════════════════

/// Display sRGB pixels as-is. Only available for sRGB types.
///
/// Attempting to use `Identity` with a linear type like `Rgb8` is a
/// compile error — use [`LinearToDisplay`] instead.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::{Srgba8, Srgb8, SrgbMono8, SrgbMonoA8};
/// use irys_cv_display::{DisplayStrategy, Identity};
///
/// // Srgba8 pass-through
/// let px = Srgba8::new(100, 150, 200, 255);
/// assert_eq!(Identity.to_display(&px), px);
///
/// // Srgb8 → Srgba8 (alpha = 255)
/// let px = Srgb8::new(128, 64, 200);
/// assert_eq!(Identity.to_display(&px), Srgba8::new(128, 64, 200, 255));
///
/// // SrgbMono8 → broadcast to Srgba8
/// let px = SrgbMono8::new(128);
/// assert_eq!(Identity.to_display(&px), Srgba8::new(128, 128, 128, 255));
///
/// // SrgbMonoA8 → broadcast value, keep alpha
/// let px = SrgbMonoA8::new(128, 64);
/// assert_eq!(Identity.to_display(&px), Srgba8::new(128, 128, 128, 64));
/// ```
///
/// ```compile_fail
/// use irys_cv::pixel::Rgb8;
/// use irys_cv_display::{DisplayStrategy, Identity};
///
/// // ERROR: Rgb8 is linear, not sRGB — no Identity impl.
/// let px = Rgb8::new(128, 64, 200);
/// let _ = Identity.to_display(&px);
/// ```
///
/// ```compile_fail
/// use irys_cv::pixel::Mono16;
/// use irys_cv_display::{DisplayStrategy, Identity};
///
/// // ERROR: Mono16 is not an sRGB type — use AutoContrast or FixedRange.
/// let px = Mono16::new(1000);
/// let _ = Identity.to_display(&px);
/// ```
pub struct Identity;

impl DisplayStrategy<Srgba8> for Identity {
    #[inline]
    fn to_display(&self, pixel: &Srgba8) -> Srgba8 {
        *pixel
    }
}

impl DisplayStrategy<Srgb8> for Identity {
    #[inline]
    fn to_display(&self, pixel: &Srgb8) -> Srgba8 {
        Srgba8::new(pixel.r.0, pixel.g.0, pixel.b.0, 255)
    }
}

impl DisplayStrategy<SrgbMono8> for Identity {
    #[inline]
    fn to_display(&self, pixel: &SrgbMono8) -> Srgba8 {
        let v = pixel.0.0;
        Srgba8::new(v, v, v, 255)
    }
}

impl DisplayStrategy<SrgbMonoA8> for Identity {
    #[inline]
    fn to_display(&self, pixel: &SrgbMonoA8) -> Srgba8 {
        let v = pixel.v.0;
        Srgba8::new(v, v, v, pixel.a.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.3  LinearToDisplay strategy
// ═══════════════════════════════════════════════════════════════════════════════

/// Apply sRGB gamma encoding to linear pixels for display.
///
/// This strategy accepts linear-light pixel types (like `Rgb8`, `RgbF32`,
/// `Mono8`, `f32`) and applies the sRGB transfer function before display.
/// Without this, linear images appear washed out on sRGB monitors.
///
/// # Performance note
///
/// This strategy does float conversion for every pixel. For debug display
/// this is acceptable. For production rendering, users should pre-convert
/// to sRGB or use GPU shaders.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::{Rgb8, Srgba8};
/// use irys_cv_display::{DisplayStrategy, LinearToDisplay};
///
/// // Black stays black
/// let px = Rgb8::new(0, 0, 0);
/// assert_eq!(LinearToDisplay.to_display(&px), Srgba8::new(0, 0, 0, 255));
///
/// // White stays white
/// let px = Rgb8::new(255, 255, 255);
/// assert_eq!(LinearToDisplay.to_display(&px), Srgba8::new(255, 255, 255, 255));
/// ```
pub struct LinearToDisplay;

impl DisplayStrategy<Rgb8> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &Rgb8) -> Srgba8 {
        let linear = RgbF32::new(
            pixel.r.0 as f32 / 255.0,
            pixel.g.0 as f32 / 255.0,
            pixel.b.0 as f32 / 255.0,
        );
        let srgb: Srgb8 = SrgbGamma.convert(&linear);
        Srgba8::new(srgb.r.0, srgb.g.0, srgb.b.0, 255)
    }
}

impl DisplayStrategy<Rgba8> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &Rgba8) -> Srgba8 {
        let linear = RgbaF32::new(
            pixel.r.0 as f32 / 255.0,
            pixel.g.0 as f32 / 255.0,
            pixel.b.0 as f32 / 255.0,
            pixel.a.0 as f32 / 255.0,
        );
        SrgbGamma.convert(&linear)
    }
}

impl DisplayStrategy<RgbF32> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &RgbF32) -> Srgba8 {
        let srgb: Srgb8 = SrgbGamma.convert(pixel);
        Srgba8::new(srgb.r.0, srgb.g.0, srgb.b.0, 255)
    }
}

impl DisplayStrategy<RgbaF32> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &RgbaF32) -> Srgba8 {
        SrgbGamma.convert(pixel)
    }
}

impl DisplayStrategy<Bgr8> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &Bgr8) -> Srgba8 {
        let linear = RgbF32::new(
            pixel.r.0 as f32 / 255.0,
            pixel.g.0 as f32 / 255.0,
            pixel.b.0 as f32 / 255.0,
        );
        let srgb: Srgb8 = SrgbGamma.convert(&linear);
        Srgba8::new(srgb.r.0, srgb.g.0, srgb.b.0, 255)
    }
}

impl DisplayStrategy<Bgra8> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &Bgra8) -> Srgba8 {
        let linear = RgbaF32::new(
            pixel.r.0 as f32 / 255.0,
            pixel.g.0 as f32 / 255.0,
            pixel.b.0 as f32 / 255.0,
            pixel.a.0 as f32 / 255.0,
        );
        SrgbGamma.convert(&linear)
    }
}

impl DisplayStrategy<Mono8> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &Mono8) -> Srgba8 {
        let linear = MonoF32::new(pixel.value() as f32 / 255.0);
        let srgb: SrgbMono8 = SrgbGamma.convert(&linear);
        let v = srgb.0.0;
        Srgba8::new(v, v, v, 255)
    }
}

impl DisplayStrategy<f32> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &f32) -> Srgba8 {
        let clamped = MonoF32::new(pixel.clamp(0.0, 1.0));
        let srgb: SrgbMono8 = SrgbGamma.convert(&clamped);
        let v = srgb.0.0;
        Srgba8::new(v, v, v, 255)
    }
}

impl DisplayStrategy<MonoF32> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &MonoF32) -> Srgba8 {
        <Self as DisplayStrategy<f32>>::to_display(self, &pixel.0)
    }
}

impl DisplayStrategy<MonoF64> for LinearToDisplay {
    #[inline]
    fn to_display(&self, pixel: &MonoF64) -> Srgba8 {
        let clamped = MonoF32::new(pixel.0.clamp(0.0, 1.0) as f32);
        let srgb: SrgbMono8 = SrgbGamma.convert(&clamped);
        let v = srgb.0.0;
        Srgba8::new(v, v, v, 255)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.4  RangeMap internal helper
// ═══════════════════════════════════════════════════════════════════════════════

/// Internal helper: maps a [min, max] range to sRGB gray `Srgba8`.
///
/// Used by [`AutoContrast`] and [`FixedRange`] to share the core
/// normalize-then-gamma-encode math.
#[derive(Clone, Copy)]
struct RangeMap {
    min: f64,
    scale: f64, // 1.0 / (max - min), or 0.0 if min == max
}

impl RangeMap {
    /// Create a new range map.
    ///
    /// If `min == max`, all values map to mid-gray (128).
    fn new(min: f64, max: f64) -> Self {
        let range = max - min;
        let scale = if range.abs() < f64::EPSILON {
            0.0
        } else {
            1.0 / range
        };
        RangeMap { min, scale }
    }

    /// Map a scalar value to an sRGB-encoded gray [`Srgba8`].
    ///
    /// 1. Normalize to [0, 1]
    /// 2. Apply sRGB gamma encoding via `SrgbGamma`
    /// 3. Broadcast to `Srgba8` with alpha = 255
    #[inline]
    fn map_to_srgba8(&self, value: f64) -> Srgba8 {
        // Degenerate range → mid-gray
        if self.scale == 0.0 {
            return Srgba8::new(128, 128, 128, 255);
        }

        let t = MonoF32::new(((value - self.min) * self.scale).clamp(0.0, 1.0) as f32);
        let srgb: SrgbMono8 = SrgbGamma.convert(&t);
        let v = srgb.0.0;
        Srgba8::new(v, v, v, 255)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.5  AutoContrast strategy
// ═══════════════════════════════════════════════════════════════════════════════

/// Automatically determines display range by scanning the image.
///
/// Must be constructed via [`AutoContrast::new()`], [`AutoContrast::scan()`],
/// or [`AutoContrast::scan_with()`].
///
/// This strategy applies sRGB gamma encoding after range mapping:
/// values at `min` map to black, values at `max` map to white.
///
/// # Supported pixel types
///
/// `AutoContrast` implements [`DisplayStrategy`] for single-channel types:
/// `Mono8`, `Mono16`, `Mono32`, `Mono64`, `Mono<10>`, `Mono<12>`, `Mono<14>`,
/// `f32`, `f64`, `u8`, `u16`.
///
/// For multi-channel types (RGB, etc.), what "auto contrast" means is
/// ambiguous. Use [`LinearToDisplay`] or a custom strategy instead.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::{Mono16, Srgba8};
/// use irys_cv_display::{DisplayStrategy, AutoContrast};
///
/// let ac = AutoContrast::new(0.0, 65535.0);
/// assert_eq!(ac.to_display(&Mono16::new(0)), Srgba8::new(0, 0, 0, 255));
/// assert_eq!(ac.to_display(&Mono16::new(65535)), Srgba8::new(255, 255, 255, 255));
/// ```
#[derive(Clone, Copy)]
pub struct AutoContrast {
    range: RangeMap,
}

impl AutoContrast {
    /// Create an `AutoContrast` with an explicit min/max range.
    ///
    /// Values at `min` map to black, values at `max` map to white.
    /// If `min == max`, all pixels map to mid-gray.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        AutoContrast {
            range: RangeMap::new(min, max),
        }
    }

    /// Scan an image to determine the display range.
    ///
    /// Iterates all pixels, converting each to `f64` via `Into<f64>`,
    /// and finds the minimum and maximum values.
    ///
    /// # Type bounds
    ///
    /// Requires `V::Pixel: Into<f64>`. For pixel types that don't
    /// implement `Into<f64>` (e.g. `Mono<BITS>`), use
    /// [`AutoContrast::scan_with()`] instead.
    ///
    /// # Panics
    ///
    /// Returns a degenerate (mid-gray) range for empty images.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv::image::{Image, ImageView};
    /// use irys_cv::pixel::{MonoF32, Srgba8};
    /// use irys_cv_display::{DisplayStrategy, AutoContrast};
    ///
    /// // ADR-0044 Phase E: pixel role for floats is `MonoF32`, not `f32`.
    /// let img = Image::<MonoF32>::fill(4, 4, MonoF32::new(0.5));
    /// let ac = AutoContrast::scan(&img);
    /// // Constant image → degenerate range → mid-gray
    /// assert_eq!(ac.to_display(&MonoF32::new(0.5)), Srgba8::new(128, 128, 128, 255));
    /// ```
    pub fn scan<V>(image: &V) -> Self
    where
        V: ImageView,
        V::Pixel: Copy + Into<f64>,
    {
        Self::scan_with(image, |p| (*p).into())
    }

    /// Scan an image to determine the display range using a custom
    /// scalar extraction function.
    ///
    /// This is the general-purpose constructor that works with any pixel
    /// type, including `Mono<BITS>` and other types without `Into<f64>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv::image::{Image, ImageView, ImageViewMut};
    /// use irys_cv::pixel::{Mono16, Srgba8};
    /// use irys_cv_display::{DisplayStrategy, AutoContrast};
    ///
    /// let mut img = Image::<Mono16>::fill(4, 4, Mono16::new(100));
    /// *img.get_mut(0, 0).unwrap() = Mono16::new(50);
    /// *img.get_mut(3, 3).unwrap() = Mono16::new(200);
    ///
    /// let ac = AutoContrast::scan_with(&img, |p| p.value() as f64);
    /// assert_eq!(ac.to_display(&Mono16::new(50)), Srgba8::new(0, 0, 0, 255));
    /// assert_eq!(ac.to_display(&Mono16::new(200)), Srgba8::new(255, 255, 255, 255));
    /// ```
    pub fn scan_with<V, F>(image: &V, to_scalar: F) -> Self
    where
        V: ImageView,
        V::Pixel: Copy,
        F: Fn(&V::Pixel) -> f64,
    {
        let w = image.width();
        let h = image.height();

        if w == 0 || h == 0 {
            return AutoContrast::new(0.0, 0.0);
        }

        let first = to_scalar(&image.pixel_at(0, 0));
        let mut min = first;
        let mut max = first;

        for y in 0..h {
            for x in 0..w {
                let v = to_scalar(&image.pixel_at(x, y));
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }

        AutoContrast::new(min, max)
    }
}

// ── AutoContrast: DisplayStrategy impls ────────────────────────────────────

/// Helper: extract a `Mono8` intensity as `f64`.
#[inline(always)]
fn mono8_to_f64(pixel: &Mono8) -> f64 {
    pixel.value() as f64
}

/// Helper: extract a `Mono16` intensity as `f64`.
#[inline(always)]
fn mono16_to_f64(pixel: &Mono16) -> f64 {
    pixel.value() as f64
}

/// Helper: extract a `Mono32` intensity as `f64`.
#[inline(always)]
fn mono32_to_f64(pixel: &Mono32) -> f64 {
    pixel.value() as f64
}

/// Helper: extract a `Mono64` intensity as `f64`.
#[inline(always)]
fn mono64_to_f64(pixel: &Mono64) -> f64 {
    pixel.value() as f64
}

impl DisplayStrategy<Mono8> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &Mono8) -> Srgba8 {
        self.range.map_to_srgba8(mono8_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono16> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &Mono16) -> Srgba8 {
        self.range.map_to_srgba8(mono16_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono32> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &Mono32) -> Srgba8 {
        self.range.map_to_srgba8(mono32_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono64> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &Mono64) -> Srgba8 {
        self.range.map_to_srgba8(mono64_to_f64(pixel))
    }
}

impl DisplayStrategy<f32> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &f32) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl DisplayStrategy<f64> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &f64) -> Srgba8 {
        self.range.map_to_srgba8(*pixel)
    }
}

impl DisplayStrategy<MonoF32> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &MonoF32) -> Srgba8 {
        self.range.map_to_srgba8(pixel.0 as f64)
    }
}

impl DisplayStrategy<MonoF64> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &MonoF64) -> Srgba8 {
        self.range.map_to_srgba8(pixel.0)
    }
}

impl DisplayStrategy<u8> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &u8) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl DisplayStrategy<u16> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &u16) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl<const BITS: usize> DisplayStrategy<Mono<BITS>> for AutoContrast {
    #[inline]
    fn to_display(&self, pixel: &Mono<BITS>) -> Srgba8 {
        self.range.map_to_srgba8(pixel.value() as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.6  FixedRange strategy
// ═══════════════════════════════════════════════════════════════════════════════

/// Display with a fixed, user-specified value range.
///
/// Values at `min` map to black, values at `max` map to white.
/// Values outside the range are clamped.
///
/// This strategy applies sRGB gamma encoding after range mapping.
///
/// # Supported pixel types
///
/// Same as [`AutoContrast`]: single-channel types only.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::{Mono16, Srgba8};
/// use irys_cv_display::{DisplayStrategy, FixedRange};
///
/// let fr = FixedRange::new(100.0, 200.0);
/// assert_eq!(fr.to_display(&Mono16::new(100)), Srgba8::new(0, 0, 0, 255));
/// assert_eq!(fr.to_display(&Mono16::new(200)), Srgba8::new(255, 255, 255, 255));
///
/// // Values outside the range are clamped
/// assert_eq!(fr.to_display(&Mono16::new(0)), Srgba8::new(0, 0, 0, 255));
/// assert_eq!(fr.to_display(&Mono16::new(65535)), Srgba8::new(255, 255, 255, 255));
/// ```
#[derive(Clone, Copy)]
pub struct FixedRange {
    range: RangeMap,
}

impl FixedRange {
    /// Create a `FixedRange` with the given min/max bounds.
    ///
    /// Values at `min` map to black, values at `max` map to white.
    /// If `min == max`, all pixels map to mid-gray.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        FixedRange {
            range: RangeMap::new(min, max),
        }
    }
}

// ── FixedRange: DisplayStrategy impls (same types as AutoContrast) ─────────

impl DisplayStrategy<Mono8> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &Mono8) -> Srgba8 {
        self.range.map_to_srgba8(mono8_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono16> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &Mono16) -> Srgba8 {
        self.range.map_to_srgba8(mono16_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono32> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &Mono32) -> Srgba8 {
        self.range.map_to_srgba8(mono32_to_f64(pixel))
    }
}

impl DisplayStrategy<Mono64> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &Mono64) -> Srgba8 {
        self.range.map_to_srgba8(mono64_to_f64(pixel))
    }
}

impl DisplayStrategy<f32> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &f32) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl DisplayStrategy<f64> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &f64) -> Srgba8 {
        self.range.map_to_srgba8(*pixel)
    }
}

impl DisplayStrategy<u8> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &u8) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl DisplayStrategy<u16> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &u16) -> Srgba8 {
        self.range.map_to_srgba8(*pixel as f64)
    }
}

impl<const BITS: usize> DisplayStrategy<Mono<BITS>> for FixedRange {
    #[inline]
    fn to_display(&self, pixel: &Mono<BITS>) -> Srgba8 {
        self.range.map_to_srgba8(pixel.value() as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.7  Framebuffer helper type
// ═══════════════════════════════════════════════════════════════════════════════

/// A width×height buffer of `u32` pixels in `0x00RRGGBB` format.
///
/// This is the result of applying a [`DisplayStrategy`] to an
/// [`ImageView`]. It can be blitted directly to a `softbuffer::Buffer`.
///
/// This type is crate-private — it is an implementation detail of the
/// debug window system.
pub(crate) struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u32>,
}

impl Framebuffer {
    /// Convert an [`ImageView`] to a [`Framebuffer`] using the given
    /// [`DisplayStrategy`].
    ///
    /// Iterates pixels row-by-row, applies the strategy to produce
    /// [`Srgba8`], then packs each result into `0x00RRGGBB` via
    /// [`DisplayPixel::to_framebuffer_u32()`].
    pub fn from_image<V, S>(image: &V, strategy: S) -> Self
    where
        V: ImageView,
        V::Pixel: Copy,
        S: DisplayStrategy<V::Pixel>,
    {
        let w = image.width();
        let h = image.height();
        let len = w * h;
        let mut data = Vec::with_capacity(len);

        for y in 0..h {
            for x in 0..w {
                let pixel = image.pixel_at(x, y);
                let display = strategy.to_display(&pixel);
                data.push(display.to_framebuffer_u32());
            }
        }

        Framebuffer {
            width: w as u32,
            height: h as u32,
            data,
        }
    }

    /// Create a `Framebuffer` from pre-built data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    #[allow(dead_code)]
    pub fn from_raw(width: u32, height: u32, data: Vec<u32>) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "Framebuffer::from_raw: data length ({}) does not match dimensions ({}×{}={})",
            data.len(),
            width,
            height,
            (width as usize) * (height as usize),
        );
        Framebuffer {
            width,
            height,
            data,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use irys_cv::image::{Image, ImageViewMut, SubView};
    use irys_cv::pixel::*;

    // ── Identity tests ─────────────────────────────────────────────────

    #[test]
    fn identity_srgba8_passthrough() {
        let px = Srgba8::new(42, 100, 200, 128);
        assert_eq!(Identity.to_display(&px), px);
    }

    #[test]
    fn identity_srgba8_all_values_preserved() {
        for r in [0u8, 1, 127, 128, 254, 255] {
            for g in [0u8, 128, 255] {
                for b in [0u8, 128, 255] {
                    for a in [0u8, 128, 255] {
                        let px = Srgba8::new(r, g, b, a);
                        assert_eq!(Identity.to_display(&px), px);
                    }
                }
            }
        }
    }

    #[test]
    fn identity_srgb8_adds_alpha() {
        let px = Srgb8::new(128, 64, 200);
        assert_eq!(Identity.to_display(&px), Srgba8::new(128, 64, 200, 255));
    }

    #[test]
    fn identity_srgb8_black() {
        let px = Srgb8::new(0, 0, 0);
        assert_eq!(Identity.to_display(&px), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn identity_srgb8_white() {
        let px = Srgb8::new(255, 255, 255);
        assert_eq!(Identity.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn identity_srgb_mono8_broadcast() {
        let px = SrgbMono8::new(128);
        assert_eq!(Identity.to_display(&px), Srgba8::new(128, 128, 128, 255));
    }

    #[test]
    fn identity_srgb_mono8_black() {
        let px = SrgbMono8::new(0);
        assert_eq!(Identity.to_display(&px), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn identity_srgb_mono8_white() {
        let px = SrgbMono8::new(255);
        assert_eq!(Identity.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn identity_srgb_mono_a8_broadcast_with_alpha() {
        let px = SrgbMonoA8::new(128, 64);
        assert_eq!(Identity.to_display(&px), Srgba8::new(128, 128, 128, 64));
    }

    #[test]
    fn identity_srgb_mono_a8_black_transparent() {
        let px = SrgbMonoA8::new(0, 0);
        assert_eq!(Identity.to_display(&px), Srgba8::new(0, 0, 0, 0));
    }

    #[test]
    fn identity_srgb_mono_a8_white_opaque() {
        let px = SrgbMonoA8::new(255, 255);
        assert_eq!(Identity.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    // ── LinearToDisplay tests ──────────────────────────────────────────

    #[test]
    fn linear_rgb8_black() {
        let px = Rgb8::new(0, 0, 0);
        assert_eq!(LinearToDisplay.to_display(&px), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn linear_rgb8_white() {
        let px = Rgb8::new(255, 255, 255);
        assert_eq!(
            LinearToDisplay.to_display(&px),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn linear_rgbf32_mid_gray() {
        // Linear 0.5 → sRGB ≈ 188 (the sRGB transfer function maps 0.5 → ~0.735)
        let px = RgbF32::new(0.5, 0.5, 0.5);
        let result = LinearToDisplay.to_display(&px);
        // srgb_encode(0.5) = 1.055 * 0.5^(1/2.4) - 0.055 ≈ 0.7354 → round(0.7354*255) = 188
        assert_eq!(result.r.0, 188);
        assert_eq!(result.g.0, 188);
        assert_eq!(result.b.0, 188);
        assert_eq!(result.a.0, 255);
    }

    #[test]
    fn linear_mono8_correct_srgb_gray() {
        // Mono8(128) → linear 128/255 ≈ 0.502 → sRGB ≈ 188
        let px = Mono8::new(128);
        let result = LinearToDisplay.to_display(&px);
        assert_eq!(result.r.0, result.g.0);
        assert_eq!(result.g.0, result.b.0);
        assert_eq!(result.a.0, 255);
        // Linear 0.502 → sRGB ≈ 188
        assert!(result.r.0 >= 187 && result.r.0 <= 189);
    }

    #[test]
    fn linear_rgba_f32_alpha_preserved() {
        // Alpha should be transferred linearly (not gamma-encoded)
        let px = RgbaF32::new(0.5, 0.5, 0.5, 0.5);
        let result = LinearToDisplay.to_display(&px);
        // Alpha: round(0.5 * 255) = 128
        assert_eq!(result.a.0, 128);
        // RGB should be gamma-encoded ≈ 188
        assert_eq!(result.r.0, 188);
    }

    #[test]
    fn linear_bgr8_channel_order() {
        // Bgr8 has fields b, g, r but the .r field is still "red"
        let px = Bgr8::new(0, 0, 255); // b=0, g=0, r=255
        let result = LinearToDisplay.to_display(&px);
        assert_eq!(result.r.0, 255); // red channel maximum
        assert_eq!(result.g.0, 0);
        assert_eq!(result.b.0, 0);
    }

    #[test]
    fn linear_bgra8_channel_order_with_alpha() {
        let px = Bgra8::new(0, 0, 255, 128); // b=0, g=0, r=255, a=128
        let result = LinearToDisplay.to_display(&px);
        assert_eq!(result.r.0, 255);
        assert_eq!(result.g.0, 0);
        assert_eq!(result.b.0, 0);
        // Alpha: round(128/255 * 255) = 128
        assert_eq!(result.a.0, 128);
    }

    #[test]
    fn linear_f32_clamps_below_zero() {
        let px: f32 = -0.5;
        let result = LinearToDisplay.to_display(&px);
        assert_eq!(result, Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn linear_f32_clamps_above_one() {
        let px: f32 = 1.5;
        let result = LinearToDisplay.to_display(&px);
        assert_eq!(result, Srgba8::new(255, 255, 255, 255));
    }

    // ── RangeMap tests ─────────────────────────────────────────────────

    #[test]
    fn range_map_zero_to_one_black() {
        let rm = RangeMap::new(0.0, 1.0);
        assert_eq!(rm.map_to_srgba8(0.0), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn range_map_zero_to_one_white() {
        let rm = RangeMap::new(0.0, 1.0);
        assert_eq!(rm.map_to_srgba8(1.0), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn range_map_zero_to_one_mid_gray() {
        let rm = RangeMap::new(0.0, 1.0);
        let result = rm.map_to_srgba8(0.5);
        // Linear 0.5 → sRGB ≈ 188
        assert_eq!(result.r.0, 188);
        assert_eq!(result.g.0, 188);
        assert_eq!(result.b.0, 188);
        assert_eq!(result.a.0, 255);
    }

    #[test]
    fn range_map_custom_range_black() {
        let rm = RangeMap::new(100.0, 200.0);
        assert_eq!(rm.map_to_srgba8(100.0), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn range_map_custom_range_white() {
        let rm = RangeMap::new(100.0, 200.0);
        assert_eq!(rm.map_to_srgba8(200.0), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn range_map_degenerate_mid_gray() {
        let rm = RangeMap::new(5.0, 5.0);
        assert_eq!(rm.map_to_srgba8(5.0), Srgba8::new(128, 128, 128, 255));
    }

    #[test]
    fn range_map_clamp_below() {
        let rm = RangeMap::new(100.0, 200.0);
        assert_eq!(rm.map_to_srgba8(50.0), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn range_map_clamp_above() {
        let rm = RangeMap::new(100.0, 200.0);
        assert_eq!(rm.map_to_srgba8(300.0), Srgba8::new(255, 255, 255, 255));
    }

    // ── AutoContrast tests ─────────────────────────────────────────────

    #[test]
    fn auto_contrast_mono16_full_range() {
        let ac = AutoContrast::new(0.0, 65535.0);
        assert_eq!(ac.to_display(&Mono16::new(0)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            ac.to_display(&Mono16::new(65535)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_custom_range() {
        let ac = AutoContrast::new(100.0, 200.0);
        assert_eq!(ac.to_display(&Mono16::new(100)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            ac.to_display(&Mono16::new(200)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_f32_mid_gray() {
        let ac = AutoContrast::new(0.0, 1.0);
        let result = ac.to_display(&0.5f32);
        assert_eq!(result.r.0, 188);
    }

    #[test]
    fn auto_contrast_scan_f32() {
        let mut img = Image::<MonoF32>::fill(4, 4, MonoF32::new(0.5));
        *img.get_mut(0, 0).unwrap() = MonoF32::new(0.0);
        *img.get_mut(3, 3).unwrap() = MonoF32::new(1.0);

        let ac = AutoContrast::scan(&img);
        assert_eq!(ac.to_display(&MonoF32::new(0.0)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            ac.to_display(&MonoF32::new(1.0)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_scan_constant_image_degenerate() {
        let img = Image::<MonoF32>::fill(4, 4, MonoF32::new(0.5));
        let ac = AutoContrast::scan(&img);
        // Constant image → degenerate range → mid-gray
        assert_eq!(
            ac.to_display(&MonoF32::new(0.5)),
            Srgba8::new(128, 128, 128, 255)
        );
    }

    #[test]
    fn auto_contrast_scan_single_pixel_degenerate() {
        let img = Image::<MonoF64>::fill(1, 1, MonoF64::new(42.0));
        let ac = AutoContrast::scan(&img);
        assert_eq!(
            ac.to_display(&MonoF64::new(42.0)),
            Srgba8::new(128, 128, 128, 255)
        );
    }

    #[test]
    fn auto_contrast_scan_empty_image() {
        let img = Image::<MonoF32>::fill(0, 0, MonoF32::new(0.0));
        let ac = AutoContrast::scan(&img);
        // Degenerate range
        assert_eq!(
            ac.to_display(&MonoF32::new(0.0)),
            Srgba8::new(128, 128, 128, 255)
        );
    }

    #[test]
    fn auto_contrast_scan_with_mono16() {
        let mut img = Image::<Mono16>::fill(4, 4, Mono16::new(100));
        *img.get_mut(0, 0).unwrap() = Mono16::new(50);
        *img.get_mut(3, 3).unwrap() = Mono16::new(200);

        let ac = AutoContrast::scan_with(&img, |p| mono16_to_f64(p));
        assert_eq!(ac.to_display(&Mono16::new(50)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            ac.to_display(&Mono16::new(200)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_mono32() {
        let ac = AutoContrast::new(0.0, u32::MAX as f64);
        assert_eq!(ac.to_display(&Mono32::new(0)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            ac.to_display(&Mono32::new(u32::MAX)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_mono64() {
        let ac = AutoContrast::new(0.0, u64::MAX as f64);
        assert_eq!(ac.to_display(&Mono64::new(0)), Srgba8::new(0, 0, 0, 255));
        // u64::MAX as f64 → exact match at 1.0 → white
        assert_eq!(
            ac.to_display(&Mono64::new(u64::MAX)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn auto_contrast_u8() {
        let ac = AutoContrast::new(0.0, 255.0);
        assert_eq!(ac.to_display(&0u8), Srgba8::new(0, 0, 0, 255));
        assert_eq!(ac.to_display(&255u8), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn auto_contrast_u16() {
        let ac = AutoContrast::new(0.0, 65535.0);
        assert_eq!(ac.to_display(&0u16), Srgba8::new(0, 0, 0, 255));
        assert_eq!(ac.to_display(&65535u16), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn auto_contrast_f64() {
        let ac = AutoContrast::new(0.0, 1.0);
        assert_eq!(ac.to_display(&0.0f64), Srgba8::new(0, 0, 0, 255));
        assert_eq!(ac.to_display(&1.0f64), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn auto_contrast_mono10() {
        let ac = AutoContrast::new(0.0, 1023.0);
        let px = Mono10::new(0);
        assert_eq!(ac.to_display(&px), Srgba8::new(0, 0, 0, 255));
        let px = Mono10::new(1023);
        assert_eq!(ac.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn auto_contrast_mono12() {
        let ac = AutoContrast::new(0.0, 4095.0);
        let px = Mono12::new(0);
        assert_eq!(ac.to_display(&px), Srgba8::new(0, 0, 0, 255));
        let px = Mono12::new(4095);
        assert_eq!(ac.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn auto_contrast_mono14() {
        let ac = AutoContrast::new(0.0, 16383.0);
        let px = Mono14::new(0);
        assert_eq!(ac.to_display(&px), Srgba8::new(0, 0, 0, 255));
        let px = Mono14::new(16383);
        assert_eq!(ac.to_display(&px), Srgba8::new(255, 255, 255, 255));
    }

    // ── FixedRange tests ───────────────────────────────────────────────

    #[test]
    fn fixed_range_mono16_boundaries() {
        let fr = FixedRange::new(100.0, 200.0);
        assert_eq!(fr.to_display(&Mono16::new(100)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            fr.to_display(&Mono16::new(200)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn fixed_range_clamping_below() {
        let fr = FixedRange::new(100.0, 200.0);
        assert_eq!(fr.to_display(&Mono16::new(0)), Srgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn fixed_range_clamping_above() {
        let fr = FixedRange::new(100.0, 200.0);
        assert_eq!(
            fr.to_display(&Mono16::new(65535)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    #[test]
    fn fixed_range_degenerate() {
        let fr = FixedRange::new(42.0, 42.0);
        assert_eq!(
            fr.to_display(&Mono16::new(42)),
            Srgba8::new(128, 128, 128, 255)
        );
    }

    #[test]
    fn fixed_range_f32() {
        let fr = FixedRange::new(0.0, 1.0);
        assert_eq!(fr.to_display(&0.0f32), Srgba8::new(0, 0, 0, 255));
        assert_eq!(fr.to_display(&1.0f32), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn fixed_range_f64() {
        let fr = FixedRange::new(-1.0, 1.0);
        assert_eq!(fr.to_display(&-1.0f64), Srgba8::new(0, 0, 0, 255));
        assert_eq!(fr.to_display(&1.0f64), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn fixed_range_u8() {
        let fr = FixedRange::new(0.0, 255.0);
        assert_eq!(fr.to_display(&0u8), Srgba8::new(0, 0, 0, 255));
        assert_eq!(fr.to_display(&255u8), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn fixed_range_u16() {
        let fr = FixedRange::new(0.0, 65535.0);
        assert_eq!(fr.to_display(&0u16), Srgba8::new(0, 0, 0, 255));
        assert_eq!(fr.to_display(&65535u16), Srgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn fixed_range_mono10() {
        let fr = FixedRange::new(0.0, 1023.0);
        assert_eq!(fr.to_display(&Mono10::new(0)), Srgba8::new(0, 0, 0, 255));
        assert_eq!(
            fr.to_display(&Mono10::new(1023)),
            Srgba8::new(255, 255, 255, 255)
        );
    }

    // ── Framebuffer tests ──────────────────────────────────────────────

    #[test]
    fn framebuffer_from_image_2x2_srgba8() {
        let mut img = Image::fill(2, 2, Srgba8::new(0, 0, 0, 255));
        *img.get_mut(0, 0).unwrap() = Srgba8::new(255, 0, 0, 255);
        *img.get_mut(1, 0).unwrap() = Srgba8::new(0, 255, 0, 255);
        *img.get_mut(0, 1).unwrap() = Srgba8::new(0, 0, 255, 255);
        *img.get_mut(1, 1).unwrap() = Srgba8::new(255, 255, 255, 255);

        let fb = Framebuffer::from_image(&img, Identity);
        assert_eq!(fb.width, 2);
        assert_eq!(fb.height, 2);
        assert_eq!(fb.data.len(), 4);
        assert_eq!(fb.data[0], 0x00FF0000); // red
        assert_eq!(fb.data[1], 0x0000FF00); // green
        assert_eq!(fb.data[2], 0x000000FF); // blue
        assert_eq!(fb.data[3], 0x00FFFFFF); // white
    }

    #[test]
    fn framebuffer_from_image_zero_size() {
        let img = Image::<Srgba8>::fill(0, 0, Srgba8::new(0, 0, 0, 0));
        let fb = Framebuffer::from_image(&img, Identity);
        assert_eq!(fb.width, 0);
        assert_eq!(fb.height, 0);
        assert!(fb.data.is_empty());
    }

    #[test]
    fn framebuffer_from_image_1x1() {
        let img = Image::fill(1, 1, Srgba8::new(0xAA, 0xBB, 0xCC, 0xFF));
        let fb = Framebuffer::from_image(&img, Identity);
        assert_eq!(fb.width, 1);
        assert_eq!(fb.height, 1);
        assert_eq!(fb.data.len(), 1);
        assert_eq!(fb.data[0], 0x00AABBCC);
    }

    #[test]
    fn framebuffer_from_image_with_strategy() {
        // Use LinearToDisplay on a Mono8 image
        let img = Image::fill(2, 1, Mono8::new(0));
        let fb = Framebuffer::from_image(&img, LinearToDisplay);
        assert_eq!(fb.width, 2);
        assert_eq!(fb.height, 1);
        assert_eq!(fb.data.len(), 2);
        // Mono8(0) → linear 0.0 → sRGB 0 → 0x00000000
        assert_eq!(fb.data[0], 0x00000000);
        assert_eq!(fb.data[1], 0x00000000);
    }

    #[test]
    fn framebuffer_from_image_roi() {
        // Create a 4x4 image, take a 2x2 ROI, and convert
        let mut img = Image::fill(4, 4, Srgba8::new(0, 0, 0, 255));
        *img.get_mut(1, 1).unwrap() = Srgba8::new(255, 0, 0, 255);
        *img.get_mut(2, 1).unwrap() = Srgba8::new(0, 255, 0, 255);
        *img.get_mut(1, 2).unwrap() = Srgba8::new(0, 0, 255, 255);
        *img.get_mut(2, 2).unwrap() = Srgba8::new(255, 255, 255, 255);

        let roi = img
            .roi(irys_cv::Rectangle::new((1usize, 1usize), (2usize, 2usize)))
            .unwrap();
        let fb = Framebuffer::from_image(&roi, Identity);
        assert_eq!(fb.width, 2);
        assert_eq!(fb.height, 2);
        assert_eq!(fb.data[0], 0x00FF0000); // red
        assert_eq!(fb.data[1], 0x0000FF00); // green
        assert_eq!(fb.data[2], 0x000000FF); // blue
        assert_eq!(fb.data[3], 0x00FFFFFF); // white
    }

    #[test]
    fn framebuffer_from_raw_valid() {
        let fb = Framebuffer::from_raw(2, 2, vec![0, 1, 2, 3]);
        assert_eq!(fb.width, 2);
        assert_eq!(fb.height, 2);
        assert_eq!(fb.data, vec![0, 1, 2, 3]);
    }

    #[test]
    fn framebuffer_from_raw_empty() {
        let fb = Framebuffer::from_raw(0, 0, vec![]);
        assert_eq!(fb.width, 0);
        assert_eq!(fb.height, 0);
        assert!(fb.data.is_empty());
    }

    #[test]
    #[should_panic(expected = "does not match dimensions")]
    fn framebuffer_from_raw_wrong_size() {
        let _ = Framebuffer::from_raw(2, 2, vec![0, 1, 2]);
    }
}
