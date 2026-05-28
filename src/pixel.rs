//! Pixel traits for display and GPU texture integration.
//!
//! This module defines:
//! - [`DisplayPixel`] — sealed trait for pixels that can be written to a softbuffer framebuffer.
//! - [`TextureFormat`] — exhaustive enum of GPU texture format descriptors.
//! - [`GpuPixel`] — maps pixel types to their GPU texture format.

use irys_cv::pixel::{
    Bgra8, Mono8, Mono16, MonoA8, MonoA16, MonoAF32, MonoF32, PlainPixel, Rgba8, Rgba16, RgbaF32,
    SrgbMono8, Srgba8,
};
// ADR-0046: byte-layout items (`SIZE`, `ALIGN`, `as_bytes`,
// `from_bytes`) live on `PlainChannel`, not `PlainPixel`. Only the
// in-crate tests reference `SIZE` directly (to assert GpuPixel <->
// TextureFormat byte-count consistency); non-test code paths use
// `<T as PlainChannel>::SIZE` explicitly where needed.
#[cfg(test)]
use irys_cv::pixel::PlainChannel;

// ═══════════════════════════════════════════════════════════════════════════════
// 1.1 — DisplayPixel (sealed)
// ═══════════════════════════════════════════════════════════════════════════════

mod sealed {
    pub trait Sealed {}
}

/// A pixel that can be written directly to a framebuffer.
///
/// This trait is sealed — users cannot implement it for their own types.
/// The intended workflow is to go through a [`DisplayStrategy`](crate::DisplayStrategy)
/// that converts arbitrary pixels to [`Srgba8`], which implements this trait.
///
/// Only [`Srgba8`] implements this today. A future `Bgra8` variant may be
/// added for platforms where `softbuffer` uses BGRA layout.
///
/// # Alpha handling
///
/// `softbuffer` has no alpha channel — the output format is `0x00RRGGBB`.
/// Alpha is **discarded** (not composited). Pre-multiplied alpha display
/// is a potential future concern.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::Srgba8;
/// use irys_cv_display::DisplayPixel;
///
/// let px = Srgba8::new(255, 128, 0, 255);
/// assert_eq!(px.to_framebuffer_u32(), 0x00FF8000);
/// ```
pub trait DisplayPixel: PlainPixel + sealed::Sealed {
    /// Convert to softbuffer's `0x00RRGGBB` format.
    ///
    /// The high byte is always `0x00`. The alpha channel (if any) is discarded.
    fn to_framebuffer_u32(&self) -> u32;
}

impl sealed::Sealed for Srgba8 {}

impl DisplayPixel for Srgba8 {
    #[inline]
    fn to_framebuffer_u32(&self) -> u32 {
        let r = self.r.0 as u32;
        let g = self.g.0 as u32;
        let b = self.b.0 as u32;
        (r << 16) | (g << 8) | b
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1.2 — TextureFormat enum
// ═══════════════════════════════════════════════════════════════════════════════

/// Exhaustive GPU texture format descriptors for irys-cv pixel types.
///
/// This enum is always available (no feature flag required). It is a pure
/// Rust data type that downstream consumers (egui, bevy, raw Vulkan, wgpu)
/// can match on directly.
///
/// **Adding a variant is semver-major.**
///
/// # Design decisions
///
/// - **No 3-channel formats.** GPU APIs generally do not support 3-channel
///   textures (`Rgb8`, `Bgr8`, `RgbF32`). Users must convert to 4-channel
///   before GPU upload. This is explicit per irys-cv Philosophy #4.
///
/// - **`R8Srgb` included.** Not all GPU APIs support `R8_SRGB` (WebGPU/wgpu
///   notably do not), but the enum models the *logical* format. Downstream
///   integrations that target such an API should map `R8Srgb` to the
///   nearest available format (typically `R8Unorm`).
///
/// - **`Bgra8Srgb` reserved.** irys-cv has no `SrgbBgra8` type today, but
///   the format is common in GPU APIs. Included for forward-compatibility.
///
/// # Examples
///
/// ```
/// use irys_cv_display::TextureFormat;
///
/// let fmt = TextureFormat::Rgba8Srgb;
/// assert_eq!(fmt.bytes_per_pixel(), 4);
/// assert_eq!(fmt.channel_count(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Single-channel 8-bit unsigned normalized (linear). Maps to `Mono8`.
    R8Unorm,
    /// Single-channel 8-bit sRGB. Maps to `SrgbMono8`.
    R8Srgb,
    /// Two-channel 8-bit unsigned normalized. Maps to `MonoA8`.
    Rg8Unorm,
    /// Four-channel 8-bit unsigned normalized (linear). Maps to `Rgba8`.
    Rgba8Unorm,
    /// Four-channel 8-bit sRGB with alpha. Maps to `Srgba8`.
    Rgba8Srgb,
    /// Four-channel 8-bit BGRA unsigned normalized (linear). Maps to `Bgra8`.
    Bgra8Unorm,
    /// Four-channel 8-bit BGRA sRGB. Reserved for future `SrgbBgra8`.
    Bgra8Srgb,
    /// Single-channel 16-bit unsigned normalized. Maps to `Mono16`.
    R16Unorm,
    /// Two-channel 16-bit unsigned normalized. Maps to `MonoA16`.
    Rg16Unorm,
    /// Four-channel 16-bit unsigned normalized. Maps to `Rgba16`.
    Rgba16Unorm,
    /// Single-channel 32-bit float. Maps to `MonoF32`.
    R32Float,
    /// Two-channel 32-bit float. Maps to `MonoAF32`.
    Rg32Float,
    /// Four-channel 32-bit float. Maps to `RgbaF32`.
    Rgba32Float,
}

impl TextureFormat {
    /// Returns the total number of bytes per pixel for this format.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_display::TextureFormat;
    ///
    /// assert_eq!(TextureFormat::R8Unorm.bytes_per_pixel(), 1);
    /// assert_eq!(TextureFormat::Rg16Unorm.bytes_per_pixel(), 4);
    /// assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
    /// ```
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            // 1 channel × 1 byte
            TextureFormat::R8Unorm | TextureFormat::R8Srgb => 1,
            // 2 channels × 1 byte
            TextureFormat::Rg8Unorm => 2,
            // 4 channels × 1 byte
            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8Srgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8Srgb => 4,
            // 1 channel × 2 bytes
            TextureFormat::R16Unorm => 2,
            // 2 channels × 2 bytes
            TextureFormat::Rg16Unorm => 4,
            // 4 channels × 2 bytes
            TextureFormat::Rgba16Unorm => 8,
            // 1 channel × 4 bytes
            TextureFormat::R32Float => 4,
            // 2 channels × 4 bytes
            TextureFormat::Rg32Float => 8,
            // 4 channels × 4 bytes
            TextureFormat::Rgba32Float => 16,
        }
    }

    /// Returns the number of channels in this format.
    ///
    /// # Examples
    ///
    /// ```
    /// use irys_cv_display::TextureFormat;
    ///
    /// assert_eq!(TextureFormat::R8Unorm.channel_count(), 1);
    /// assert_eq!(TextureFormat::Rg8Unorm.channel_count(), 2);
    /// assert_eq!(TextureFormat::Rgba8Srgb.channel_count(), 4);
    /// ```
    #[inline]
    #[must_use]
    pub const fn channel_count(&self) -> usize {
        match self {
            TextureFormat::R8Unorm
            | TextureFormat::R8Srgb
            | TextureFormat::R16Unorm
            | TextureFormat::R32Float => 1,

            TextureFormat::Rg8Unorm | TextureFormat::Rg16Unorm | TextureFormat::Rg32Float => 2,

            TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8Srgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8Srgb
            | TextureFormat::Rgba16Unorm
            | TextureFormat::Rgba32Float => 4,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1.3 — GpuPixel trait
// ═══════════════════════════════════════════════════════════════════════════════

/// Maps a pixel type to its GPU texture format.
///
/// Only implemented for pixel types that have a **direct** GPU representation
/// (no conversion needed for upload). Notably **not** implemented for 3-channel
/// types like `Rgb8`, `Bgr8`, or `RgbF32` — users must convert to 4-channel
/// before GPU upload.
///
/// # Examples
///
/// ```
/// use irys_cv::pixel::Srgba8;
/// use irys_cv_display::{GpuPixel, TextureFormat};
///
/// assert_eq!(Srgba8::TEXTURE_FORMAT, TextureFormat::Rgba8Srgb);
/// ```
///
/// ```compile_fail
/// use irys_cv::pixel::Rgb8;
/// use irys_cv_display::GpuPixel;
///
/// // ERROR: Rgb8 does not implement GpuPixel — 3-channel types have
/// // no direct GPU representation. Convert to Rgba8 first.
/// let _ = Rgb8::TEXTURE_FORMAT;
/// ```
///
/// ```compile_fail
/// use irys_cv::pixel::Bgr8;
/// use irys_cv_display::GpuPixel;
///
/// // ERROR: Bgr8 does not implement GpuPixel — 3-channel types have
/// // no direct GPU representation. Convert to Bgra8 first.
/// let _ = Bgr8::TEXTURE_FORMAT;
/// ```
///
/// ```compile_fail
/// use irys_cv::pixel::RgbF32;
/// use irys_cv_display::GpuPixel;
///
/// // ERROR: RgbF32 does not implement GpuPixel — 3-channel types have
/// // no direct GPU representation. Convert to RgbaF32 first.
/// let _ = RgbF32::TEXTURE_FORMAT;
/// ```
pub trait GpuPixel: PlainPixel {
    /// The GPU texture format that corresponds to this pixel's memory layout.
    const TEXTURE_FORMAT: TextureFormat;
}

impl GpuPixel for Mono8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::R8Unorm;
}

impl GpuPixel for SrgbMono8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::R8Srgb;
}

impl GpuPixel for MonoA8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg8Unorm;
}

impl GpuPixel for Rgba8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
}

impl GpuPixel for Srgba8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Srgb;
}

impl GpuPixel for Bgra8 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;
}

impl GpuPixel for Mono16 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::R16Unorm;
}

impl GpuPixel for MonoA16 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg16Unorm;
}

impl GpuPixel for Rgba16 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Unorm;
}

// ADR-0044 Phase E: `f32` is no longer a pixel. The
// single-channel 32-bit float GPU format is carried by `MonoF32`,
// whose `#[repr(transparent)]` layout over `f32` is
// byte-identical to the previous bare-float impl.
impl GpuPixel for MonoF32 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::R32Float;
}

impl GpuPixel for MonoAF32 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg32Float;
}

impl GpuPixel for RgbaF32 {
    const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── DisplayPixel (1.1) ──────────────────────────────────────────────

    #[test]
    fn srgba8_to_framebuffer_red() {
        assert_eq!(Srgba8::new(255, 0, 0, 255).to_framebuffer_u32(), 0x00FF0000);
    }

    #[test]
    fn srgba8_to_framebuffer_green() {
        assert_eq!(Srgba8::new(0, 255, 0, 128).to_framebuffer_u32(), 0x0000FF00);
    }

    #[test]
    fn srgba8_to_framebuffer_blue() {
        assert_eq!(Srgba8::new(0, 0, 255, 0).to_framebuffer_u32(), 0x000000FF);
    }

    #[test]
    fn srgba8_to_framebuffer_black() {
        assert_eq!(Srgba8::new(0, 0, 0, 0).to_framebuffer_u32(), 0x00000000);
    }

    #[test]
    fn srgba8_to_framebuffer_white() {
        assert_eq!(
            Srgba8::new(255, 255, 255, 255).to_framebuffer_u32(),
            0x00FFFFFF
        );
    }

    #[test]
    fn srgba8_to_framebuffer_0x123456() {
        assert_eq!(
            Srgba8::new(18, 52, 86, 200).to_framebuffer_u32(),
            0x00123456
        );
    }

    #[test]
    fn srgba8_alpha_is_discarded() {
        // Same RGB, different alpha → same framebuffer value
        let a = Srgba8::new(100, 150, 200, 0).to_framebuffer_u32();
        let b = Srgba8::new(100, 150, 200, 255).to_framebuffer_u32();
        assert_eq!(a, b);
    }

    // ── TextureFormat bytes_per_pixel (1.2) ─────────────────────────────

    #[test]
    fn bytes_per_pixel_1byte_formats() {
        assert_eq!(TextureFormat::R8Unorm.bytes_per_pixel(), 1);
        assert_eq!(TextureFormat::R8Srgb.bytes_per_pixel(), 1);
    }

    #[test]
    fn bytes_per_pixel_2byte_formats() {
        assert_eq!(TextureFormat::Rg8Unorm.bytes_per_pixel(), 2);
        assert_eq!(TextureFormat::R16Unorm.bytes_per_pixel(), 2);
    }

    #[test]
    fn bytes_per_pixel_4byte_formats() {
        assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Rgba8Srgb.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Bgra8Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Bgra8Srgb.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Rg16Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::R32Float.bytes_per_pixel(), 4);
    }

    #[test]
    fn bytes_per_pixel_8byte_formats() {
        assert_eq!(TextureFormat::Rgba16Unorm.bytes_per_pixel(), 8);
        assert_eq!(TextureFormat::Rg32Float.bytes_per_pixel(), 8);
    }

    #[test]
    fn bytes_per_pixel_16byte_formats() {
        assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
    }

    // ── TextureFormat channel_count (1.2) ───────────────────────────────

    #[test]
    fn channel_count_1ch() {
        assert_eq!(TextureFormat::R8Unorm.channel_count(), 1);
        assert_eq!(TextureFormat::R8Srgb.channel_count(), 1);
        assert_eq!(TextureFormat::R16Unorm.channel_count(), 1);
        assert_eq!(TextureFormat::R32Float.channel_count(), 1);
    }

    #[test]
    fn channel_count_2ch() {
        assert_eq!(TextureFormat::Rg8Unorm.channel_count(), 2);
        assert_eq!(TextureFormat::Rg16Unorm.channel_count(), 2);
        assert_eq!(TextureFormat::Rg32Float.channel_count(), 2);
    }

    #[test]
    fn channel_count_4ch() {
        assert_eq!(TextureFormat::Rgba8Unorm.channel_count(), 4);
        assert_eq!(TextureFormat::Rgba8Srgb.channel_count(), 4);
        assert_eq!(TextureFormat::Bgra8Unorm.channel_count(), 4);
        assert_eq!(TextureFormat::Bgra8Srgb.channel_count(), 4);
        assert_eq!(TextureFormat::Rgba16Unorm.channel_count(), 4);
        assert_eq!(TextureFormat::Rgba32Float.channel_count(), 4);
    }

    // ── TextureFormat consistency check (1.2) ───────────────────────────

    #[test]
    fn bytes_per_pixel_equals_channels_times_component_size() {
        // Verify that bytes_per_pixel == channel_count × component_size
        // for every variant.
        let all = [
            TextureFormat::R8Unorm,
            TextureFormat::R8Srgb,
            TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8Srgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8Srgb,
            TextureFormat::R16Unorm,
            TextureFormat::Rg16Unorm,
            TextureFormat::Rgba16Unorm,
            TextureFormat::R32Float,
            TextureFormat::Rg32Float,
            TextureFormat::Rgba32Float,
        ];
        for fmt in &all {
            let bpp = fmt.bytes_per_pixel();
            let ch = fmt.channel_count();
            assert!(
                bpp % ch == 0,
                "{fmt:?}: bytes_per_pixel ({bpp}) not divisible by channel_count ({ch})"
            );
        }
    }

    // ── GpuPixel mappings (1.3) ─────────────────────────────────────────

    #[test]
    fn gpu_pixel_mono8() {
        assert_eq!(Mono8::TEXTURE_FORMAT, TextureFormat::R8Unorm);
    }

    #[test]
    fn gpu_pixel_srgb_mono8() {
        assert_eq!(SrgbMono8::TEXTURE_FORMAT, TextureFormat::R8Srgb);
    }

    #[test]
    fn gpu_pixel_mono_a8() {
        assert_eq!(MonoA8::TEXTURE_FORMAT, TextureFormat::Rg8Unorm);
    }

    #[test]
    fn gpu_pixel_rgba8() {
        assert_eq!(Rgba8::TEXTURE_FORMAT, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn gpu_pixel_srgba8() {
        assert_eq!(Srgba8::TEXTURE_FORMAT, TextureFormat::Rgba8Srgb);
    }

    #[test]
    fn gpu_pixel_bgra8() {
        assert_eq!(Bgra8::TEXTURE_FORMAT, TextureFormat::Bgra8Unorm);
    }

    #[test]
    fn gpu_pixel_mono16() {
        assert_eq!(Mono16::TEXTURE_FORMAT, TextureFormat::R16Unorm);
    }

    #[test]
    fn gpu_pixel_mono_a16() {
        assert_eq!(MonoA16::TEXTURE_FORMAT, TextureFormat::Rg16Unorm);
    }

    #[test]
    fn gpu_pixel_rgba16() {
        assert_eq!(Rgba16::TEXTURE_FORMAT, TextureFormat::Rgba16Unorm);
    }

    #[test]
    fn gpu_pixel_mono_f32() {
        assert_eq!(MonoF32::TEXTURE_FORMAT, TextureFormat::R32Float);
    }

    #[test]
    fn gpu_pixel_mono_af32() {
        assert_eq!(MonoAF32::TEXTURE_FORMAT, TextureFormat::Rg32Float);
    }

    #[test]
    fn gpu_pixel_rgba_f32() {
        assert_eq!(RgbaF32::TEXTURE_FORMAT, TextureFormat::Rgba32Float);
    }

    // ── GpuPixel ↔ TextureFormat consistency (1.3) ──────────────────────

    #[test]
    fn gpu_pixel_format_bytes_matches_pixel_size() {
        // For every GpuPixel impl, verify that the TextureFormat's
        // bytes_per_pixel matches the pixel type's SIZE.
        assert_eq!(Mono8::TEXTURE_FORMAT.bytes_per_pixel(), Mono8::SIZE);
        assert_eq!(SrgbMono8::TEXTURE_FORMAT.bytes_per_pixel(), SrgbMono8::SIZE);
        assert_eq!(MonoA8::TEXTURE_FORMAT.bytes_per_pixel(), MonoA8::SIZE);
        assert_eq!(Rgba8::TEXTURE_FORMAT.bytes_per_pixel(), Rgba8::SIZE);
        assert_eq!(Srgba8::TEXTURE_FORMAT.bytes_per_pixel(), Srgba8::SIZE);
        assert_eq!(Bgra8::TEXTURE_FORMAT.bytes_per_pixel(), Bgra8::SIZE);
        assert_eq!(Mono16::TEXTURE_FORMAT.bytes_per_pixel(), Mono16::SIZE);
        assert_eq!(MonoA16::TEXTURE_FORMAT.bytes_per_pixel(), MonoA16::SIZE);
        assert_eq!(Rgba16::TEXTURE_FORMAT.bytes_per_pixel(), Rgba16::SIZE);
        assert_eq!(MonoF32::TEXTURE_FORMAT.bytes_per_pixel(), MonoF32::SIZE);
        assert_eq!(MonoAF32::TEXTURE_FORMAT.bytes_per_pixel(), MonoAF32::SIZE);
        assert_eq!(RgbaF32::TEXTURE_FORMAT.bytes_per_pixel(), RgbaF32::SIZE);
    }
}
