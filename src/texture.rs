//! Zero-copy texture data source for GPU upload.
//!
//! This module defines [`TextureSource`], a trait that combines texture format
//! metadata with byte-level access to pixel data. It is blanket-implemented
//! for any [`PlainImage`](fovea::image::PlainImage) image whose pixel type implements
//! [`GpuPixel`](crate::GpuPixel).
//!
//! # Design note: `bytes_per_row` alignment
//!
//! GPU APIs often require row alignment (e.g., wgpu requires 256-byte
//! alignment for `COPY_BYTES_PER_ROW_ALIGNMENT`). This is **not** handled
//! here — it is the responsibility of the downstream upload code.
//! [`TextureSource::bytes_per_row`] reports the *logical* bytes per row
//! (width × pixel size), with no padding.

use fovea::image::PlainImage;
use fovea::pixel::PlainChannel;

use crate::pixel::{GpuPixel, TextureFormat};

/// Zero-copy texture data source for GPU upload.
///
/// Provides format metadata and byte access in a single trait, suitable
/// for passing directly to GPU texture creation and upload helpers.
///
/// This trait is blanket-implemented for any [`PlainImage`] image whose
/// pixel implements [`GpuPixel`]. You do not need to implement it manually.
///
/// # What implements `TextureSource`?
///
/// | Type                        | Implements? | Why                                        |
/// |-----------------------------|-------------|--------------------------------------------|
/// | `Image<Srgba8>`          | ✅ Yes      | `PlainImage` + `GpuPixel`                   |
/// | `Image<Mono8>`           | ✅ Yes      | `PlainImage` + `GpuPixel`                   |
/// | `ImageArray<Rgba8, 4, 4>`  | ✅ Yes      | `PlainImage` + `GpuPixel`                   |
/// | `Image<Rgb8>`            | ❌ No       | `Rgb8` has no `GpuPixel` impl (3-channel)  |
/// | `ImageRef<'_, Srgba8>`     | ❌ No       | No `PlainImage` (non-contiguous)            |
///
/// # Examples
///
/// ```
/// use fovea::image::{Image, PlainImage};
/// use fovea::pixel::Srgba8;
/// use fovea_display::{TextureSource, TextureFormat};
///
/// let img = Image::fill(16, 8, Srgba8::new(255, 0, 0, 255));
/// assert_eq!(img.texture_format(), TextureFormat::Rgba8Srgb);
/// assert_eq!(img.texture_width(), 16);
/// assert_eq!(img.texture_height(), 8);
/// assert_eq!(img.bytes_per_row(), 16 * 4); // 16 pixels × 4 bytes each
/// assert_eq!(img.texture_bytes().len(), 16 * 8 * 4);
/// ```
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::Rgb8;
/// use fovea_display::TextureSource;
///
/// // ERROR: Rgb8 does not implement GpuPixel — 3-channel types have
/// // no direct GPU representation. Convert to Rgba8 first.
/// let img = Image::fill(4, 4, Rgb8::new(0, 0, 0));
/// let _ = img.texture_format();
/// ```
pub trait TextureSource {
    /// The GPU texture format for this image's pixel type.
    fn texture_format(&self) -> TextureFormat;

    /// Image width in pixels.
    fn texture_width(&self) -> u32;

    /// Image height in pixels.
    fn texture_height(&self) -> u32;

    /// The raw pixel bytes, in row-major order with no padding.
    ///
    /// The returned slice has length
    /// `texture_width() * texture_height() * texture_format().bytes_per_pixel()`.
    fn texture_bytes(&self) -> &[u8];

    /// Logical bytes per row (width × bytes per pixel), **without** GPU
    /// alignment padding.
    ///
    /// Upload helpers are responsible for adding alignment padding if the
    /// target GPU API requires it (e.g. wgpu's 256-byte row alignment).
    fn bytes_per_row(&self) -> u32;
}

impl<T> TextureSource for T
where
    T: PlainImage,
    T::Pixel: GpuPixel,
{
    #[inline]
    fn texture_format(&self) -> TextureFormat {
        <T::Pixel as GpuPixel>::TEXTURE_FORMAT
    }

    #[inline]
    fn texture_width(&self) -> u32 {
        self.size().width as u32
    }

    #[inline]
    fn texture_height(&self) -> u32 {
        self.size().height as u32
    }

    #[inline]
    fn texture_bytes(&self) -> &[u8] {
        self.as_bytes()
    }

    #[inline]
    fn bytes_per_row(&self) -> u32 {
        // `SIZE` lives on `PlainChannel` (inherited by `PlainPixel`
        // via the supertrait relation). The bound on `T::Pixel` is
        // still `PlainPixel`; we just resolve the constant through the
        // byte-layout role.
        (self.size().width * <T::Pixel as PlainChannel>::SIZE) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fovea::image::{Image, ImageArray};
    use fovea::pixel::*;

    // ── Image<Mono8> ─────────────────────────────────────────────────

    #[test]
    fn image2d_mono8_texture_source() {
        let img = Image::<Mono8>::zero(10, 5);
        assert_eq!(img.texture_format(), TextureFormat::R8Unorm);
        assert_eq!(img.texture_width(), 10);
        assert_eq!(img.texture_height(), 5);
        assert_eq!(img.bytes_per_row(), 10);
        assert_eq!(img.texture_bytes().len(), 10 * 5);
    }

    // ── Image<Srgba8> ────────────────────────────────────────────────

    #[test]
    fn image2d_srgba8_texture_source() {
        let img = Image::fill(16, 8, Srgba8::new(255, 0, 0, 255));
        assert_eq!(img.texture_format(), TextureFormat::Rgba8Srgb);
        assert_eq!(img.texture_width(), 16);
        assert_eq!(img.texture_height(), 8);
        assert_eq!(img.bytes_per_row(), 16 * 4);
        assert_eq!(img.texture_bytes().len(), 16 * 8 * 4);
    }

    // ── Image<Rgba8> ─────────────────────────────────────────────────

    #[test]
    fn image2d_rgba8_texture_source() {
        let img = Image::fill(3, 7, Rgba8::new(1, 2, 3, 4));
        assert_eq!(img.texture_format(), TextureFormat::Rgba8Unorm);
        assert_eq!(img.texture_width(), 3);
        assert_eq!(img.texture_height(), 7);
        assert_eq!(img.bytes_per_row(), 3 * 4);
        assert_eq!(img.texture_bytes().len(), 3 * 7 * 4);
    }

    // ── Image<Mono16> ────────────────────────────────────────────────

    #[test]
    fn image2d_mono16_texture_source() {
        let img = Image::<Mono16>::zero(100, 50);
        assert_eq!(img.texture_format(), TextureFormat::R16Unorm);
        assert_eq!(img.texture_width(), 100);
        assert_eq!(img.texture_height(), 50);
        assert_eq!(img.bytes_per_row(), 100 * 2);
        assert_eq!(img.texture_bytes().len(), 100 * 50 * 2);
    }

    // ── Image<MonoF32> ──────────────────────────────────────────────

    #[test]
    fn image2d_f32_texture_source() {
        // the pixel role for floats is `MonoF32`,
        // not raw `f32`. `MonoF32` is `#[repr(transparent)]` over
        // `f32`, so the resulting texture layout is identical
        // (4-byte R32Float per pixel).
        let img = Image::<fovea::pixel::MonoF32>::zero(8, 4);
        assert_eq!(img.texture_format(), TextureFormat::R32Float);
        assert_eq!(img.texture_width(), 8);
        assert_eq!(img.texture_height(), 4);
        assert_eq!(img.bytes_per_row(), 8 * 4);
        assert_eq!(img.texture_bytes().len(), 8 * 4 * 4);
    }

    // ── Image<RgbaF32> ──────────────────────────────────────────────

    #[test]
    fn image2d_rgba_f32_texture_source() {
        let img = Image::fill(
            2,
            3,
            RgbaF32 {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        );
        assert_eq!(img.texture_format(), TextureFormat::Rgba32Float);
        assert_eq!(img.texture_width(), 2);
        assert_eq!(img.texture_height(), 3);
        assert_eq!(img.bytes_per_row(), 2 * 16);
        assert_eq!(img.texture_bytes().len(), 2 * 3 * 16);
    }

    // ── ImageArray ─────────────────────────────────────────────────────

    #[test]
    fn image_array_rgba8_texture_source() {
        let img = ImageArray::<Rgba8, 4, 4>::new([Rgba8::new(0, 0, 0, 0); 16]);
        assert_eq!(img.texture_format(), TextureFormat::Rgba8Unorm);
        assert_eq!(img.texture_width(), 4);
        assert_eq!(img.texture_height(), 4);
        assert_eq!(img.bytes_per_row(), 4 * 4);
        assert_eq!(img.texture_bytes().len(), 4 * 4 * 4);
    }

    // ── bytes_per_row consistency ──────────────────────────────────────

    #[test]
    fn bytes_per_row_equals_width_times_pixel_size() {
        let img = Image::fill(17, 3, MonoA16::new(0, 0));
        assert_eq!(
            img.bytes_per_row(),
            img.texture_width() * img.texture_format().bytes_per_pixel() as u32
        );
    }

    // ── texture_bytes length consistency ───────────────────────────────

    #[test]
    fn texture_bytes_len_equals_width_times_height_times_bpp() {
        let img = Image::fill(13, 7, Bgra8::new(0, 0, 0, 0));
        let expected = img.texture_width() as usize
            * img.texture_height() as usize
            * img.texture_format().bytes_per_pixel();
        assert_eq!(img.texture_bytes().len(), expected);
    }

    // ── zero-size image ────────────────────────────────────────────────

    #[test]
    fn zero_size_image_texture_source() {
        let img = Image::<Mono8>::zero(0, 0);
        assert_eq!(img.texture_width(), 0);
        assert_eq!(img.texture_height(), 0);
        assert_eq!(img.bytes_per_row(), 0);
        assert_eq!(img.texture_bytes().len(), 0);
    }

    // ── pixel byte content verification ────────────────────────────────

    #[test]
    fn texture_bytes_contain_correct_pixel_data() {
        let img = Image::fill(2, 1, Srgba8::new(0xAA, 0xBB, 0xCC, 0xDD));
        let bytes = img.texture_bytes();
        // Srgba8 is repr(C): r, g, b, a — each as Saturating<u8>
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC, 0xDD, 0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn single_pixel_image_texture_bytes() {
        let img = Image::fill(1, 1, Mono8::new(42));
        assert_eq!(img.texture_bytes(), &[42]);
    }
}
