//! Integration tests for fovea-display traits.
//!
//! These tests verify the end-to-end pipeline from image creation through
//! display strategy application to final `u32` framebuffer values, as well
//! as the `TextureSource` blanket implementations.

use fovea::Rectangle;
use fovea::image::{Image, ImageArray, ImageView, ImageViewMut, SubView};
use fovea::pixel::{Bgra8, Mono8, Mono16, MonoF32, RgbF32, Rgba8, RgbaF32, Srgb8, Srgba8};
use fovea_display::{
    AutoContrast, DisplayPixel, DisplayStrategy, FixedRange, Identity, LinearToDisplay,
    TextureFormat, TextureSource,
};

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: Image<Mono16> → AutoContrast → Srgba8 → u32
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mono16_auto_contrast_end_to_end() {
    // Create a 4×4 Mono16 image with a known gradient.
    let mut img = Image::<Mono16>::zero(4, 4);
    // Set corners to known values.
    *img.get_mut(0, 0).unwrap() = Mono16::new(0);
    *img.get_mut(3, 0).unwrap() = Mono16::new(32768);
    *img.get_mut(0, 3).unwrap() = Mono16::new(49152);
    *img.get_mut(3, 3).unwrap() = Mono16::new(65535);

    // Scan using the `.value()` accessor.
    let strategy = AutoContrast::scan_with(&img, |p: &Mono16| p.value() as f64);

    // Black pixel (min=0) → should map to black (0, 0, 0).
    let black = strategy.to_display(&Mono16::new(0));
    assert_eq!(black, Srgba8::new(0, 0, 0, 255));

    // White pixel (max=65535) → should map to white (255, 255, 255).
    let white = strategy.to_display(&Mono16::new(65535));
    assert_eq!(white, Srgba8::new(255, 255, 255, 255));

    // Verify the full pipeline through to u32 framebuffer value.
    let black_u32 = black.to_framebuffer_u32();
    assert_eq!(black_u32, 0x00000000);

    let white_u32 = white.to_framebuffer_u32();
    assert_eq!(white_u32, 0x00FFFFFF);

    // Mid-gray should be somewhere between 0 and 255.
    let mid = strategy.to_display(&Mono16::new(32768));
    let mid_u32 = mid.to_framebuffer_u32();
    assert!(mid_u32 > 0x00000000);
    assert!(mid_u32 < 0x00FFFFFF);
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: Image<f32> → AutoContrast::scan → Srgba8 → u32
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn f32_auto_contrast_scan_end_to_end() {
    // ADR-0044 Phase E: the pixel role for floats is `MonoF32`,
    // not raw `f32`. `MonoF32` is `#[repr(transparent)]` over `f32`.
    let mut img = Image::<MonoF32>::zero(3, 3);
    *img.get_mut(0, 0).unwrap() = MonoF32::new(0.0);
    *img.get_mut(1, 1).unwrap() = MonoF32::new(0.5);
    *img.get_mut(2, 2).unwrap() = MonoF32::new(1.0);

    // MonoF32 implements Into<f64> (via its inner f32), so scan() works.
    let strategy = AutoContrast::scan(&img);

    let black = strategy.to_display(&MonoF32::new(0.0));
    assert_eq!(black, Srgba8::new(0, 0, 0, 255));

    let white = strategy.to_display(&MonoF32::new(1.0));
    assert_eq!(white, Srgba8::new(255, 255, 255, 255));

    // Verify u32 encoding.
    assert_eq!(black.to_framebuffer_u32(), 0x00000000);
    assert_eq!(white.to_framebuffer_u32(), 0x00FFFFFF);
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: Image<Srgba8> → Identity → Srgba8 → u32
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn srgba8_identity_end_to_end() {
    let px = Srgba8::new(0x12, 0x34, 0x56, 0xFF);
    let img = Image::fill(2, 2, px);

    let display = Identity.to_display(&img.pixel_at(0, 0));
    assert_eq!(display, px);
    assert_eq!(display.to_framebuffer_u32(), 0x00123456);
}

#[test]
fn srgb8_identity_end_to_end() {
    let px = Srgb8::new(0xAA, 0xBB, 0xCC);
    let display = Identity.to_display(&px);
    assert_eq!(display, Srgba8::new(0xAA, 0xBB, 0xCC, 255));
    assert_eq!(display.to_framebuffer_u32(), 0x00AABBCC);
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: FixedRange
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn fixed_range_u16_end_to_end() {
    let strategy = FixedRange::new(100.0, 200.0);

    // Below range → black.
    let below = strategy.to_display(&50u16);
    assert_eq!(below, Srgba8::new(0, 0, 0, 255));

    // Above range → white.
    let above = strategy.to_display(&250u16);
    assert_eq!(above, Srgba8::new(255, 255, 255, 255));

    // At minimum → black.
    let at_min = strategy.to_display(&100u16);
    assert_eq!(at_min, Srgba8::new(0, 0, 0, 255));

    // At maximum → white.
    let at_max = strategy.to_display(&200u16);
    assert_eq!(at_max, Srgba8::new(255, 255, 255, 255));
}

// ═══════════════════════════════════════════════════════════════════════════════
// End-to-end: LinearToDisplay for linear types
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn linear_rgb_f32_end_to_end() {
    let black = RgbF32 {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    let white = RgbF32 {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };

    let display_black = LinearToDisplay.to_display(&black);
    assert_eq!(display_black, Srgba8::new(0, 0, 0, 255));

    let display_white = LinearToDisplay.to_display(&white);
    assert_eq!(display_white, Srgba8::new(255, 255, 255, 255));
}

#[test]
fn linear_rgba_f32_alpha_preserved() {
    let px = RgbaF32 {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 0.5,
    };
    let display = LinearToDisplay.to_display(&px);
    assert_eq!(display.r.0, 255);
    assert_eq!(display.g.0, 0);
    // Alpha is preserved (linearly mapped to 0–255 range).
    assert!(display.a.0 > 100 && display.a.0 < 150);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TextureSource blanket impl: Image
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn texture_source_image2d_srgba8() {
    let img = Image::fill(16, 8, Srgba8::new(255, 0, 0, 255));
    assert_eq!(img.texture_format(), TextureFormat::Rgba8Srgb);
    assert_eq!(img.texture_width(), 16);
    assert_eq!(img.texture_height(), 8);
    assert_eq!(img.bytes_per_row(), 16 * 4);
    assert_eq!(img.texture_bytes().len(), 16 * 8 * 4);
}

#[test]
fn texture_source_image2d_mono8() {
    let img = Image::<Mono8>::zero(100, 50);
    assert_eq!(img.texture_format(), TextureFormat::R8Unorm);
    assert_eq!(img.texture_width(), 100);
    assert_eq!(img.texture_height(), 50);
    assert_eq!(img.bytes_per_row(), 100);
    assert_eq!(img.texture_bytes().len(), 100 * 50);
}

#[test]
fn texture_source_image2d_mono16() {
    let img = Image::<Mono16>::zero(64, 32);
    assert_eq!(img.texture_format(), TextureFormat::R16Unorm);
    assert_eq!(img.texture_width(), 64);
    assert_eq!(img.texture_height(), 32);
    assert_eq!(img.bytes_per_row(), 64 * 2);
    assert_eq!(img.texture_bytes().len(), 64 * 32 * 2);
}

#[test]
fn texture_source_image2d_rgba8() {
    let img = Image::fill(5, 3, Rgba8::new(1, 2, 3, 4));
    assert_eq!(img.texture_format(), TextureFormat::Rgba8Unorm);
    assert_eq!(img.texture_width(), 5);
    assert_eq!(img.texture_height(), 3);
    assert_eq!(img.bytes_per_row(), 5 * 4);
    assert_eq!(img.texture_bytes().len(), 5 * 3 * 4);
}

#[test]
fn texture_source_image2d_bgra8() {
    let img = Image::fill(10, 10, Bgra8::new(0, 0, 0, 255));
    assert_eq!(img.texture_format(), TextureFormat::Bgra8Unorm);
    assert_eq!(img.texture_width(), 10);
    assert_eq!(img.texture_height(), 10);
    assert_eq!(img.bytes_per_row(), 10 * 4);
}

#[test]
fn texture_source_image2d_f32() {
    // ADR-0044 Phase E: `Image<MonoF32>` replaces `Image<f32>` for
    // the pixel role. `MonoF32` is `#[repr(transparent)]` over `f32`,
    // so the R32Float texture layout is unchanged.
    let img = Image::<MonoF32>::zero(8, 4);
    assert_eq!(img.texture_format(), TextureFormat::R32Float);
    assert_eq!(img.bytes_per_row(), 8 * 4);
    assert_eq!(img.texture_bytes().len(), 8 * 4 * 4);
}

#[test]
fn texture_source_image2d_rgba_f32() {
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
    assert_eq!(img.bytes_per_row(), 2 * 16);
    assert_eq!(img.texture_bytes().len(), 2 * 3 * 16);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TextureSource blanket impl: ImageArray
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn texture_source_image_array_rgba8() {
    let img = ImageArray::<Rgba8, 4, 4>::new([Rgba8::new(0, 0, 0, 0); 16]);
    assert_eq!(img.texture_format(), TextureFormat::Rgba8Unorm);
    assert_eq!(img.texture_width(), 4);
    assert_eq!(img.texture_height(), 4);
    assert_eq!(img.bytes_per_row(), 4 * 4);
    assert_eq!(img.texture_bytes().len(), 4 * 4 * 4);
}

#[test]
fn texture_source_image_array_srgba8() {
    let img = ImageArray::<Srgba8, 8, 2>::new([Srgba8::new(255, 0, 0, 255); 16]);
    assert_eq!(img.texture_format(), TextureFormat::Rgba8Srgb);
    assert_eq!(img.texture_width(), 8);
    assert_eq!(img.texture_height(), 2);
    assert_eq!(img.bytes_per_row(), 8 * 4);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TextureSource: bytes_per_row consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bytes_per_row_equals_width_times_bpp() {
    let img = Image::fill(17, 3, Srgba8::new(0, 0, 0, 0));
    assert_eq!(
        img.bytes_per_row(),
        img.texture_width() * img.texture_format().bytes_per_pixel() as u32
    );
}

#[test]
fn texture_bytes_len_equals_rows_times_bytes_per_row() {
    let img = Image::fill(13, 7, Bgra8::new(0, 0, 0, 0));
    assert_eq!(
        img.texture_bytes().len(),
        img.texture_height() as usize * img.bytes_per_row() as usize
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TextureSource: zero-size image
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn texture_source_zero_size() {
    let img = Image::<Mono8>::zero(0, 0);
    assert_eq!(img.texture_width(), 0);
    assert_eq!(img.texture_height(), 0);
    assert_eq!(img.bytes_per_row(), 0);
    assert_eq!(img.texture_bytes().len(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TextureSource: pixel byte content verification
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn texture_bytes_contain_correct_pixel_data() {
    let img = Image::fill(2, 1, Srgba8::new(0xAA, 0xBB, 0xCC, 0xDD));
    let bytes = img.texture_bytes();
    // Srgba8 is repr(C): r, g, b, a — each as Saturating<u8>.
    assert_eq!(bytes, &[0xAA, 0xBB, 0xCC, 0xDD, 0xAA, 0xBB, 0xCC, 0xDD]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROI display: ImageView (not just Image) works with strategies
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn roi_with_identity_strategy() {
    let img = Image::fill(10, 10, Srgba8::new(100, 150, 200, 255));
    let roi = img.roi(Rectangle::new((2, 2), (4, 4))).unwrap();

    // ROI implements ImageView, so we can iterate and apply strategy.
    for y in 0..roi.height() {
        for x in 0..roi.width() {
            let px = roi.pixel_at(x, y);
            let display = Identity.to_display(&px);
            assert_eq!(display, Srgba8::new(100, 150, 200, 255));
        }
    }
}

#[test]
fn roi_with_auto_contrast_strategy() {
    // Create a gradient image.
    // ADR-0044 Phase E: pixel role for floats is `MonoF32`.
    let mut img = Image::<MonoF32>::zero(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            *img.get_mut(x, y).unwrap() = MonoF32::new((x + y * 10) as f32 / 99.0);
        }
    }

    // Take an ROI and scan it for auto-contrast.
    let roi = img.roi(Rectangle::new((2, 2), (6, 6))).unwrap();
    let strategy = AutoContrast::scan(&roi);

    // The ROI's min/max should determine the mapping.
    // Min pixel in ROI: (2,2) = (2 + 20)/99 ≈ 0.222
    // Max pixel in ROI: (7,7) = (7 + 70)/99 ≈ 0.778
    let min_val = roi.pixel_at(0, 0);
    let max_val = roi.pixel_at(roi.width() - 1, roi.height() - 1);

    let display_min = strategy.to_display(&min_val);
    let display_max = strategy.to_display(&max_val);

    // Min of ROI should map to black.
    assert_eq!(display_min, Srgba8::new(0, 0, 0, 255));
    // Max of ROI should map to white.
    assert_eq!(display_max, Srgba8::new(255, 255, 255, 255));
}
