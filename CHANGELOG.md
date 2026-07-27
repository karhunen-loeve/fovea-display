# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-07-27

### Changed

- Minimum required version of `fovea` bumped to `0.3.0`. The public API is
  otherwise unchanged from `0.2.0`; the release exists so `fovea-display`
  remains co-installable with the `fovea 0.3.0` release.

### Fixed

- A default-feature (no `debug-window`) build emitted two `dead_code`
  warnings for the crate-private `Framebuffer` helper, whose only consumers
  are behind that feature. The type is now annotated, so a default build is
  warning-free.

## [0.2.0] — 2026-06-12

### Changed

- Minimum required version of `fovea` bumped to `0.2.0`.

## [0.1.1] — 2026-05-29

First real public release. `0.1.0` was a name-reservation placeholder.

### Added

- Initial public release of the `fovea-display` crate.
- `DisplayStrategy` trait and built-in strategies: `Identity`,
  `LinearToDisplay`, `AutoContrast`, `FixedRange`.
- `TextureSource` trait for GPU upload of contiguous images.
- `DisplayPixel`, `GpuPixel`, `TextureFormat` types for sealed
  pixel/format compatibility.
- Optional `debug-window` Cargo feature gating:
  - `show()` — quick interactive preview of an image.
  - `DebugDisplay`, `DisplayContext` — embeddable debug viewer.
  - Histogram debug windows (`debug_histogram`,
    `debug_histogram_layers`, `render_histogram`,
    `render_histogram_layers`, `HistogramPlotOptions`,
    `HistogramRenderOptions`).

[0.3.0]: https://github.com/karhunen-loeve/fovea-display/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/karhunen-loeve/fovea-display/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karhunen-loeve/fovea-display/releases/tag/v0.1.1
