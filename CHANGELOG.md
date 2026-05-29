# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/karhunen-loeve/fovea-display/releases/tag/v0.1.1
