# fovea-display

[![Crates.io](https://img.shields.io/crates/v/fovea-display.svg)](https://crates.io/crates/fovea-display)
[![Documentation](https://docs.rs/fovea-display/badge.svg)](https://docs.rs/fovea-display)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/karhunen-loeve/fovea-display/blob/main/LICENSE)

`fovea-display` turns typed fovea images into display-ready pixels without hiding the decision of how values should be mapped to a screen.

That matters because a `Mono16` inspection frame, a linear `RgbF32` render target, and an already encoded `Srgb8` image need different display policies. This crate makes that policy explicit with display strategies.

## Install

Core display conversion has no windowing dependency:

```toml
[dependencies]
fovea = "0.1.1"
fovea-display = "0.1.1"
```

Enable local debug windows only when you want interactive inspection during development:

```toml
[dependencies]
fovea-display = { version = "0.1.1", features = ["debug-window"] }
```

## Features

| Feature | APIs | Intended use |
|---|---|---|
| *(default)* | `DisplayStrategy`, `Identity`, `LinearToDisplay`, `AutoContrast`, `FixedRange`, `TextureSource` | Convert typed images for a renderer or GUI without depending on a windowing stack. |
| `debug-window` | `show`, `DebugDisplay`, histogram debug windows | Quick local inspection of images and histograms while developing. |

## Pick a display strategy

Every display operation names how image data becomes `Srgba8` screen pixels.

| Strategy | Use when |
|---|---|
| `Identity` | Pixels are already display-encoded sRGB. |
| `LinearToDisplay` | Pixels are linear-light and need sRGB gamma encoding. |
| `AutoContrast` | You are debugging high-bit-depth or float data and want to see its current range. |
| `FixedRange` | You know the numeric range that should map to black/white. |

```rust
use fovea::pixel::{Srgb8, Srgba8};
use fovea_display::{DisplayStrategy, Identity};

let px = Srgb8::new(128, 64, 200);
let display = Identity.to_display(&px);

assert_eq!(display, Srgba8::new(128, 64, 200, 255));
```

## Preview an image during development

```rust,ignore
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea_display::{Identity, show};

let img = Image::fill(320, 240, Srgb8::new(128, 64, 200));
show("Preview", &img, Identity);
```

For linear or high-bit-depth data, do not use `Identity`. Choose the strategy that describes the display mapping:

```rust,ignore
use fovea::image::Image;
use fovea::pixel::Mono16;
use fovea_display::{AutoContrast, show};

let img = Image::generate(512, 512, |x, y| Mono16::new((x + y) as u16));
let strategy = AutoContrast::scan(&img);
show("Auto-contrast", &img, strategy);
```

## GPU and renderer integration

`TextureSource` is for renderer boundaries. It requires the stronger `PlainImage`/contiguous-byte-access path because GPU upload needs a stable memory layout, not just random pixel access.

Use `DisplayStrategy` when you need to decide what values should be shown. Use `TextureSource` when you need to hand bytes and texture metadata to a renderer.

## When NOT to use fovea-display

- You need a full GUI toolkit: use `egui`, `iced`, `winit`, or your application framework.
- You need video playback: use a media pipeline.
- You need color-management policy for a production display system: integrate fovea's typed pixels with your chosen color-management stack.

## Crate ecosystem

| Crate | Purpose |
|---|---|
| `fovea` | Core typed image model. |
| `fovea-io` | PNG/JPEG/BMP file boundaries. |
| `fovea-display` | Display mappings, texture metadata, and debug windows. |
| `fovea-examples` | Repo-only examples showing these crates together. |

## License

Licensed under the [MIT License](https://github.com/karhunen-loeve/fovea-display/blob/main/LICENSE).
