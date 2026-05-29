# fovea-display

[![Crates.io](https://img.shields.io/crates/v/fovea-display.svg)](https://crates.io/crates/fovea-display)
[![Documentation](https://docs.rs/fovea-display/badge.svg)](https://docs.rs/fovea-display)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`fovea-display` provides display strategies, GPU texture metadata, and optional debug windows for [`fovea`](https://github.com/karhunen-loeve/fovea) images.

```toml
[dependencies]
fovea = "0.1.0"
fovea-display = "0.1.0"
```

Enable the development-only window viewer with:

```toml
[dependencies]
fovea-display = { version = "0.1.0", features = ["debug-window"] }
```

## Features

| Feature | Enabled APIs | Dependencies | Intended use |
|---|---|---|---|
| *(default)* | `DisplayStrategy`, `Identity`, `LinearToDisplay`, `AutoContrast`, `TextureSource` | `fovea`, `log` | Conversion and rendering integration without a windowing dependency. |
| `debug-window` | `show`, `DebugDisplay`, histogram debug windows | `winit`, `softbuffer` | Quick local inspection during development. |

## Quick start

Convert a typed fovea pixel to a display-ready `Srgba8` value with an explicit strategy:

```rust
use fovea::pixel::{Srgb8, Srgba8};
use fovea_display::{DisplayStrategy, Identity};

let px = Srgb8::new(128, 64, 200);
let display = Identity.to_display(&px);

assert_eq!(display, Srgba8::new(128, 64, 200, 255));
```

With the `debug-window` feature enabled, inspect an image interactively:

```rust,ignore
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea_display::{show, Identity};

let img = Image::fill(320, 240, Srgb8::new(128, 64, 200));
show("Preview", &img, Identity);
```

## Display strategies

Every display operation names how image data becomes screen-ready RGBA pixels. There is no silent default mapping for high-bit-depth, linear-light, or HDR data.

| Strategy | Use case |
|---|---|
| `Identity` | sRGB pixels that are already display encoded. |
| `LinearToDisplay` | Linear-light pixels that need sRGB gamma encoding for a standard display. |
| `AutoContrast` | Scan an image and stretch its value range for debugging or inspection. |
| `FixedRange` | Map a caller-specified numeric range into display values. |

## Design notes

- `show()` accepts any `ImageView`, so owned images, ROIs, and custom image backends can be displayed.
- `TextureSource` requires contiguous byte access via `PlainImage`, which is the stronger bound needed for GPU upload.
- Display conversion is strategy-based to keep colour-space and range decisions explicit.

## Part of the fovea project

- Core crate: [`fovea`](https://github.com/karhunen-loeve/fovea)
- Codec support: [`fovea-io`](https://github.com/karhunen-loeve/fovea-io)
- End-to-end demos: [`fovea-examples`](https://github.com/karhunen-loeve/fovea-examples)

## License

Licensed under the [MIT License](LICENSE).
