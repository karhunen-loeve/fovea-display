#![doc = include_str!("../README.md")]

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
