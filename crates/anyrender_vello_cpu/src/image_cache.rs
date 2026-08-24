//! A per-renderer cache of converted images.
//!
//! Converting a [`peniko::ImageData`] to a vello_cpu `Pixmap` (format swizzle +
//! alpha premultiplication) is expensive, so converted images are registered in
//! the vello_cpu image registry once and subsequent draws reference them by
//! [`ImageId`].

use peniko::WeakBlob;
use std::collections::HashMap;
use std::sync::Arc;
use vello_common::color::PremulRgba8;
use vello_common::fearless_simd::{Level, Simd, SimdBase, SimdInt, SimdMask, dispatch, mask8x16};
use vello_common::paint::{ImageId, ImageSource};
use vello_common::util::Div255Ext;
use vello_cpu::{Pixmap, Resources};

/// Configuration for the image cache eviction policy.
#[derive(Clone, Copy, Debug)]
pub struct ImageCacheConfig {
    /// Evict entries unused for this many frames.
    pub max_age: u64,
    /// Soft cap on total converted-pixmap bytes.
    pub max_bytes: usize,
    /// Only walk the cache every N frames unless over budget.
    pub prune_interval: u64,
}

impl Default for ImageCacheConfig {
    fn default() -> Self {
        Self {
            max_age: 64,
            max_bytes: 64 * 1024 * 1024,
            prune_interval: 8,
        }
    }
}

struct Entry {
    image_id: ImageId,
    may_have_transparency: bool,
    bytes: usize,
    last_used: u64,
    /// Weak handle to the source data. When the embedder drops the last strong
    /// reference (e.g. a document is destroyed), the entry is evicted at the
    /// next prune.
    source: WeakBlob<u8>,
}

/// Caches image conversions, keyed on the source blob's unique id.
#[derive(Default)]
pub(crate) struct ImageCache {
    entries: HashMap<u64, Entry>,
    serial: u64,
    total_bytes: usize,
    config: ImageCacheConfig,
}

impl ImageCache {
    pub(crate) fn new(config: ImageCacheConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Look up the converted image for `image`, converting and registering it
    /// in `resources` on a cache miss.
    pub(crate) fn get_or_register(
        &mut self,
        resources: &mut Resources,
        image: &peniko::ImageData,
    ) -> ImageSource {
        let serial = self.serial;
        let total_bytes = &mut self.total_bytes;
        let entry = self.entries.entry(image.data.id()).or_insert_with(|| {
            let pixmap = convert_image(image);
            let may_have_transparency = pixmap.may_have_transparency();
            let bytes = pixmap.data().len() * 4;
            *total_bytes += bytes;
            Entry {
                image_id: resources.register_image(pixmap),
                may_have_transparency,
                bytes,
                last_used: serial,
                source: image.data.downgrade(),
            }
        });
        entry.last_used = serial;
        ImageSource::opaque_id_with_transparency_hint(entry.image_id, entry.may_have_transparency)
    }

    /// Advance the frame counter and evict stale entries.
    ///
    /// Must be called once per frame, after rendering (so that ids referenced
    /// by the just-rendered scene are not destroyed before being resolved).
    pub(crate) fn maintain(&mut self, resources: &mut Resources) {
        self.serial += 1;
        let over_budget = self.total_bytes > self.config.max_bytes;
        if !over_budget && !self.serial.is_multiple_of(self.config.prune_interval) {
            return;
        }

        // Evict entries whose source data has been dropped or that have not
        // been used recently.
        let serial = self.serial;
        let max_age = self.config.max_age;
        let total_bytes = &mut self.total_bytes;
        self.entries.retain(|_, entry| {
            let keep = entry.source.upgrade().is_some() && serial - entry.last_used <= max_age;
            if !keep {
                *total_bytes -= entry.bytes;
                resources.destroy_image(entry.image_id);
            }
            keep
        });

        // Evict least-recently-used entries until under the byte budget,
        // skipping entries used by the frame that was just rendered.
        while self.total_bytes > self.config.max_bytes {
            let key = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.last_used + 1 != serial)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key);
            let Some(key) = key else { break };
            let entry = self.entries.remove(&key).unwrap();
            self.total_bytes -= entry.bytes;
            resources.destroy_image(entry.image_id);
        }
    }

    /// Drop all cached conversions and their registry entries.
    pub(crate) fn clear(&mut self, resources: &mut Resources) {
        for (_, entry) in self.entries.drain() {
            resources.destroy_image(entry.image_id);
        }
        self.total_bytes = 0;
    }
}

/// Convert a [`peniko::ImageData`] to a premultiplied RGBA8 [`Pixmap`].
///
/// Equivalent to `ImageSource::from_peniko_image_data`, but with a SIMD
/// premultiply vendored from <https://github.com/linebender/vello/pull/1834>.
/// TODO: use `from_peniko_image_data` directly once a vello_cpu release
/// includes that PR.
///
/// # Panics
///
/// Panics if `image` has a `width` or `height` greater than `u16::MAX`.
fn convert_image(image: &peniko::ImageData) -> Arc<Pixmap> {
    assert!(
        image.width <= u16::MAX as u32 && image.height <= u16::MAX as u32,
        "The image is too big. Its width and height can be no larger than {} pixels.",
        u16::MAX,
    );
    let width = image.width.try_into().unwrap();
    let height = image.height.try_into().unwrap();

    let data = image.data.data();
    let pixels: Vec<PremulRgba8> = match image.format {
        peniko::ImageFormat::Rgba8 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| PremulRgba8 {
                r: p[0],
                g: p[1],
                b: p[2],
                a: p[3],
            })
            .collect(),
        peniko::ImageFormat::Bgra8 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| PremulRgba8 {
                r: p[2],
                g: p[1],
                b: p[0],
                a: p[3],
            })
            .collect(),
        format => unimplemented!("Unsupported image format: {format:?}"),
    };

    let mut pixmap = Pixmap::from_parts_with_opacity(pixels, width, height, true);
    if image.alpha_type == peniko::ImageAlphaType::AlphaPremultiplied {
        pixmap.recompute_may_have_transparency();
    } else {
        let may_have_transparency = premultiply_rgba8(pixmap.data_as_u8_slice_mut());
        pixmap.set_may_have_transparency(may_have_transparency);
    }
    Arc::new(pixmap)
}

/// Premultiplies each RGBA8 pixel in `data`.
///
/// Returns `true` if at least one pixel is not fully opaque.
///
/// Vendored from <https://github.com/linebender/vello/pull/1834>.
fn premultiply_rgba8(data: &mut [u8]) -> bool {
    let level = Level::try_detect().unwrap_or(Level::baseline());

    dispatch!(level, simd => premultiply_rgba8_impl(simd, data))
}

#[inline(always)]
fn premultiply_rgba8_impl<S: Simd>(simd: S, data: &mut [u8]) -> bool {
    let (body, tail) = data.as_chunks_mut::<64>();
    let mut transparency = mask8x16::splat(simd, 0);

    for chunk in body {
        let rgba = simd.load_interleaved_128_u8x64(chunk);
        let (rg, ba) = simd.split_u8x64(rgba);
        let (r, g) = simd.split_u8x32(rg);
        let (b, a) = simd.split_u8x32(ba);

        transparency |= !a.simd_eq(255);
        let premultiply = {
            #[inline(always)]
            |component| {
                let product = simd.widen_u8x16(component) * simd.widen_u8x16(a);
                simd.narrow_u16x16(product.div_255())
            }
        };
        let premultiplied = simd.combine_u8x32(
            simd.combine_u8x16(premultiply(r), premultiply(g)),
            simd.combine_u8x16(premultiply(b), a),
        );
        simd.store_interleaved_128_u8x64(premultiplied, chunk);
    }

    let mut may_have_transparency = transparency.any_true();
    for pixel in tail.as_chunks_mut::<4>().0 {
        let alpha = u16::from(pixel[3]);
        may_have_transparency |= alpha != 255;
        let premultiply = |component| ((u16::from(component) * alpha + 255) >> 8) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }

    may_have_transparency
}

#[cfg(test)]
mod tests {
    use super::*;
    use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};

    fn image(pixels: &[[u8; 4]]) -> ImageData {
        ImageData {
            data: Blob::new(Arc::new(pixels.concat())),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: pixels.len() as u32,
            height: 1,
        }
    }

    #[test]
    fn premultiply() {
        let pixmap = convert_image(&image(&[[100, 150, 200, 128], [10, 20, 30, 255]]));
        assert!(pixmap.may_have_transparency());
        let px = pixmap.data()[0];
        assert_eq!((px.r, px.g, px.b, px.a), (50, 75, 100, 128));
        let px = pixmap.data()[1];
        assert_eq!((px.r, px.g, px.b, px.a), (10, 20, 30, 255));
    }

    #[test]
    fn premultiply_bgra_opaque() {
        let mut img = image(&[[1, 2, 3, 255]; 20]);
        img.format = ImageFormat::Bgra8;
        let pixmap = convert_image(&img);
        assert!(!pixmap.may_have_transparency());
        let px = pixmap.data()[0];
        assert_eq!((px.r, px.g, px.b, px.a), (3, 2, 1, 255));
    }

    #[test]
    fn cache_hit_reuses_registered_image() {
        let mut cache = ImageCache::new(ImageCacheConfig::default());
        let mut resources = Resources::new();
        let img = image(&[[0, 0, 0, 255]]);
        let a = cache.get_or_register(&mut resources, &img);
        let b = cache.get_or_register(&mut resources, &img);
        let ImageSource::OpaqueId { id: id_a, .. } = a else {
            panic!("expected OpaqueId");
        };
        let ImageSource::OpaqueId { id: id_b, .. } = b else {
            panic!("expected OpaqueId");
        };
        assert_eq!(id_a, id_b);
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn dead_blob_is_evicted() {
        let mut cache = ImageCache::new(ImageCacheConfig {
            prune_interval: 1,
            ..Default::default()
        });
        let mut resources = Resources::new();
        let img = image(&[[0, 0, 0, 255]]);
        let ImageSource::OpaqueId { id, .. } = cache.get_or_register(&mut resources, &img) else {
            panic!("expected OpaqueId");
        };
        drop(img);
        cache.maintain(&mut resources);
        assert!(cache.entries.is_empty());
        assert_eq!(cache.total_bytes, 0);
        assert!(resources.resolve_image(id).is_none());
    }

    #[test]
    fn old_entries_age_out() {
        let mut cache = ImageCache::new(ImageCacheConfig {
            max_age: 2,
            prune_interval: 1,
            ..Default::default()
        });
        let mut resources = Resources::new();
        let img = image(&[[0, 0, 0, 255]]);
        cache.get_or_register(&mut resources, &img);
        for _ in 0..2 {
            cache.maintain(&mut resources);
        }
        assert_eq!(cache.entries.len(), 1);
        cache.maintain(&mut resources);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn byte_budget_evicts_lru_but_not_current_frame() {
        let mut cache = ImageCache::new(ImageCacheConfig {
            max_bytes: 4,
            ..Default::default()
        });
        let mut resources = Resources::new();
        let old = image(&[[0, 0, 0, 255]]);
        let new = image(&[[1, 1, 1, 255]]);
        cache.get_or_register(&mut resources, &old);
        cache.maintain(&mut resources);
        cache.get_or_register(&mut resources, &new);
        cache.maintain(&mut resources);
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.entries.contains_key(&new.data.id()));
    }

    #[test]
    fn clear_destroys_registry_entries() {
        let mut cache = ImageCache::new(ImageCacheConfig::default());
        let mut resources = Resources::new();
        let img = image(&[[0, 0, 0, 255]]);
        let ImageSource::OpaqueId { id, .. } = cache.get_or_register(&mut resources, &img) else {
            panic!("expected OpaqueId");
        };
        cache.clear(&mut resources);
        assert!(cache.entries.is_empty());
        assert!(resources.resolve_image(id).is_none());
    }
}
