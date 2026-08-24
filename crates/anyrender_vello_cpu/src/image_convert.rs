//! Conversion of [`peniko::ImageData`] to vello_cpu [`Pixmap`]s.

use std::sync::Arc;
use vello_common::color::PremulRgba8;
use vello_common::fearless_simd::{Level, Simd, SimdBase, SimdInt, SimdMask, dispatch, mask8x16};
use vello_common::util::Div255Ext;
use vello_cpu::Pixmap;

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
pub(crate) fn convert_image(image: &peniko::ImageData) -> Arc<Pixmap> {
    assert!(
        image.width <= u16::MAX as u32 && image.height <= u16::MAX as u32,
        "The image is too big. Its width and height can be no larger than {} pixels.",
        u16::MAX,
    );
    let width = image.width.try_into().unwrap();
    let height = image.height.try_into().unwrap();

    let data = image.data.data();
    // Bulk-copy the source bytes (RGBA8 is byte-identical to `PremulRgba8`)
    // rather than converting pixel-by-pixel, then mutate in place.
    let pixel_bytes = data.len() & !3;
    let mut bytes = Vec::with_capacity(pixel_bytes);
    bytes.extend_from_slice(&data[..pixel_bytes]);

    match image.format {
        peniko::ImageFormat::Rgba8 => {}
        peniko::ImageFormat::Bgra8 => {
            for pixel in bytes.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        format => unimplemented!("Unsupported image format: {format:?}"),
    }

    let premultiplied = image.alpha_type == peniko::ImageAlphaType::AlphaPremultiplied;
    let may_have_transparency = if premultiplied {
        bytes.as_chunks::<4>().0.iter().any(|p| p[3] != 255)
    } else {
        premultiply_rgba8(&mut bytes)
    };

    let pixels: Vec<PremulRgba8> = bytemuck::try_cast_vec(bytes).unwrap_or_else(|(_, bytes)| {
        // Fall back to copying if the allocation is incompatible with an
        // in-place cast (e.g. over-allocated capacity).
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&p| PremulRgba8::from_u8_array(p))
            .collect()
    });

    Arc::new(Pixmap::from_parts_with_opacity(
        pixels,
        width,
        height,
        may_have_transparency,
    ))
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
}
