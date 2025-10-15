#pragma once

#include <algorithm>
#include <cassert>

struct Rect {
    int x0, x1; // [x0, x1) exclusive
    int y0, y1; // [y0, y1) exclusive

    __host__ __device__ int width()  const noexcept { return std::max(0, x1 - x0); }
    __host__ __device__ int height() const noexcept { return std::max(0, y1 - y0); }
    __host__ __device__ bool empty() const noexcept { return width() <= 0 || height() <= 0; }
    __host__ __device__ bool contains(int x, int y) const noexcept {
        return (x >= x0 && x < x1 && y >= y0 && y < y1);
    }
};

inline std::pair<Rect, Rect> compute_aligned_patch_edges(
    int x_center,
    int y_center,
    int image_width,
    int image_height,
    int psf_width,
    int psf_height) noexcept
{
    // Basic sanity checks
    assert(image_width  >= 0 && image_height >= 0);
    assert(psf_width    >= 0 && psf_height   >= 0);

    // Split PSF size into left/right and top/bottom halves
    // so the PSF interval around the center is [center - left, center + right)
    const int left   = psf_width  / 2;
    const int right  = psf_width  - left;   // covers even sizes cleanly
    const int top    = psf_height / 2;
    const int bottom = psf_height - top;

    // Desired PSF-aligned box in image coordinates
    const int want_x0 = x_center - left;
    const int want_x1 = x_center + right;
    const int want_y0 = y_center - top;
    const int want_y1 = y_center + bottom;

    // Clamp desired image region to the image bounds
    auto clamp_ex = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    const int img_x0 = clamp_ex(want_x0, 0, image_width);
    const int img_x1 = clamp_ex(want_x1, 0, image_width);
    const int img_y0 = clamp_ex(want_y0, 0, image_height);
    const int img_y1 = clamp_ex(want_y1, 0, image_height);

    Rect image_patch{img_x0, img_x1, img_y0, img_y1};

    // If nothing intersects the image, return empty/empty
    if (image_patch.empty()) {
        return { Rect{0,0,0,0}, Rect{0,0,0,0} };
    }

    // Offsets in PSF space due to clamping on the image side
    // (how much we had to crop on each side)
    const int crop_left   = img_x0 - want_x0;         // >= 0 if clamped on left
    const int crop_right  = want_x1 - img_x1;         // >= 0 if clamped on right
    const int crop_top    = img_y0 - want_y0;         // >= 0 if clamped on top
    const int crop_bottom = want_y1 - img_y1;         // >= 0 if clamped on bottom

    // PSF patch in PSF coordinates (exclusive-ended)
    // [x0_psf, x1_psf) × [y0_psf, y1_psf)
    const int psf_x0 = crop_left;
    const int psf_x1 = psf_width  - crop_right;
    const int psf_y0 = crop_top;
    const int psf_y1 = psf_height - crop_bottom;

    Rect psf_patch{psf_x0, psf_x1, psf_y0, psf_y1};

    // Keep both regions the same shape (they should already be)
    // but if extreme inputs created a mismatch, trim conservatively.
    const int w = std::min(image_patch.width(),  psf_patch.width());
    const int h = std::min(image_patch.height(), psf_patch.height());
    image_patch.x1 = image_patch.x0 + w;
    image_patch.y1 = image_patch.y0 + h;
    psf_patch.x1   = psf_patch.x0   + w;
    psf_patch.y1   = psf_patch.y0   + h;

    return { image_patch, psf_patch };
}
