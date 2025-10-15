#include <driver_types.h>
#include <limits>
#include <utility>

#include "rect.h"
#include "wscms.h"

extern "C" std::pair<int, float> argmax(float* data, bool* mask, size_t size, bool use_abs);



void _clean_dirty_kernel(Float *dirty, Float *psf, uint patch_width, uint patch_height,
                         Rect dirty_patch, Rect psf_patch, float gain, float coeffs) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        uint row = idx / patch_width;
        uint col = idx % patch_width;
        uint dirty_idx = (dirty_patch.y0 + row) * dirty_width + (dirty_patch.x0 + col); 
        uint psf_idx = (psf_patch.y0 + row) * psf_width + (psf_patch.x0 + col); 
        dirty[dirty_idx] -= psf[psf_idx] * coeffs * gain;
    }
   
}

void clean_dirty_launcher(Gpu2D<float> dirty, Gpu2D<float> psf, Rect dirty_patch, Rect psf_patch, float gain, float coeffs, cudaStream_t stream) {
    // FIXME: Fine tune conf
    _clean_dirty_kernel<<<32*40, 256>>>(
        dirty.data(), psf.data(), dirty.shape(1), dirty.shape(0),
        dirty_patch.width(), dirty_patch.height(), dirty_patch, psf_patch,
        gain, coeffs);
}




SubminorLoopResults WSCMS::run_subminor_loop(Gpu2D<float> dirty, Gpu2D<float> scaled_dirty, Gpu2D<bool> mask, uint scale_idx, cudaStream_t stream) {
    
    // ===== SUMARRY =====  
    // 1. Find peak x,y
    // 2. Load Facet / PSF at pos x,y
    // 3. Compute sky model component
    // 4. Compute aligned patch edges
    // 5. Clean dirty
    //   a. compute flux_scaled_dirty
    //   b. subtract flux scaled_dirty from dirty
    // 6. Clean Scaled Dirty
    //   a. scaled_dirty - conv2_PSF
    
    SubminorLoopResults results;

    // TODO: Add do_absolute argument
    size_t nof_elements = dirty.size();

    auto max_dirty = argmax(dirty.data(), mask.data(), nof_elements, false);
    size_t peak_idx = max_dirty.first;
    float peak_value = max_dirty.second;
    printf("Dirty max_dirty = %f at %d/n",
           max_dirty.second, max_dirty.first);

    float threshold = this->_peak_factor * peak_value;
    // Update the mask where scaled_dirty > threshold and mask==False
    // We do not follow Cyril's convention (searching for peaks where mask==False)
    // Here we search for peaks where mask==True
    //
    // MAYBE, we do not need to update the mask 
    
    uint n_iter = 0;
    uint max_iter = 1000; // TODO: pass it as an arguments
    while (peak_value > threshold && n_iter < max_iter) {
        
        uint peak_x = peak_idx / dirty.shape(1);
        uint peak_y = peak_idx % dirty.shape(0);

        // 2. Find facet idx
        uint facet_idx = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (uint i; i < this->_facets.centers.size(); i++) {
            float l = this->_cell_size_radian.x * (peak_x - static_cast<float>(dirty.shape(0)) / 2);
            float m = this->_cell_size_radian.y * (peak_y - static_cast<float>(dirty.shape(1)) / 2);
            float2 facet_center = this->_facets.centers[i];
            float distance = std::sqrt((l - facet_center.x) * (l - facet_center.x) + (m - facet_center.y) * (m - facet_center.y));
            if (distance < min_dist) {
                facet_idx = i;
                min_dist = distance;
            }
        }

        float gain = this->_gains[facet_idx];
        float coeffs = peak_value;
        
        // 3. Save sky model component
        results.components_center.push_back({peak_x, peak_y});
        results.sols.push_back(peak_value);
        results.scales_idx.push_back(scale_idx);
        results.gains.push_back(gain);

        auto [dirty_patch, psf_patch] = compute_aligned_patch_edges(
            peak_x, peak_y, dirty.shape(0), dirty.shape(1), this->_conv_PSFs.shape(0), this->_conv_PSFs.shape(1));

        // Clean dirty image
        auto psf = this->_conv_PSFs(facet_idx);
        clean_dirty_launcher(dirty, psf, dirty_patch, psf_patch, gain, coeffs, stream);

        // Clean Scaled image


    }

 
    return results;
}
