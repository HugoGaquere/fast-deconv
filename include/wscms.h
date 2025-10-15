#pragma once
#include <vector>
#include <vector_types.h>

#include "gpu_utils.h"

// Notes:
// For now, we asssume 1 channel and 1 polarization

template<typename T>
struct Facets {
    std::vector<std::vector<T>> edges;
    std::vector<float2> centers;
};

          
//  Sols: Nd array of coeffs with length equal to the number of basis functions representing the component.
//  iScale: the scale index
//  Gain: clean loop gain
struct SubminorLoopResults {
    std::vector<std::pair<uint, uint>> components_center; //  key: the (l,m) centre of the component in pixels
    std::vector<float> sols;
    std::vector<int> scales_idx;
    std::vector<float> gains;
};


class WSCMS {
public:
    WSCMS() {}; 
    WSCMS(Facets<float> facets, Gpu3D<float> PSFs, ) : _facets(facets), _PSFs(PSFs) {}; 
    SubminorLoopResults  run_subminor_loop(Gpu2D<float> dirty, Gpu2D<float> scaled_dirty, Gpu2D<bool> mask, uint scale_idx);
private:
    Facets<float> _facets;
    Gpu3D<float> _PSFs;
    Gpu3D<float> _conv_PSFs;
    Gpu3D<float> _conv_PSFs_mean;
    std::vector<float> _gains;
    float2 _cell_size_radian;
    float _peak_factor;
    float _gamma;
};
