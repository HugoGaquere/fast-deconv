#pragma once

#include <cmath>
#include <fast_deconv/algorithm/ddmsc.hpp>
#include <optional>
#include <string>

#include "npy_loader.hpp"

// Reads the files a FAST_DECONV_DUMP session writes (see fast_deconv/__init__.py).
namespace ddmsc_dump {

// A NaN scalar in the dump is an unset optional on the Python side.
inline std::optional<float> opt_finite(float v) { return std::isnan(v) ? std::nullopt : std::optional<float>(v); }

/// Every algorithm parameter of one dumped cycle; @p cycle_dir ends with a '/'.
inline fast_deconv::algorithm::ddmsc::params read_params(const std::string& cycle_dir)
{
  auto load = [&](const char* name) { return npy::load_npy(cycle_dir + name + ".npy"); };
  fast_deconv::algorithm::ddmsc::params p;
  p.clean_negative = load("clean_negative").scalar<bool>();
  p.peak_factor = load("peak_factor").scalar<float>();
  p.gamma = load("gamma").scalar<float>();
  p.max_clean_iteration = load("max_sub_iteration").scalar<int>();
  p.flux_threshold = load("flux_threshold").scalar<float>();
  p.stop_rms_factor = load("stop_rms_factor").scalar<float>();
  p.stop_peak_factor = load("stop_peak_factor").scalar<float>();
  p.stop_cycle_factor = load("stop_cycle_factor").scalar<float>();
  p.stop_sidelobe_level = load("stop_sidelobe_level").scalar<float>();
  p.max_iteration = load("max_iteration").scalar<int>();
  p.divergence_factor = load("divergence_factor").scalar<float>();
  p.scale_stall_threshold = load("stall_threshold").scalar<float>();
  p.enable_auto_mask = load("auto_mask").scalar<bool>();
  p.force_enable_auto_mask = load("force_auto_mask").scalar<bool>();
  p.auto_mask_peak_threshold = opt_finite(load("auto_mask_peak_threshold").scalar<float>());
  p.auto_mask_rms_threshold = opt_finite(load("auto_mask_rms_threshold").scalar<float>());
  return p;
}

}  // namespace ddmsc_dump
