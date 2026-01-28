#pragma once
#include "fast_deconv/core/stream_resources.hpp"

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/access_policy.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/kernel_traits.hpp>

#include <type_traits>

namespace fast_deconv::core {

template <typename KernelTag, typename F, typename... Mdspans>
  requires(cpts::mdspan<std::remove_cvref_t<Mdspans>> && ...)
void dispatch(stream_resources& resources, F&& function, Mdspans&&... mdspans)
{
  // TODO: Assert kernel trait validity
  using Traits = kernel_traits<KernelTag>;

  // Load / Store policy
  AccessPolicy access_policy = determine_policy<KernelTag>(mdspans...);

  // fmt::println("DISPATCHER");
  // fmt::println("access_policy  {} {}",
  //              static_cast<int>(access_policy.load_policy),
  //              static_cast<int>(access_policy.store_policy));

  function(access_policy, resources, std::forward<Mdspans>(mdspans)...);

}

}  // namespace fast_deconv::core
