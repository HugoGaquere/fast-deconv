#pragma once
#include "fast_deconv/core/resources.hpp"

#include <emu/cuda/device/mdspan.hpp>
#include <fast_deconv/core/access_policy.hpp>
#include <fast_deconv/core/concepts.hpp>
#include <fast_deconv/core/kernel_traits.hpp>

#include <tuple>
#include <type_traits>

namespace fast_deconv::core {

namespace detail {

template <typename KernelTag, typename... Args>
AccessPolicy determine_policy_from_args([[maybe_unused]] const Args&... args)
{
  // Filter only mdspan arguments for policy determination
  if constexpr ((cpts::mdspan<std::remove_cvref_t<Args>> || ...)) {
    auto collect = [](const auto&... mdspans) { return determine_policy<KernelTag>(mdspans...); };
    // Apply collect only to mdspan args
    auto filter = [&](const auto&... all_args) {
      return std::apply(collect, std::tuple_cat([](const auto& arg) {
                          if constexpr (cpts::mdspan<std::remove_cvref_t<decltype(arg)>>)
                            return std::tie(arg);
                          else
                            return std::tuple<>();
                        }(all_args)...));
    };
    return filter(args...);
  } else {
    return {.load_policy = AccessType::Scalar, .store_policy = AccessType::Scalar};
  }
}

}  // namespace detail

template <typename KernelTag, typename F, typename... Args>
void dispatch(stream_resources& resources, F&& function, Args&&... args)
{
  // TODO: Assert kernel trait validity
  using Traits = kernel_traits<KernelTag>;

  // Load / Store policy (determined from mdspan arguments only)
  AccessPolicy access_policy = detail::determine_policy_from_args<KernelTag>(args...);

  function(access_policy, resources, std::forward<Args>(args)...);
}

}  // namespace fast_deconv::core
