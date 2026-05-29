#pragma once

#include <emu/pybind11.hpp>

#include <pybind11/pybind11.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <string_view>

#include <iostream>

namespace fmt
{

// nvcc 12.x cannot disambiguate two sibling concept-constrained partial
// specializations of the same template; merge into one disjunctive form.
template <typename T>
    requires emu::pybind11::cpts::derived_from_handle<T>
          || emu::pybind11::cpts::accessor<T>
struct formatter<T> : fmt::formatter<std::string> {
    using base = fmt::formatter<std::string>;

    std::string_view format_spec = "{}";

    constexpr auto parse(format_parse_context& ctx) {
        auto it = ctx.begin(), end = ctx.end();

        // If spec is empty, return directly.
        if (it == end or *it == '}') return it;

        format_spec = std::string_view(it - 2, (end - it + 2));

        // Advance iterator to the end.
        for(;it != end and *it != '}'; it++);

        return it;
    }

    template <typename FormatContext>
    auto format(const T& obj, FormatContext& ctx) const {
        auto formatted_string = pybind11::str(format_spec.data(), format_spec.size()).format(obj);

        return base::format(static_cast<std::string>(formatted_string), ctx);
    }
};

    // Specialization for nanobind python based type to never be treated as a range by fmt.
    template <typename T, typename Char>
        requires emu::pybind11::cpts::derived_from_handle<T>
              || emu::pybind11::cpts::accessor<T>
    struct is_range<T, Char>: std::false_type{};

} // namespace fmt
