find_package(GTest CONFIG REQUIRED)

# Mark GTest targets as SYSTEM so clangd doesn't surface their warnings.
foreach(_tp_target IN ITEMS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
  if(TARGET ${_tp_target})
    set_target_properties(${_tp_target} PROPERTIES SYSTEM TRUE)
  endif()
endforeach()

#-------------------------------------------------------------------#
# Unit tests (label UNIT): fast, one module per file, CPU oracles.
#-------------------------------------------------------------------#
add_executable(fast_deconv_unit_tests
  test_main.cpp
  test_convergence.cpp
  test_resources.cu
  test_utils.cu
  test_fft.cu
  test_linalg.cu
  test_pseudo_inverse.cu
  test_matrix.cu
  test_morphology.cu
  test_mask.cu
  test_tiled_argmax.cu
  test_clean.cu
  test_gain.cu
  test_multi_frequency.cu
  test_scales.cu
  test_wscms_class.cu
)

target_link_libraries(fast_deconv_unit_tests PRIVATE
  fast_deconv
  GTest::gtest
)

target_include_directories(fast_deconv_unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(fast_deconv_unit_tests PROPERTIES LABELS "UNIT")

#-------------------------------------------------------------------#
# Non-regression test (label NONREG): full synthetic WSCMS run whose
# scalar metrics are compared against a committed JSON baseline.
# Regenerate the baseline with FAST_DECONV_UPDATE_BASELINE=1.
#-------------------------------------------------------------------#
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(fast_deconv_nonreg_tests
  test_main.cpp
  test_wscms_nonreg.cu
)

target_link_libraries(fast_deconv_nonreg_tests PRIVATE
  fast_deconv
  GTest::gtest
  nlohmann_json::nlohmann_json
)

target_include_directories(fast_deconv_nonreg_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Baseline lives in the source tree so a regeneration shows up as a git diff.
target_compile_definitions(fast_deconv_nonreg_tests PRIVATE
  FAST_DECONV_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../baselines/wscms_synthetic.json"
)

gtest_discover_tests(fast_deconv_nonreg_tests PROPERTIES LABELS "NONREG" TIMEOUT 300)
