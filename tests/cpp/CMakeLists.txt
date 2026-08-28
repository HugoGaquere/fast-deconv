find_package(GTest CONFIG REQUIRED)

#-------------------------------------------------------------------#
# Unit tests (label UNIT). The sources are backend-agnostic: every
# transfer goes through core::exec_ctx's neutral primitives, so the
# same files compile for both backends.
#
# They only PASS on cuda — each host kernel is still a stub that throws
# "host backend not implemented yet", so on host the ones that drive a
# kernel fail until it lands.
#-------------------------------------------------------------------#
add_executable(fast_deconv_unit_tests
  test_convergence.cpp
  test_resources.cpp
  test_fft.cpp
  test_linalg.cpp
  test_pseudo_inverse.cpp
  test_matrix.cpp
  test_morphology.cpp
  test_mask.cpp
  test_tiled_argmax.cpp
  test_clean.cpp
  test_gain.cpp
  test_multi_frequency.cpp
  test_scales.cpp
  test_ddmsc_class.cpp
)

target_link_libraries(fast_deconv_unit_tests PRIVATE
  fast_deconv
  GTest::gtest_main
)

target_include_directories(fast_deconv_unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(fast_deconv_unit_tests PROPERTIES LABELS "UNIT")

#-------------------------------------------------------------------#
# Non-regression test (label NONREG): full synthetic DDMSC run whose
# scalar metrics are compared against a committed JSON baseline.
# Still cuda-only: not ported alongside the rest.
# Regenerate the baseline with FAST_DECONV_UPDATE_BASELINE=1.
#-------------------------------------------------------------------#
if(FAST_DECONV_WITH_CUDA)
  find_package(nlohmann_json CONFIG REQUIRED)

  add_executable(fast_deconv_nonreg_tests
    test_ddmsc_nonreg.cu
  )

  target_link_libraries(fast_deconv_nonreg_tests PRIVATE
    fast_deconv
    GTest::gtest_main
    nlohmann_json::nlohmann_json
  )

  target_include_directories(fast_deconv_nonreg_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # Baseline lives in the source tree so a regeneration shows up as a git diff.
  target_compile_definitions(fast_deconv_nonreg_tests PRIVATE
    FAST_DECONV_BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../baselines/ddmsc_synthetic.json"
  )

  gtest_discover_tests(fast_deconv_nonreg_tests PROPERTIES LABELS "NONREG" TIMEOUT 300)
endif()
