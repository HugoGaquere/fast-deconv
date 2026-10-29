find_package(GTest CONFIG REQUIRED)

#-------------------------------------------------------------------#
# Unit tests (label UNIT). The sources are backend-agnostic: every
# transfer goes through core::exec_ctx's neutral primitives, so the
# same files compile for both backends.
#-------------------------------------------------------------------#
add_executable(fast_deconv_unit_tests
  test_convergence.cpp
  test_fft.cpp
  test_linalg.cpp
  test_matrix.cpp
  test_morphology.cpp
  test_mask.cpp
  test_tiled_argmax.cpp
  test_subtract.cpp
  test_gain.cpp
  test_multi_frequency.cpp
  test_core_parallel.cpp
  test_ddmsc_auto_mask.cpp
)

# Includes fd_backend/core/{reduce,for_each}.hpp directly, so under CUDA nvcc must compile it.
if(FAST_DECONV_WITH_CUDA)
  set_source_files_properties(test_core_parallel.cpp PROPERTIES LANGUAGE CUDA)
endif()

target_link_libraries(fast_deconv_unit_tests PRIVATE
  fast_deconv
  GTest::gtest_main
)

target_include_directories(fast_deconv_unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(FAST_DECONV_WITH_CUDA)
  target_compile_definitions(fast_deconv_unit_tests PRIVATE FAST_DECONV_WITH_CUDA)
endif()

include(GoogleTest)
gtest_discover_tests(fast_deconv_unit_tests PROPERTIES LABELS "UNIT")

#-------------------------------------------------------------------#
# Non-regression test (label NONREG): replays a captured DDMSC session
# and compares each cycle with the values in the test. Backend agnostic
# like the unit tests.
#-------------------------------------------------------------------#
add_executable(fast_deconv_nonreg_tests
  test_ddmsc_nonreg.cpp
)

target_link_libraries(fast_deconv_nonreg_tests PRIVATE
  fast_deconv
  GTest::gtest_main
)

# npy_loader.hpp and ddmsc_dump.hpp are shared with replay_ddmsc.
target_include_directories(fast_deconv_nonreg_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(fast_deconv_nonreg_tests PRIVATE
  FAST_DECONV_NONREG_DUMP_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/lofar_lba_point_sources"
)

gtest_discover_tests(fast_deconv_nonreg_tests PROPERTIES LABELS "NONREG" TIMEOUT 300)
