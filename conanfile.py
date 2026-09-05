from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain


class Recipe(ConanFile):
    name = "fast-deconv"
    version = "0.7.0"

    # Conan supplies dependencies and CMake configuration for local builds.
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "python_module": [True, False],
        "backend": ["cuda", "host"],
        "with_tests": [True, False],
    }
    default_options = {"python_module": False, "backend": "cuda", "with_tests": True}

    options_descriptions = {
        "python_module": "Tells conan to adapt to the python module build",
        "with_tests": "Resolve C++ test dependencies and enable BUILD_TESTING",
        "backend": "Compute backend to build: cuda | host. Drives FAST_DECONV_BACKEND "
                   "and whether emu is pulled in with its CUDA extension.",
    }

    def requirements(self):
        self.requires("fmt/11.2.0")
        self.requires("spdlog/1.15.3")
        self.requires(
            "emu/0.1.0-rc.7",
            options={
                "python": self.options.python_module,
                "cuda": self.options.backend == "cuda",
            },
        )

        if self.options.backend == "host":
            # cache_size keeps the twiddle plans alive across calls; the
            # default of 0 re-plans on every transform.
            self.requires("pocketfft/0.0.0.cci.20240801", options={"cache_size": 8})
            # NVTX3 headers on their own; the cuda build takes them from the toolkit.
            self.requires("nvtx/3.3.0")

    @property
    def build_tests(self):
        return (
            bool(self.options.with_tests)
            and not self.options.python_module
            and not self.conf.get("tools.build:skip_test", default=False, check_type=bool)
        )

    def build_requirements(self):
        if self.build_tests:
            self.test_requires("gtest/1.15.0")
            self.test_requires("nlohmann_json/3.11.3")

    def layout(self):
        if self.options.python_module:
            # Using conan as CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake_layout does not work
            # We don't want to pollute the build folder with conan. We put everything in "generators"
            self.folders.generators = "generators"
        else:
            # One tree per backend so cuda and host builds don't clobber each
            # other: build/release-backend_cuda, build/release-backend_host, and so on.
            self.folders.build_folder_vars = ["settings.build_type", "options.backend"]
            cmake_layout(self)

    generators = "CMakeConfigDeps"

    def generate(self):
        if not self.options.python_module:
            tc = CMakeToolchain(self)
            tc.cache_variables["FAST_DECONV_BACKEND"] = str(self.options.backend)
            tc.cache_variables["BUILD_TESTING"] = self.build_tests
            tc.generate()
