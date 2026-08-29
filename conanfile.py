from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake, CMakeToolchain


class Recipe(ConanFile):
    name = "fast-deconv"
    version = "0.7.0"

    # Keep as it is
    settings = "os", "compiler", "build_type", "arch"

    # List of files to export to the conan cache when creating package.
    exports_sources = "CMakeLists.txt", "include/*", "src/*", "tests/*"

    options = {"python_module": [True, False], "backend": ["cuda", "host"]}
    default_options = {"python_module": False, "backend": "cuda"}

    options_descriptions = {
        "python_module": "Tells conan to adapt to the python module build",
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
        self.requires("gtest/1.15.0")
        self.requires("nlohmann_json/3.11.3")

        if self.options.backend == "host":
            # cache_size keeps the twiddle plans alive across calls; the
            # default of 0 re-plans on every transform.
            self.requires("pocketfft/0.0.0.cci.20240801", options={"cache_size": 8})

    def layout(self):
        if self.options.python_module:
            # Using conan as CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake_layout does not work
            # We don't want to pollute the build folder with conan. We put everything in "generators"
            self.folders.generators = "generators"
        else:
            # One tree per backend so cuda and host builds don't clobber each
            # other: build/release-cuda, build/release-host, and so on.
            self.folders.build_folder_vars = ["settings.build_type", "options.backend"]
            cmake_layout(self)

    generators = "CMakeConfigDeps"

    def generate(self):
        if not self.options.python_module:
            tc = CMakeToolchain(self)
            tc.cache_variables["FAST_DECONV_BACKEND"] = str(self.options.backend)
            tc.generate()

    def build(self):
        cmake = CMake(self)

        cmake.configure()
        cmake.build()

        # If you have test, consider uncommenting this
        # cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["fast-deconv"]
