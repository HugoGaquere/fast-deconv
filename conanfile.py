from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake, CMakeToolchain

class Recipe(ConanFile):
    name = "fast-deconv"
    version = "0.0.1"

    # Keep as it is
    settings = "os", "compiler", "build_type", "arch"

    # List of files to export to the conan cache when creating package.
    exports_sources = 'CMakeLists.txt', 'include/*', 'src/*', 'test/*'

    options = {"python_module": [True, False]}
    default_options = {"python_module": False}

    options_descriptions = {
        "python_module": "Tells conan to adapt to the python module build",
    }

    def requirements(self):
        self.requires("fmt/11.2.0")
        self.requires('emu/0.1.0-rc.3',
                      options={"python": self.options.python_module, "cuda": True}
                      )

    def layout(self):
        if self.options.python_module:
            # Using conan as CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake_layout does not work
            # We don't want to pollute the build folder with conan. We put everything in "generators"
            self.folders.generators = "generators"
        else:
            # Otherwise, we use the default cmake layout
            cmake_layout(self)

    generators = "CMakeConfigDeps"

    def generate(self):
        if not self.options.python_module:
            tc = CMakeToolchain(self)
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
        self.cpp_info.libs = ['fast-deconv']
