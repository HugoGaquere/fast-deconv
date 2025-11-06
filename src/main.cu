#include <cstdio>
#include <cstddef>

#include "examples/kronecker_example.cuh"
#include "examples/wscms_example.cuh"


int main(int argc, char** argv)
{
  std::printf("Hello gpu world\n");

  run_kronecker_example();


  return 0;
}
