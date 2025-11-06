#include <cstdio>
#include <cstddef>

#include "kronecker_example.cuh"
#include "wscms_example.cuh"


int main(int argc, char** argv)
{
  std::printf("Hello gpu world\n");

  run_kronecker_example();


  return 0;
}
