#pragma once

template <typename T>
inline void fill_with_random(T view)
{
  for (int i = 0; i < view.size(); i++) {
    view.data_handle()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
}

void run_kronecker_tensor_example();
