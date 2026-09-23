#pragma once

#ifdef __CUDACC__
#define FD_HOST_DEVICE __host__ __device__
#else
#define FD_HOST_DEVICE
#endif
