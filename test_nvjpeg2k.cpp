#include <iostream>
#include <cuda_runtime.h>
#include <nvjpeg2k.h>

int main() {
    cudaError_t cuda_err = cudaFree(0);
    std::cout << "cudaFree(0) status = " << cuda_err << " (" << cudaGetErrorString(cuda_err) << ")" << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA Device Count = " << deviceCount << std::endl;
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "CUDA Device 0: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    }

    nvjpeg2kHandle_t handle;
    auto st = nvjpeg2kCreate(NVJPEG2K_BACKEND_DEFAULT, nullptr, nullptr, &handle);
    std::cout << "create status = " << st << std::endl;

    if (st == NVJPEG2K_STATUS_SUCCESS) {
        nvjpeg2kDecodeState_t state;
        st = nvjpeg2kDecodeStateCreate(handle, &state);
        std::cout << "decode state status = " << st << std::endl;
        if (st == NVJPEG2K_STATUS_SUCCESS) {
            nvjpeg2kDecodeStateDestroy(state);
        }
        nvjpeg2kDestroy(handle);
    }

    return 0;
}