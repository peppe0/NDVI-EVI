#include <cuda_runtime.h>
#include <nvjpeg2k.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

#define CHECK_CUDA(call)                                     \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            std::cerr << "CUDA Error: "                    \
                      << cudaGetErrorString(err)             \
                      << std::endl;                          \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

#define CHECK_NVJPEG2K(call)                                 \
    do {                                                     \
        nvjpeg2kStatus_t status = call;                      \
        if (status != NVJPEG2K_STATUS_SUCCESS) {             \
            std::cerr << "nvJPEG2000 Error: "              \
                      << status                              \
                      << std::endl;                          \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

std::vector<unsigned char> load_file(const char* filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open input file");
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> buffer(size);

    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0); // Disable stdout buffering

    if (argc < 2) {
        std::cerr << "Usage: ./test_nvjpeg2k image.jp2" << std::endl;
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];

    // Initialize CUDA context explicitly to prevent execution failures in library
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(0));

    //----------------------------------------------------------
    // Print nvJPEG2000 version
    //----------------------------------------------------------

    int major = 0;
    int minor = 0;
    int patch = 0;

    CHECK_NVJPEG2K(
        nvjpeg2kGetProperty(MAJOR_VERSION, &major)
    );

    CHECK_NVJPEG2K(
        nvjpeg2kGetProperty(MINOR_VERSION, &minor)
    );

    CHECK_NVJPEG2K(
        nvjpeg2kGetProperty(PATCH_LEVEL, &patch)
    );

    std::cout << "nvJPEG2000 version: "
              << major << "."
              << minor << "."
              << patch << std::endl;

    //----------------------------------------------------------
    // Create CUDA stream
    //----------------------------------------------------------

    cudaStream_t stream;

    CHECK_CUDA(
        cudaStreamCreate(&stream)
    );

    //----------------------------------------------------------
    // Create nvJPEG2000 handle
    //----------------------------------------------------------

    std::cout << "Creating nvJPEG2000 handle..." << std::endl;
    nvjpeg2kHandle_t handle;

    CHECK_NVJPEG2K(
        nvjpeg2kCreateSimple(&handle)
    );

    std::cout << "Creating nvJPEG2000 decode state..." << std::endl;
    nvjpeg2kDecodeState_t decode_state;

    CHECK_NVJPEG2K(
        nvjpeg2kDecodeStateCreate(handle, &decode_state)
    );

    std::cout << "Successfully created library handles." << std::endl;

    //----------------------------------------------------------
    // Load JPEG2000 file
    //----------------------------------------------------------

    std::cout << "Loading file: " << filename << std::endl;

    auto compressed_data = load_file(filename);

    std::cout << "Compressed size: "
              << compressed_data.size()
              << " bytes"
              << std::endl;

    //----------------------------------------------------------
    // Parse JPEG2000 stream
    //----------------------------------------------------------

    nvjpeg2kStream_t jpeg2k_stream;

    CHECK_NVJPEG2K(
        nvjpeg2kStreamCreate(&jpeg2k_stream)
    );

    CHECK_NVJPEG2K(
        nvjpeg2kStreamParse(
            handle,
            compressed_data.data(),
            compressed_data.size(),
            0,
            0,
            jpeg2k_stream
        )
    );

    //----------------------------------------------------------
    // Get image info
    //----------------------------------------------------------

    nvjpeg2kImageInfo_t image_info;

    CHECK_NVJPEG2K(
        nvjpeg2kStreamGetImageInfo(
            jpeg2k_stream,
            &image_info
        )
    );

    nvjpeg2kImageComponentInfo_t comp_info;
    CHECK_NVJPEG2K(
        nvjpeg2kStreamGetImageComponentInfo(
            jpeg2k_stream,
            &comp_info,
            0
        )
    );

    std::cout << "Image info:" << std::endl;
    std::cout << "  Width       : " << image_info.image_width << std::endl;
    std::cout << "  Height      : " << image_info.image_height << std::endl;
    std::cout << "  Components  : " << image_info.num_components << std::endl;
    std::cout << "  Precision   : " << static_cast<int>(comp_info.precision) << " bits" << std::endl;

    //----------------------------------------------------------
    // Allocate GPU memory
    //----------------------------------------------------------

    int bytes_per_component = (comp_info.precision > 8) ? 2 : 1;

    size_t pixel_count =
        (size_t)image_info.image_width *
        (size_t)image_info.image_height *
        (size_t)image_info.num_components;
    size_t buffer_size = pixel_count * bytes_per_component;

    unsigned char* d_pixels = nullptr;

    CHECK_CUDA(
        cudaMalloc(&d_pixels, buffer_size)
    );

    //----------------------------------------------------------
    // Setup output image
    //----------------------------------------------------------

    std::vector<void*> pixel_data_ptrs(image_info.num_components);
    std::vector<size_t> pitch_ptrs(image_info.num_components);

    nvjpeg2kImage_t output_image;

    std::memset(&output_image, 0, sizeof(output_image));

    output_image.num_components = image_info.num_components;
    output_image.pixel_type = (comp_info.precision > 8) ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;
    output_image.pixel_data = pixel_data_ptrs.data();
    output_image.pitch_in_bytes = pitch_ptrs.data();
    
    size_t pixels_per_component = (size_t)image_info.image_width * (size_t)image_info.image_height;

    for (uint32_t c = 0; c < image_info.num_components; ++c) {
        pixel_data_ptrs[c] = d_pixels + c * (pixels_per_component * bytes_per_component);
        pitch_ptrs[c] = image_info.image_width * bytes_per_component; // DO NOT multiply by num_components
    }

    //----------------------------------------------------------
    // Decode image
    //----------------------------------------------------------

    std::cout << "Decoding on GPU..." << std::endl;

    CHECK_NVJPEG2K(
        nvjpeg2kDecode(
            handle,
            decode_state,
            jpeg2k_stream,
            &output_image,
            stream
        )
    );

    CHECK_CUDA(
        cudaStreamSynchronize(stream)
    );

    std::cout << "Decode completed successfully." << std::endl;

    //----------------------------------------------------------
    // Copy result back to CPU
    //----------------------------------------------------------

    std::vector<unsigned char> host_pixels(buffer_size);

    CHECK_CUDA(
        cudaMemcpy(
            host_pixels.data(),
            d_pixels,
            buffer_size,
            cudaMemcpyDeviceToHost
        )
    );

    //----------------------------------------------------------
    // Print first pixels
    //----------------------------------------------------------

    std::cout << "First 16 decoded bytes:" << std::endl;

    for (int i = 0; i < 16 && i < host_pixels.size(); ++i) {
        std::cout << static_cast<int>(host_pixels[i]) << " ";
    }

    std::cout << std::endl;

    //----------------------------------------------------------
    // Cleanup
    //----------------------------------------------------------

    CHECK_CUDA(cudaFree(d_pixels));

    CHECK_NVJPEG2K(
        nvjpeg2kStreamDestroy(jpeg2k_stream)
    );

    CHECK_NVJPEG2K(
        nvjpeg2kDecodeStateDestroy(decode_state)
    );

    CHECK_NVJPEG2K(
        nvjpeg2kDestroy(handle)
    );

    CHECK_CUDA(
        cudaStreamDestroy(stream)
    );

    std::cout << "Done." << std::endl;

    return EXIT_SUCCESS;
}