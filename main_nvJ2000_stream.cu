#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <time.h>
#include <thread>
#include <sys/stat.h>
#include <sys/types.h>
// nvJPEG2000 for hardware-accelerated JP2 reading
#include <nvjpeg2k.h>
#include <zlib.h>  // for PNG compression (-lz)

// ==========================================
// ERROR-CHECKING MACROS
// ==========================================
#define CHECK_NVJPEG2K(call)                                 \
    do {                                                     \
        nvjpeg2kStatus_t status = call;                      \
        if (status != NVJPEG2K_STATUS_SUCCESS) {             \
            std::cerr << "nvJPEG2000 Error: " << status      \
                      << " at " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

#define CHECK_CUDA(call)                                     \
    do {                                                     \
        const cudaError_t error = call;                      \
        if (error != cudaSuccess) {                          \
            fprintf(stderr, "CUDA Error: %s:%d, code:%d, "  \
                    "reason: %s\n", __FILE__, __LINE__,      \
                    error, cudaGetErrorString(error));        \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)

// --- CONSTANTS FOR EVI ---
#define G  2.5f
#define C1 6.0f
#define C2 7.5f
#define L  1.0f

// --- QUALITY FLAG BITMASK ---
#define QFLAG_BLACK      (1u << 0)
#define QFLAG_SATURATED  (1u << 1)
#define QFLAG_OUTLIER    (1u << 2)
#define QFLAG_FLAT       (1u << 3)
#define QFLAG_INVALID    (1u << 4)
#define QFLAG_STRIPE     (1u << 5)

// --- HEALTH CLASS LABELS ---
#define HEALTH_DEAD      0u
#define HEALTH_DISEASED  1u
#define HEALTH_HEALTHY   2u
#define HEALTH_UNCERTAIN 3u

// ==========================================
// SCENE STRUCTURE FOR MULTI-TEMPORAL ANALYSIS
// ==========================================
struct Scene {
    const char* red_path;
    const char* nir_path;
    const char* blue_path;
    const char* name;
};

static const Scene SCENES[] = {
  #include "scenes_files/scenes_gpu.txt"

};
static const int NUM_SCENES = sizeof(SCENES) / sizeof(SCENES[0]);

// --- PIPELINE STREAMS CONFIGURATION ---
// Change this value to control how many CUDA streams are used for overlapping.
//   1 = tutto sequenziale su un unico stream
//   2 = compute e colormap si sovrappongono, transfer usa lo stream di compute
//   3 = compute, colormap e transfer si sovrappongono (default ottimizzato)
static const int NUM_PIPELINE_STREAMS = 2;

// ==========================================
// FORWARD DECLARATIONS
// ==========================================
unsigned short* load_raster_image(const char* filename,
                                  int* width, int* height,
                                  nvjpeg2kHandle_t handle,
                                  nvjpeg2kDecodeState_t decode_state,
                                  cudaStream_t stream);

void report_band_quality(const unsigned short* band,
                         int num_pixels,
                         const char* band_name,
                         float black_ratio_threshold = 0.95f);

// ==========================================
// KERNEL 1: NDVI CALCULATION
// ==========================================
__global__ void ndvi_kernel(const unsigned short* d_red,
                            const unsigned short* d_nir,
                            float* d_result,
                            int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float red = (float)d_red[idx];
        float nir = (float)d_nir[idx];
        float denominator = nir + red;
        if (denominator == 0.0f) {
            d_result[idx] = 0.0f;
        } else {
            d_result[idx] = (nir - red) / denominator;
        }
    }
}

// ==========================================
// KERNEL 2: EVI CALCULATION
// ==========================================
__global__ void evi_kernel(const unsigned short* d_red,
                           const unsigned short* d_nir,
                           const unsigned short* d_blue,
                           float* d_result,
                           int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        float red  = (float)d_red[idx];
        float nir  = (float)d_nir[idx];
        float blue = (float)d_blue[idx];

        float numerator   = nir - red;
        float denominator = nir + (C1 * red) - (C2 * blue) + L;

        if (denominator == 0.0f) {
            d_result[idx] = 0.0f;
        } else {
            float value = G * (numerator / denominator);
            if (value >  1.0f) value =  1.0f;
            if (value < -1.0f) value = -1.0f;
            d_result[idx] = value;
        }
    }
}

// ==========================================
// KERNEL 3: LOCAL STATISTICS ON NDVI MAP
// ==========================================
__global__ void local_stats_kernel(const float* d_src,
                                   float* d_mean,
                                   float* d_var,
                                   float* d_grad,
                                   int width,
                                   int height,
                                   int radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;
    if (idx >= num_pixels) return;

    int x = idx % width;
    int y = idx / width;

    float sum    = 0.0f;
    float sum_sq = 0.0f;
    int   count  = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            float v = d_src[ny * width + nx];
            sum    += v;
            sum_sq += v * v;
            count++;
        }
    }

    float mean = (count > 0) ? (sum / (float)count) : 0.0f;
    float var  = (count > 0) ? ((sum_sq / (float)count) - (mean * mean)) : 0.0f;
    if (var < 0.0f) var = 0.0f;

    // Sobel-like 3x3 gradient magnitude
    int xm1 = max(0, x - 1);
    int xp1 = min(width - 1, x + 1);
    int ym1 = max(0, y - 1);
    int yp1 = min(height - 1, y + 1);

    float p00 = d_src[ym1 * width + xm1];
    float p01 = d_src[ym1 * width + x];
    float p02 = d_src[ym1 * width + xp1];
    float p10 = d_src[y   * width + xm1];
    float p12 = d_src[y   * width + xp1];
    float p20 = d_src[yp1 * width + xm1];
    float p21 = d_src[yp1 * width + x];
    float p22 = d_src[yp1 * width + xp1];

    float gx = (-p00 + p02) + (-2.0f * p10 + 2.0f * p12) + (-p20 + p22);
    float gy = ( p00 + 2.0f * p01 + p02) - (p20 + 2.0f * p21 + p22);
    float grad = sqrtf(gx * gx + gy * gy);

    d_mean[idx] = mean;
    d_var[idx]  = var;
    d_grad[idx] = grad;
}

// ==========================================
// KERNEL 4: PIXEL QUALITY FLAGS + SCORE
// ==========================================
__global__ void quality_flags_kernel(const unsigned short* d_red,
                                     const unsigned short* d_nir,
                                     const unsigned short* d_blue,
                                     const float* d_ndvi,
                                     const float* d_evi,
                                     const float* d_ndvi_mean,
                                     const float* d_ndvi_var,
                                     const float* d_ndvi_grad,
                                     unsigned int* d_flags,
                                     float* d_quality_score,
                                     int width,
                                     int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;
    if (idx >= num_pixels) return;

    int x = idx % width;
    int y = idx / width;

    unsigned short r = d_red[idx];
    unsigned short n = d_nir[idx];
    unsigned short b = d_blue[idx];
    float ndvi = d_ndvi[idx];
    float evi  = d_evi[idx];
    float mean = d_ndvi_mean[idx];
    float var  = d_ndvi_var[idx];
    float grad = d_ndvi_grad[idx];

    unsigned int flags = 0u;

    if (r <= 1u || n <= 1u || b <= 1u)             flags |= QFLAG_BLACK;
    if (r >= 65530u || n >= 65530u || b >= 65530u)  flags |= QFLAG_SATURATED;
    if (!isfinite(ndvi) || !isfinite(evi))          flags |= QFLAG_INVALID;

    float z = fabsf(ndvi - mean) / sqrtf(var + 1e-8f);
    if (z > 3.0f) flags |= QFLAG_OUTLIER;

    if (var < 1e-5f && grad < 1e-2f) flags |= QFLAG_FLAT;

    // Stripe detector
    int xm1 = max(0, x - 1);
    int xp1 = min(width - 1, x + 1);
    int ym1 = max(0, y - 1);
    int yp1 = min(height - 1, y + 1);
    unsigned short row_min = min(d_red[y * width + xm1], min(d_red[idx], d_red[y * width + xp1]));
    unsigned short row_max = max(d_red[y * width + xm1], max(d_red[idx], d_red[y * width + xp1]));
    unsigned short col_min = min(d_red[ym1 * width + x], min(d_red[idx], d_red[yp1 * width + x]));
    unsigned short col_max = max(d_red[ym1 * width + x], max(d_red[idx], d_red[yp1 * width + x]));
    if ((row_max - row_min) <= 1u && (col_max - col_min) <= 1u) flags |= QFLAG_STRIPE;

    float score = 1.0f;
    if (flags & QFLAG_BLACK)     score -= 0.40f;
    if (flags & QFLAG_SATURATED) score -= 0.30f;
    if (flags & QFLAG_OUTLIER)   score -= 0.20f;
    if (flags & QFLAG_FLAT)      score -= 0.10f;
    if (flags & QFLAG_INVALID)   score -= 0.50f;
    if (flags & QFLAG_STRIPE)    score -= 0.20f;
    if (score < 0.0f) score = 0.0f;

    d_flags[idx]         = flags;
    d_quality_score[idx] = score;
}

// ==========================================
// KERNEL 5: HEALTH CLASSIFICATION MASK
// ==========================================
__global__ void classification_kernel(const float* d_ndvi,
                                      const float* d_evi,
                                      const unsigned int* d_quality_flags,
                                      unsigned char* d_health_class,
                                      int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    unsigned int flags = d_quality_flags[idx];
    float ndvi = d_ndvi[idx];
    float evi  = d_evi[idx];

    if (flags & (QFLAG_INVALID | QFLAG_SATURATED)) {
        d_health_class[idx] = HEALTH_UNCERTAIN;
        return;
    }

    if (ndvi < 0.20f || evi < 0.10f) {
        d_health_class[idx] = HEALTH_DEAD;
    } else if (ndvi < 0.45f || evi < 0.25f ||
               (flags & (QFLAG_OUTLIER | QFLAG_BLACK | QFLAG_STRIPE))) {
        d_health_class[idx] = HEALTH_DISEASED;
    } else {
        d_health_class[idx] = HEALTH_HEALTHY;
    }
}

// ==========================================
// KERNEL 6: COLORMAP (RdYlGn)
// ==========================================
__global__ void colormap_rdylgn_kernel(const float* d_index,
                                       unsigned char* d_rgb,
                                       int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float val = d_index[idx];
    if (val < -1.0f) val = -1.0f;
    if (val >  1.0f) val =  1.0f;
    float t = (val + 1.0f) * 0.5f;

    unsigned char r = 0, g = 0, b = 0;
    if (t <= 0.5f) {
        r = 255;
        g = (unsigned char)(255.0f * (t / 0.5f));
    } else {
        r = (unsigned char)(255.0f * (1.0f - ((t - 0.5f) / 0.5f)));
        g = 255;
    }
    d_rgb[idx * 3 + 0] = r;
    d_rgb[idx * 3 + 1] = g;
    d_rgb[idx * 3 + 2] = b;
}

// ==========================================
// KERNEL 7: GRAYSCALE MASK FOR HEALTH
// ==========================================
__global__ void mask_grayscale_kernel(const unsigned char* d_health_class,
                                      unsigned char* d_rgb,
                                      int num_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    unsigned char val = (d_health_class[idx] == HEALTH_HEALTHY) ? 255 : 0;
    d_rgb[idx * 3 + 0] = val;
    d_rgb[idx * 3 + 1] = val;
    d_rgb[idx * 3 + 2] = val;
}

// ==========================================
// HELPER: save PNG (minimal encoder using zlib)
// ==========================================
static void png_write_chunk(FILE* fp, const char* type,
                            const unsigned char* data, uint32_t length) {
    // Write: length (4B big-endian) | type (4B) | data | CRC32 (4B big-endian)
    unsigned char len_be[4] = {
        (unsigned char)((length >> 24) & 0xFF),
        (unsigned char)((length >> 16) & 0xFF),
        (unsigned char)((length >>  8) & 0xFF),
        (unsigned char)((length      ) & 0xFF)
    };
    fwrite(len_be, 1, 4, fp);
    fwrite(type, 1, 4, fp);

    uint32_t crc = crc32(0, (const Bytef*)type, 4);
    if (data && length > 0) {
        fwrite(data, 1, length, fp);
        crc = crc32(crc, data, length);
    }

    unsigned char crc_be[4] = {
        (unsigned char)((crc >> 24) & 0xFF),
        (unsigned char)((crc >> 16) & 0xFF),
        (unsigned char)((crc >>  8) & 0xFF),
        (unsigned char)((crc      ) & 0xFF)
    };
    fwrite(crc_be, 1, 4, fp);
}

void save_png(const char* filename, const unsigned char* rgb, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Error: cannot create %s\n", filename); return; }

    // PNG signature
    const unsigned char signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(signature, 1, 8, fp);

    // IHDR chunk: width, height, bit_depth=8, color_type=2 (RGB)
    unsigned char ihdr[13];
    ihdr[0]  = (width  >> 24) & 0xFF;
    ihdr[1]  = (width  >> 16) & 0xFF;
    ihdr[2]  = (width  >>  8) & 0xFF;
    ihdr[3]  = (width       ) & 0xFF;
    ihdr[4]  = (height >> 24) & 0xFF;
    ihdr[5]  = (height >> 16) & 0xFF;
    ihdr[6]  = (height >>  8) & 0xFF;
    ihdr[7]  = (height      ) & 0xFF;
    ihdr[8]  = 8;   // bit depth
    ihdr[9]  = 2;   // color type: RGB
    ihdr[10] = 0;   // compression method
    ihdr[11] = 0;   // filter method
    ihdr[12] = 0;   // interlace method
    png_write_chunk(fp, "IHDR", ihdr, 13);

    // Prepare raw scanlines: each row gets a filter byte (0 = None) + RGB data
    size_t row_bytes = (size_t)width * 3;
    size_t raw_size  = (size_t)height * (1 + row_bytes);
    unsigned char* raw = (unsigned char*)malloc(raw_size);
    if (!raw) {
        fprintf(stderr, "Error: malloc failed for PNG raw buffer\n");
        fclose(fp);
        return;
    }

    for (int y = 0; y < height; y++) {
        raw[y * (1 + row_bytes)] = 0;  // filter byte: None
        memcpy(&raw[y * (1 + row_bytes) + 1], &rgb[y * row_bytes], row_bytes);
    }

    // Compress with zlib
    uLongf compressed_size = compressBound(raw_size);
    unsigned char* compressed = (unsigned char*)malloc(compressed_size);
    if (!compressed) {
        fprintf(stderr, "Error: malloc failed for PNG compress buffer\n");
        free(raw);
        fclose(fp);
        return;
    }

    int z_ret = compress2(compressed, &compressed_size, raw, raw_size, Z_BEST_SPEED);
    free(raw);
    if (z_ret != Z_OK) {
        fprintf(stderr, "Error: zlib compress2 failed (%d)\n", z_ret);
        free(compressed);
        fclose(fp);
        return;
    }

    // IDAT chunk
    png_write_chunk(fp, "IDAT", compressed, (uint32_t)compressed_size);
    free(compressed);

    // IEND chunk
    png_write_chunk(fp, "IEND", nullptr, 0);

    fclose(fp);
}

// ==========================================
// PROCESS A SINGLE SCENE
// ==========================================
void process_scene(const Scene& scene, int window_size,
                   nvjpeg2kHandle_t nv_handle,
                   nvjpeg2kDecodeState_t nv_decode_state,
                   cudaStream_t nv_stream,
                   std::vector<std::thread>& background_threads)
{
    std::cerr << "\n===== Processing Scene: " << scene.name << " =====" << std::endl;

    // Create CUDA streams for pipeline overlap
    std::vector<cudaStream_t> streams(NUM_PIPELINE_STREAMS);
    for (int i = 0; i < NUM_PIPELINE_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    std::cerr << "Using " << NUM_PIPELINE_STREAMS << " pipeline stream(s)" << std::endl;

    // Map logical pipeline roles to the available streams (modulo wrapping)
    cudaStream_t stream_compute  = streams[0];
    cudaStream_t stream_colormap = streams[1 % NUM_PIPELINE_STREAMS];
    cudaStream_t stream_transfer = streams[2 % NUM_PIPELINE_STREAMS];

    int width = 0, height = 0;
    int width_nir = 0, height_nir = 0;
    int width_blue = 0, height_blue = 0;

    // ---- PHASE 1: Decode JP2 directly to GPU ----
    auto decode_start = std::chrono::high_resolution_clock::now();

    unsigned short* d_red  = load_raster_image(scene.red_path,  &width,      &height,      nv_handle, nv_decode_state, nv_stream);
    unsigned short* d_nir  = load_raster_image(scene.nir_path,  &width_nir,  &height_nir,  nv_handle, nv_decode_state, nv_stream);
    unsigned short* d_blue = load_raster_image(scene.blue_path, &width_blue, &height_blue, nv_handle, nv_decode_state, nv_stream);

    auto decode_end = std::chrono::high_resolution_clock::now();
    double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

    if (!d_red || !d_nir || !d_blue) {
        std::cerr << "Error: Failed to load bands for " << scene.name << std::endl;
        if (d_red)  cudaFree(d_red);
        if (d_nir)  cudaFree(d_nir);
        if (d_blue) cudaFree(d_blue);
        return;
    }

    if (width != width_nir || height != height_nir ||
        width != width_blue || height != height_blue) {
        std::cerr << "Error: Image dimensions do not match for " << scene.name << std::endl;
        cudaFree(d_red); cudaFree(d_nir); cudaFree(d_blue);
        return;
    }

    int num_pixels = width * height;
    size_t band_bytes  = (size_t)num_pixels * sizeof(unsigned short);
    size_t float_bytes = (size_t)num_pixels * sizeof(float);
    size_t rgb_bytes   = (size_t)num_pixels * 3;

    std::cerr << "Images loaded: " << width << " x " << height
              << " (" << num_pixels << " px) in " << decode_seconds << "s" << std::endl;

    // ---- PHASE 2: Quality check ----
    {
        unsigned short* h_band_tmp = nullptr;
        CHECK_CUDA(cudaMallocHost((void**)&h_band_tmp, band_bytes));
        if (h_band_tmp) {
            CHECK_CUDA(cudaMemcpy(h_band_tmp, d_red, band_bytes, cudaMemcpyDeviceToHost));
            report_band_quality(h_band_tmp, num_pixels, "RED");

            CHECK_CUDA(cudaMemcpy(h_band_tmp, d_nir, band_bytes, cudaMemcpyDeviceToHost));
            report_band_quality(h_band_tmp, num_pixels, "NIR");

            CHECK_CUDA(cudaMemcpy(h_band_tmp, d_blue, band_bytes, cudaMemcpyDeviceToHost));
            report_band_quality(h_band_tmp, num_pixels, "BLUE");

            cudaFreeHost(h_band_tmp);
        }
    }

    // ---- PHASE 3: Allocate GPU result buffers ----
    float *d_result_ndvi, *d_result_evi;
    float *d_ndvi_mean, *d_ndvi_var, *d_ndvi_grad;
    float *d_quality_score;
    unsigned int  *d_quality_flags;
    unsigned char *d_health_class;
    unsigned char *d_rgb_ndvi, *d_rgb_evi, *d_rgb_mask;  // separate buffers for parallel stream rendering

    CHECK_CUDA(cudaMalloc(&d_result_ndvi,   float_bytes));
    CHECK_CUDA(cudaMalloc(&d_result_evi,    float_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_mean,     float_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_var,      float_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_grad,     float_bytes));
    CHECK_CUDA(cudaMalloc(&d_quality_score, float_bytes));
    CHECK_CUDA(cudaMalloc(&d_quality_flags, (size_t)num_pixels * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_health_class,  (size_t)num_pixels * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_rgb_ndvi,      rgb_bytes));
    CHECK_CUDA(cudaMalloc(&d_rgb_evi,       rgb_bytes));
    CHECK_CUDA(cudaMalloc(&d_rgb_mask,      rgb_bytes));

    // ---- PHASE 4: Launch compute kernels (with stream overlap) ----
    int blockSize    = 256;
    int gridSize     = (num_pixels + blockSize - 1) / blockSize;
    int local_radius = window_size / 2;

    cudaEvent_t ev_start, ev_stop, ev_evi_done, ev_compute_done;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_evi_done));
    CHECK_CUDA(cudaEventCreate(&ev_compute_done));
    cudaEventRecord(ev_start, stream_compute);

    // NDVI and EVI in parallel on separate streams
    ndvi_kernel<<<gridSize, blockSize, 0, stream_compute>>>(d_red, d_nir, d_result_ndvi, num_pixels);

    evi_kernel<<<gridSize, blockSize, 0, stream_colormap>>>(d_red, d_nir, d_blue, d_result_evi, num_pixels);
    cudaEventRecord(ev_evi_done, stream_colormap);

    // local_stats depends on NDVI (implicit ordering on same stream_compute)
    local_stats_kernel<<<gridSize, blockSize, 0, stream_compute>>>(
        d_result_ndvi, d_ndvi_mean, d_ndvi_var, d_ndvi_grad,
        width, height, local_radius);

    // quality_flags needs EVI result from stream_colormap
    cudaStreamWaitEvent(stream_compute, ev_evi_done, 0);
    quality_flags_kernel<<<gridSize, blockSize, 0, stream_compute>>>(
        d_red, d_nir, d_blue,
        d_result_ndvi, d_result_evi,
        d_ndvi_mean, d_ndvi_var, d_ndvi_grad,
        d_quality_flags, d_quality_score,
        width, height);

    classification_kernel<<<gridSize, blockSize, 0, stream_compute>>>(
        d_result_ndvi, d_result_evi, d_quality_flags,
        d_health_class, num_pixels);

    cudaEventRecord(ev_compute_done, stream_compute);
    cudaEventRecord(ev_stop, stream_compute);

    // ---- PHASE 5: Generate output images (with stream overlap) ----
    mkdir("gpu_results", 0755);
    std::string prefix = std::string("gpu_results/") + scene.name + "_";

    // Colormap/mask kernels wait for compute to finish
    cudaStreamWaitEvent(stream_colormap, ev_compute_done, 0);
    cudaStreamWaitEvent(stream_transfer, ev_compute_done, 0);

    // 1. NDVI Colormap on stream_colormap
    colormap_rdylgn_kernel<<<gridSize, blockSize, 0, stream_colormap>>>(d_result_ndvi, d_rgb_ndvi, num_pixels);
    unsigned char* h_rgb_ndvi = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_rgb_ndvi, rgb_bytes));
    CHECK_CUDA(cudaMemcpyAsync(h_rgb_ndvi, d_rgb_ndvi, rgb_bytes, cudaMemcpyDeviceToHost, stream_colormap));

    // 2. EVI Colormap on stream_transfer (parallel to NDVI D2H)
    colormap_rdylgn_kernel<<<gridSize, blockSize, 0, stream_transfer>>>(d_result_evi, d_rgb_evi, num_pixels);
    unsigned char* h_rgb_evi = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_rgb_evi, rgb_bytes));
    CHECK_CUDA(cudaMemcpyAsync(h_rgb_evi, d_rgb_evi, rgb_bytes, cudaMemcpyDeviceToHost, stream_transfer));

    // 3. Health Mask on stream_compute (free after classification)
    mask_grayscale_kernel<<<gridSize, blockSize, 0, stream_compute>>>(d_health_class, d_rgb_mask, num_pixels);
    unsigned char* h_rgb_mask = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_rgb_mask, rgb_bytes));
    CHECK_CUDA(cudaMemcpyAsync(h_rgb_mask, d_rgb_mask, rgb_bytes, cudaMemcpyDeviceToHost, stream_compute));

    // Synchronize each stream before launching its PNG saving thread
    CHECK_CUDA(cudaStreamSynchronize(stream_colormap));
    {
        std::string ndvi_path = prefix + "ndvi_colormap.png";
        background_threads.push_back(std::thread([h_rgb_ndvi, width, height, ndvi_path]() {
            save_png(ndvi_path.c_str(), h_rgb_ndvi, width, height);
            cudaFreeHost(h_rgb_ndvi);
        }));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream_transfer));
    {
        std::string evi_path = prefix + "evi_colormap.png";
        background_threads.push_back(std::thread([h_rgb_evi, width, height, evi_path]() {
            save_png(evi_path.c_str(), h_rgb_evi, width, height);
            cudaFreeHost(h_rgb_evi);
        }));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    {
        std::string mask_path = prefix + "health_mask.png";
        background_threads.push_back(std::thread([h_rgb_mask, width, height, mask_path]() {
            save_png(mask_path.c_str(), h_rgb_mask, width, height);
            cudaFreeHost(h_rgb_mask);
        }));
    }

    // Compute kernel timing (events already completed after stream syncs)
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    std::cerr << "Compute kernels: " << gpu_ms << " ms" << std::endl;

    // ---- PHASE 6: Copy results D2H and compute statistics ----
    float *h_result_ndvi    = nullptr;
    float *h_result_evi     = nullptr;
    float *h_ndvi_mean      = nullptr;
    float *h_ndvi_var       = nullptr;
    float *h_ndvi_grad      = nullptr;
    float *h_quality_score  = nullptr;
    unsigned int  *h_quality_flags = nullptr;
    unsigned char *h_health_class  = nullptr;

    CHECK_CUDA(cudaMallocHost((void**)&h_result_ndvi,   float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_result_evi,    float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_ndvi_mean,     float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_ndvi_var,      float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_ndvi_grad,     float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_quality_score, float_bytes));
    CHECK_CUDA(cudaMallocHost((void**)&h_quality_flags, (size_t)num_pixels * sizeof(unsigned int)));
    CHECK_CUDA(cudaMallocHost((void**)&h_health_class,  (size_t)num_pixels * sizeof(unsigned char)));

    // Async D2H transfers on stream_transfer
    CHECK_CUDA(cudaMemcpyAsync(h_result_ndvi,   d_result_ndvi,   float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_result_evi,    d_result_evi,    float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_ndvi_mean,     d_ndvi_mean,     float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_ndvi_var,      d_ndvi_var,      float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_ndvi_grad,     d_ndvi_grad,     float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_quality_score, d_quality_score, float_bytes, cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_quality_flags, d_quality_flags, (size_t)num_pixels * sizeof(unsigned int),  cudaMemcpyDeviceToHost, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(h_health_class,  d_health_class,  (size_t)num_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream_transfer));

    // Wait for all transfers before CPU statistics
    CHECK_CUDA(cudaStreamSynchronize(stream_transfer));

    // ---- Statistics ----
    double sum_ndvi = 0.0, sum_evi = 0.0;
    double sum_ndvi_mean_acc = 0.0, sum_ndvi_var_acc = 0.0;
    double sum_ndvi_grad_acc = 0.0, sum_quality_score_acc = 0.0;
    int count_black = 0, count_saturated = 0, count_outlier = 0;
    int count_flat = 0, count_invalid = 0, count_stripe = 0;
    int count_dead = 0, count_diseased = 0, count_healthy = 0, count_uncertain = 0;

    for (int i = 0; i < num_pixels; i++) {
        sum_ndvi            += (double)h_result_ndvi[i];
        sum_evi             += (double)h_result_evi[i];
        sum_ndvi_mean_acc   += (double)h_ndvi_mean[i];
        sum_ndvi_var_acc    += (double)h_ndvi_var[i];
        sum_ndvi_grad_acc   += (double)h_ndvi_grad[i];
        sum_quality_score_acc += (double)h_quality_score[i];

        unsigned int f = h_quality_flags[i];
        if (f & QFLAG_BLACK)     count_black++;
        if (f & QFLAG_SATURATED) count_saturated++;
        if (f & QFLAG_OUTLIER)   count_outlier++;
        if (f & QFLAG_FLAT)      count_flat++;
        if (f & QFLAG_INVALID)   count_invalid++;
        if (f & QFLAG_STRIPE)    count_stripe++;

        unsigned char hc = h_health_class[i];
        if      (hc == HEALTH_DEAD)     count_dead++;
        else if (hc == HEALTH_DISEASED) count_diseased++;
        else if (hc == HEALTH_HEALTHY)  count_healthy++;
        else                            count_uncertain++;
    }

    std::cerr << "--- Results for " << scene.name << " ---" << std::endl;
    std::cerr << "Average NDVI: " << sum_ndvi / num_pixels << std::endl;
    std::cerr << "Average EVI:  " << sum_evi / num_pixels << std::endl;
    std::cerr << "Average NDVI local mean (" << window_size << "x" << window_size << "): " << sum_ndvi_mean_acc / num_pixels << std::endl;
    std::cerr << "Average NDVI local var  (" << window_size << "x" << window_size << "): " << sum_ndvi_var_acc / num_pixels << std::endl;
    std::cerr << "Average NDVI Sobel-like grad: " << sum_ndvi_grad_acc / num_pixels << std::endl;
    std::cerr << "Average Quality Score: " << sum_quality_score_acc / num_pixels << std::endl;

    std::cerr << "Quality Flags:" << std::endl;
    std::cerr << "  Black:     " << count_black << " (" << 100.0 * count_black / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Saturated: " << count_saturated << " (" << 100.0 * count_saturated / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Outlier:   " << count_outlier << " (" << 100.0 * count_outlier / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Flat:      " << count_flat << " (" << 100.0 * count_flat / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Invalid:   " << count_invalid << " (" << 100.0 * count_invalid / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Stripe:    " << count_stripe << " (" << 100.0 * count_stripe / (double)num_pixels << "%)" << std::endl;

    std::cerr << "Health Classes:" << std::endl;
    std::cerr << "  Dead:      " << count_dead << " (" << 100.0 * count_dead / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Diseased:  " << count_diseased << " (" << 100.0 * count_diseased / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Healthy:   " << count_healthy << " (" << 100.0 * count_healthy / (double)num_pixels << "%)" << std::endl;
    std::cerr << "  Uncertain: " << count_uncertain << " (" << 100.0 * count_uncertain / (double)num_pixels << "%)" << std::endl;

    std::cerr << "Decode time: " << decode_seconds << "s" << std::endl;

    // ---- Cleanup ----
    cudaFree(d_red);  cudaFree(d_nir);  cudaFree(d_blue);
    cudaFree(d_result_ndvi);  cudaFree(d_result_evi);
    cudaFree(d_ndvi_mean);    cudaFree(d_ndvi_var);    cudaFree(d_ndvi_grad);
    cudaFree(d_quality_score); cudaFree(d_quality_flags);
    cudaFree(d_health_class);
    cudaFree(d_rgb_ndvi); cudaFree(d_rgb_evi); cudaFree(d_rgb_mask);

    cudaFreeHost(h_result_ndvi);  cudaFreeHost(h_result_evi);
    cudaFreeHost(h_ndvi_mean);    cudaFreeHost(h_ndvi_var);    cudaFreeHost(h_ndvi_grad);
    cudaFreeHost(h_quality_score); cudaFreeHost(h_quality_flags);
    cudaFreeHost(h_health_class);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(ev_evi_done);
    cudaEventDestroy(ev_compute_done);

    for (int i = 0; i < NUM_PIPELINE_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

// ==========================================
// JP2 LOADER — returns DEVICE pointer
// ==========================================
unsigned short* load_raster_image(const char* filename,
                                  int* width, int* height,
                                  nvjpeg2kHandle_t handle,
                                  nvjpeg2kDecodeState_t decode_state,
                                  cudaStream_t stream)
{
    std::string s(filename);
    std::string ext;
    size_t pos = s.find_last_of('.');
    if (pos != std::string::npos) ext = s.substr(pos);

    if (ext != ".jp2" && ext != ".JP2") {
        std::cerr << "Error: Unsupported file type: " << filename << std::endl;
        return nullptr;
    }

    // Read compressed data from disk
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size == 0) {
        std::cerr << "Error: File is empty " << filename << std::endl;
        return nullptr;
    }

    // Pinned host memory for the bitstream
    unsigned char* h_compressed_data = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_compressed_data, size));

    if (!file.read(reinterpret_cast<char*>(h_compressed_data), size)) {
        std::cerr << "Error: Failed to read " << filename << std::endl;
        cudaFreeHost(h_compressed_data);
        return nullptr;
    }
    file.close();

    // Parse bitstream
    nvjpeg2kStream_t jpeg2k_stream;
    CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&jpeg2k_stream));

    CHECK_NVJPEG2K(nvjpeg2kStreamParse(
        handle, h_compressed_data, size, 0, 0, jpeg2k_stream));

    nvjpeg2kImageInfo_t image_info;
    CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(jpeg2k_stream, &image_info));

    nvjpeg2kImageComponentInfo_t comp_info;
    CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp_info, 0));

    *width  = image_info.image_width;
    *height = image_info.image_height;

    int bytes_per_component = (comp_info.precision > 8) ? 2 : 1;
    size_t pixel_count = (size_t)(*width) * (*height) * image_info.num_components;
    size_t buffer_size = pixel_count * bytes_per_component;

    if (image_info.num_components > 16) {
        std::cerr << "Error: Too many components (" << image_info.num_components << ")" << std::endl;
        cudaFreeHost(h_compressed_data);
        nvjpeg2kStreamDestroy(jpeg2k_stream);
        return nullptr;
    }

    // Allocate pitched GPU buffers for decoder output
    void*  pixel_data_ptrs[16]      = { nullptr };
    size_t pitch_ptrs[16]           = { 0 };
    void*  d_pitched_components[16] = { nullptr };

    for (uint32_t c = 0; c < image_info.num_components; ++c) {
        nvjpeg2kImageComponentInfo_t comp_info_c;
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp_info_c, c));
        int comp_bytes = (comp_info_c.precision > 8) ? 2 : 1;

        size_t comp_pitch = 0;
        void*  d_comp_pixels = nullptr;
        CHECK_CUDA(cudaMallocPitch(&d_comp_pixels, &comp_pitch,
                                   comp_info_c.component_width * comp_bytes,
                                   comp_info_c.component_height));

        d_pitched_components[c] = d_comp_pixels;
        pixel_data_ptrs[c]      = d_comp_pixels;
        pitch_ptrs[c]           = comp_pitch;
    }

    nvjpeg2kImage_t output_image;
    std::memset(&output_image, 0, sizeof(output_image));
    output_image.pixel_data     = pixel_data_ptrs;
    output_image.pitch_in_bytes = pitch_ptrs;
    output_image.num_components = image_info.num_components;
    output_image.pixel_type     = (comp_info.precision > 8) ? NVJPEG2K_UINT16 : NVJPEG2K_UINT8;

    // Create local decode state and params
    nvjpeg2kDecodeState_t local_decode_state;
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle, &local_decode_state));

    nvjpeg2kDecodeParams_t decode_params;
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsCreate(&decode_params));
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetOutputFormat(decode_params, NVJPEG2K_FORMAT_PLANAR));

    // Decode
    nvjpeg2kStatus_t decode_status = nvjpeg2kDecodeImage(
        handle, local_decode_state, jpeg2k_stream,
        decode_params, &output_image, 0);

    if (decode_status != NVJPEG2K_STATUS_SUCCESS) {
        std::cerr << "Error: nvjpeg2kDecodeImage failed (status=" << decode_status
                  << ") for " << filename << std::endl;
        nvjpeg2kDecodeParamsDestroy(decode_params);
        nvjpeg2kDecodeStateDestroy(local_decode_state);
        for (uint32_t c = 0; c < image_info.num_components; ++c) {
            if (d_pitched_components[c]) cudaFree(d_pitched_components[c]);
        }
        nvjpeg2kStreamDestroy(jpeg2k_stream);
        cudaFreeHost(h_compressed_data);
        return nullptr;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Cleanup decode resources
    CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(decode_params));
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(local_decode_state));

    // Copy from pitched to flat contiguous buffer
    unsigned short* d_flat_pixels = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_flat_pixels, buffer_size));

    size_t flat_offset = 0;
    for (uint32_t c = 0; c < image_info.num_components; ++c) {
        nvjpeg2kImageComponentInfo_t comp_info_c;
        CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(jpeg2k_stream, &comp_info_c, c));
        int comp_bytes = (comp_info_c.precision > 8) ? 2 : 1;

        CHECK_CUDA(cudaMemcpy2D(
            (char*)d_flat_pixels + flat_offset,
            comp_info_c.component_width * comp_bytes,
            d_pitched_components[c],
            pitch_ptrs[c],
            comp_info_c.component_width * comp_bytes,
            comp_info_c.component_height,
            cudaMemcpyDeviceToDevice));

        flat_offset += (size_t)comp_info_c.component_width
                     * comp_info_c.component_height * comp_bytes;

        CHECK_CUDA(cudaFree(d_pitched_components[c]));
    }

    CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(jpeg2k_stream));
    CHECK_CUDA(cudaFreeHost(h_compressed_data));

    return d_flat_pixels;
}

// ==========================================
// BAND QUALITY REPORT (CPU)
// ==========================================
void report_band_quality(const unsigned short* band, int num_pixels,
                         const char* band_name, float black_ratio_threshold)
{
    if (!band || num_pixels <= 0) return;

    unsigned short min_v = band[0];
    unsigned short max_v = band[0];
    double sum    = 0.0;
    double sum_sq = 0.0;
    int black_pixels = 0;

    for (int i = 0; i < num_pixels; i++) {
        unsigned short v = band[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v <= 1) black_pixels++;
        double dv = static_cast<double>(v);
        sum    += dv;
        sum_sq += dv * dv;
    }

    double mean     = sum / static_cast<double>(num_pixels);
    double variance = (sum_sq / static_cast<double>(num_pixels)) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    double stddev = std::sqrt(variance);

    float black_ratio = static_cast<float>(black_pixels) / static_cast<float>(num_pixels);

    std::cerr << band_name << " | min=" << min_v
              << " max=" << max_v << " std=" << stddev
              << " black=" << black_ratio * 100.0f << "%" << std::endl;

    if (black_ratio >= black_ratio_threshold) {
        std::cerr << "WARNING: " << band_name << " mostly black ("
                  << black_ratio * 100.0f << "%)" << std::endl;
    }
    if (max_v == min_v || stddev < 1e-9) {
        std::cerr << "WARNING: " << band_name << " appears constant." << std::endl;
    }
}

// ==========================================
// HOST MAIN FUNCTION
// ==========================================
int main(int argc, char** argv) {
    setvbuf(stderr, NULL, _IONBF, 0);

    // Initialize CUDA context
    CHECK_CUDA(cudaFree(0));

    int window_size = 3;
    if (argc == 2) {
        window_size = std::atoi(argv[1]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [window_size(3|5)]" << std::endl;
        return -1;
    }
    if (window_size != 3 && window_size != 5) {
        std::cerr << "Error: window_size must be 3 or 5." << std::endl;
        return -1;
    }

    // Initialize nvJPEG2000
    nvjpeg2kHandle_t      nv_handle;
    nvjpeg2kDecodeState_t nv_decode_state = nullptr;
    cudaStream_t          nv_stream;

    // Use explicit backend — nvjpeg2kCreateSimple may probe for HW JPEG2000
    // decoder that some GPUs (e.g. A40) do not have.
    nvjpeg2kStatus_t create_status;
    create_status = nvjpeg2kCreate(NVJPEG2K_BACKEND_DEFAULT, nullptr, nullptr, &nv_handle);
    if (create_status != NVJPEG2K_STATUS_SUCCESS) {
        std::cerr << "FATAL: nvjpeg2kCreate failed (status=" << create_status << ")" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaStreamCreate(&nv_stream));

    // Create output directory
    mkdir("gpu_results", 0755);

    auto total_start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> background_threads;

    std::cerr << "Processing " << NUM_SCENES << " scenes (window=" << window_size << ")..." << std::endl;
    for (int i = 0; i < NUM_SCENES; i++) {
        try {
            process_scene(SCENES[i], window_size, nv_handle, nv_decode_state, nv_stream, background_threads);
        } catch (const std::exception& e) {
            std::cerr << "Error in scene " << SCENES[i].name << ": " << e.what() << std::endl;
        }
    }

    std::cerr << "Waiting for background PNG saving threads to finish..." << std::endl;
    for (auto& t : background_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(total_end - total_start).count();
    std::cerr << "\nTotal Execution Time: " << total_seconds << " seconds" << std::endl;

    CHECK_CUDA(cudaStreamDestroy(nv_stream));
    CHECK_NVJPEG2K(nvjpeg2kDestroy(nv_handle));

    return 0;
}
