#include <iostream>
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
// GDAL for JP2/TIFF reading
#include "gdal_priv.h"
#include "cpl_conv.h"

// --- CONSTANTS FOR EVI ---
// storing them in constant memory or macros is faster
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

// --- ERROR HANDLING MACRO ---
// Crucial for debugging GPU crashes
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ==========================================
// SCENE STRUCTURE FOR MULTI-TEMPORAL ANALYSIS
// ==========================================
struct Scene {
    const char* red_path;
    const char* nir_path;
    const char* blue_path;
    const char* name;
};

// Define scenes with fixed paths for multi-temporal analysis
static const Scene SCENES[] = {

    {
        "images/2026-04-05/20260405_B04_10m.jp2",
        "images/2026-04-05/20260405_B08_10m.jp2",
        "images/2026-04-05/20260405_B02_10m.jp2",
        "2026-04-05"
    },
    {
        "images/2026-04-15/20260415_B04_10m.jp2",
        "images/2026-04-15/20260415_B08_10m.jp2",
        "images/2026-04-15/20260415_B02_10m.jp2",
        "2026-04-15"
    },
    {
        "images/2026-04-25/20260425_B04_10m.jp2",
        "images/2026-04-25/20260425_B08_10m.jp2",
        "images/2026-04-25/20260425_B02_10m.jp2",
        "2026-04-25"
    },
     {
        "images/2026-05-05/20260505_B04_10m.jp2",
        "images/2026-05-05/20260505_B08_10m.jp2",
        "images/2026-05-05/20260505_B02_10m.jp2",
        "2026-05-05"
    },
   
};
static const int NUM_SCENES = sizeof(SCENES) / sizeof(SCENES[0]);

// Forward declarations (functions defined later)
static double cpu_seconds_now();
unsigned short* load_raster_image(const char* filename, int* width, int* height);
void report_band_quality(const unsigned short* band, int num_pixels, const char* band_name, float black_ratio_threshold = 0.95f);

// Single implementation of cpu_seconds_now
static double cpu_seconds_now() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// Crucial for debugging GPU crashes
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// ==========================================
// KERNEL 1: NDVI CALCULATION
// ==========================================
__global__ void ndvi_kernel(const unsigned short* d_red, 
                            const unsigned short* d_nir, 
                            float* d_result, 
                            int num_pixels) 
{
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (ensure we don't go outside the image)
    if (idx < num_pixels) {
        // Convert integer raw data to float for math
        float red = (float)d_red[idx];
        float nir = (float)d_nir[idx];

        // The Formula: (NIR - RED) / (NIR + RED)
        float denominator = nir + red;
        
        // Avoid division by zero!
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

        // The Formula: G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
        float numerator = nir - red;
        float denominator = nir + (C1 * red) - (C2 * blue) + L;

        if (denominator == 0.0f) {
            d_result[idx] = 0.0f; 
        } else {
            float value = G * (numerator / denominator);
            
            // Clamp value between -1 and 1 (optional but recommended)
            if (value > 1.0f) value = 1.0f;
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

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;
            float v = d_src[ny * width + nx];
            sum += v;
            sum_sq += v * v;
            count++;
        }
    }

    float mean = (count > 0) ? (sum / (float)count) : 0.0f;
    float var = (count > 0) ? ((sum_sq / (float)count) - (mean * mean)) : 0.0f;
    if (var < 0.0f) var = 0.0f;

    // Sobel-like 3x3 gradient magnitude with clamped borders.
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
    float gy = (p00 + 2.0f * p01 + p02) - (p20 + 2.0f * p21 + p22);
    float grad = sqrtf(gx * gx + gy * gy);

    d_mean[idx] = mean;
    d_var[idx] = var;
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
    float evi = d_evi[idx];
    float mean = d_ndvi_mean[idx];
    float var = d_ndvi_var[idx];
    float grad = d_ndvi_grad[idx];

    unsigned int flags = 0u;

    if (r <= 1u || n <= 1u || b <= 1u) flags |= QFLAG_BLACK;
    if (r >= 65530u || n >= 65530u || b >= 65530u) flags |= QFLAG_SATURATED;
    if (!isfinite(ndvi) || !isfinite(evi)) flags |= QFLAG_INVALID;

    float z = fabsf(ndvi - mean) / sqrtf(var + 1e-8f);
    if (z > 3.0f) flags |= QFLAG_OUTLIER;

    if (var < 1e-5f && grad < 1e-2f) flags |= QFLAG_FLAT;

    // Simple stripe-like detector: very low 3-point variation on both row and column.
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
    if (flags & QFLAG_BLACK) score -= 0.40f;
    if (flags & QFLAG_SATURATED) score -= 0.30f;
    if (flags & QFLAG_OUTLIER) score -= 0.20f;
    if (flags & QFLAG_FLAT) score -= 0.10f;
    if (flags & QFLAG_INVALID) score -= 0.50f;
    if (flags & QFLAG_STRIPE) score -= 0.20f;
    if (score < 0.0f) score = 0.0f;

    d_flags[idx] = flags;
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
    float evi = d_evi[idx];

    // If quality flags indicate invalid or saturated data, mark as uncertain.
    if (flags & (QFLAG_INVALID | QFLAG_SATURATED)) {
        d_health_class[idx] = HEALTH_UNCERTAIN;
        return;
    }

    // Conservative threshold-based health classes.
    if (ndvi < 0.20f || evi < 0.10f) {
        d_health_class[idx] = HEALTH_DEAD;
    } else if (ndvi < 0.45f || evi < 0.25f || (flags & (QFLAG_OUTLIER | QFLAG_BLACK | QFLAG_STRIPE))) {
        d_health_class[idx] = HEALTH_DISEASED;
    } else {
        d_health_class[idx] = HEALTH_HEALTHY;
    }
}

// Process a single scene (moved from main logic)
void process_scene(const Scene& scene, int window_size) {
    const char* red_path = scene.red_path;
    const char* nir_path = scene.nir_path;
    const char* blue_path = scene.blue_path;
    printf("\n===== Processing Scene: %s =====\n", scene.name);

    int width, height;
    int width_nir, height_nir;
    int width_blue, height_blue;

    double decode_seconds = 0.0;
    double quality_seconds = 0.0;
    double h2d_seconds = 0.0;
    double kernel_phase_seconds = 0.0;
    double d2h_seconds = 0.0;

    printf("Loading images for %s...\n", scene.name);
    auto decode_start = std::chrono::high_resolution_clock::now();

    struct BandLoadResult {
        unsigned short* data = nullptr;
        int w = 0;
        int h = 0;
        const char* name = nullptr;
    };

    BandLoadResult red_res, nir_res, blue_res;

    auto load_band_job = [](const char* path, const char* band_name, BandLoadResult* out) {
        out->name = band_name;
        out->data = load_raster_image(path, &out->w, &out->h);
    };

    std::thread red_thread(load_band_job, red_path, "RED", &red_res);
    std::thread nir_thread(load_band_job, nir_path, "NIR", &nir_res);
    std::thread blue_thread(load_band_job, blue_path, "BLUE", &blue_res);

    red_thread.join();
    nir_thread.join();
    blue_thread.join();

    unsigned short *h_red = red_res.data;
    unsigned short *h_nir = nir_res.data;
    unsigned short *h_blue = blue_res.data;
    width = red_res.w;
    height = red_res.h;
    width_nir = nir_res.w;
    height_nir = nir_res.h;
    width_blue = blue_res.w;
    height_blue = blue_res.h;

    if (!h_red || !h_nir || !h_blue) {
        if (!h_red) std::cerr << "Error: Failed to load " << red_res.name << " band." << std::endl;
        if (!h_nir) std::cerr << "Error: Failed to load " << nir_res.name << " band." << std::endl;
        if (!h_blue) std::cerr << "Error: Failed to load " << blue_res.name << " band." << std::endl;
        if (h_red) free(h_red);
        if (h_nir) free(h_nir);
        if (h_blue) free(h_blue);
        return;
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

    // Ensure dimensions match
    if (width != width_nir || height != height_nir || width != width_blue || height != height_blue) {
        std::cerr << "Error: Image dimensions do not match for " << scene.name << std::endl;
        std::cerr << "  RED: " << width << "x" << height << std::endl;
        std::cerr << "  NIR: " << width_nir << "x" << height_nir << std::endl;
        std::cerr << "  BLUE: " << width_blue << "x" << height_blue << std::endl;
        free(h_red); free(h_nir); free(h_blue);
        return;
    }

    int num_pixels = width * height;
    printf("All images loaded successfully: %d x %d (%d pixels)\n", width, height, num_pixels);

    auto quality_start = std::chrono::high_resolution_clock::now();
    report_band_quality(h_red, num_pixels, "RED");
    report_band_quality(h_nir, num_pixels, "NIR");
    report_band_quality(h_blue, num_pixels, "BLUE");
    auto quality_end = std::chrono::high_resolution_clock::now();
    quality_seconds = std::chrono::duration<double>(quality_end - quality_start).count();

    size_t input_bytes = num_pixels * sizeof(unsigned short);
    size_t output_bytes = num_pixels * sizeof(float);

    // Allocate host memory for results
    float *h_result_ndvi = (float*)malloc(output_bytes);
    float *h_result_evi  = (float*)malloc(output_bytes);
    float *h_ndvi_mean   = (float*)malloc(output_bytes);
    float *h_ndvi_var    = (float*)malloc(output_bytes);
    float *h_ndvi_grad   = (float*)malloc(output_bytes);
    float *h_quality_score = (float*)malloc(output_bytes);
    unsigned int *h_quality_flags = (unsigned int*)malloc(num_pixels * sizeof(unsigned int));
    unsigned char *h_health_class = (unsigned char*)malloc(num_pixels * sizeof(unsigned char));

    // Allocate device memory
    unsigned short *d_red, *d_nir, *d_blue;
    float *d_result_ndvi, *d_result_evi;
    float *d_ndvi_mean, *d_ndvi_var, *d_ndvi_grad;
    float *d_quality_score;
    unsigned int *d_quality_flags;
    unsigned char *d_health_class;
    
    CHECK_CUDA(cudaMalloc(&d_red, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_nir, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_blue, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_result_ndvi, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_result_evi, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_mean, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_var, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_ndvi_grad, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_quality_score, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_quality_flags, num_pixels * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_health_class, num_pixels * sizeof(unsigned char)));

    // Copy host to device
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_red, h_red, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nir, h_nir, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_blue, h_blue, input_bytes, cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    h2d_seconds = std::chrono::duration<double>(h2d_end - h2d_start).count();

    // Launch kernels
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    int local_radius = window_size / 2;

    printf("Launching Kernels with Grid: %d, Block: %d, Local Window: %dx%d\n", gridSize, blockSize, window_size, window_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    auto kernel_phase_start = std::chrono::high_resolution_clock::now();
    ndvi_kernel<<<gridSize, blockSize>>>(d_red, d_nir, d_result_ndvi, num_pixels);
    evi_kernel<<<gridSize, blockSize>>>(d_red, d_nir, d_blue, d_result_evi, num_pixels);
    local_stats_kernel<<<gridSize, blockSize>>>(
        d_result_ndvi,
        d_ndvi_mean,
        d_ndvi_var,
        d_ndvi_grad,
        width,
        height,
        local_radius
    );
    quality_flags_kernel<<<gridSize, blockSize>>>(
        d_red,
        d_nir,
        d_blue,
        d_result_ndvi,
        d_result_evi,
        d_ndvi_mean,
        d_ndvi_var,
        d_ndvi_grad,
        d_quality_flags,
        d_quality_score,
        width,
        height
    );
    classification_kernel<<<gridSize, blockSize>>>(
        d_result_ndvi,
        d_result_evi,
        d_quality_flags,
        d_health_class,
        num_pixels
    );

    cudaEventRecord(stop);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventSynchronize(stop);
    auto kernel_phase_end = std::chrono::high_resolution_clock::now();
    kernel_phase_seconds = std::chrono::duration<double>(kernel_phase_end - kernel_phase_start).count();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy device to host
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(h_result_ndvi, d_result_ndvi, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_result_evi, d_result_evi, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ndvi_mean, d_ndvi_mean, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ndvi_var, d_ndvi_var, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ndvi_grad, d_ndvi_grad, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_quality_score, d_quality_score, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_quality_flags, d_quality_flags, num_pixels * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_health_class, d_health_class, num_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    d2h_seconds = std::chrono::duration<double>(d2h_end - d2h_start).count();

    // Compute statistics
    double sum_ndvi = 0.0;
    double sum_evi = 0.0;
    double sum_ndvi_mean = 0.0;
    double sum_ndvi_var = 0.0;
    double sum_ndvi_grad = 0.0;
    double sum_quality_score = 0.0;
    int count_black = 0;
    int count_saturated = 0;
    int count_outlier = 0;
    int count_flat = 0;
    int count_invalid = 0;
    int count_stripe = 0;
    int count_dead = 0;
    int count_diseased = 0;
    int count_healthy = 0;
    int count_uncertain = 0;
    
    for (int i = 0; i < num_pixels; i++) {
        sum_ndvi += (double)h_result_ndvi[i];
        sum_evi  += (double)h_result_evi[i];
        sum_ndvi_mean += (double)h_ndvi_mean[i];
        sum_ndvi_var += (double)h_ndvi_var[i];
        sum_ndvi_grad += (double)h_ndvi_grad[i];
        sum_quality_score += (double)h_quality_score[i];

        unsigned int f = h_quality_flags[i];
        if (f & QFLAG_BLACK) count_black++;
        if (f & QFLAG_SATURATED) count_saturated++;
        if (f & QFLAG_OUTLIER) count_outlier++;
        if (f & QFLAG_FLAT) count_flat++;
        if (f & QFLAG_INVALID) count_invalid++;
        if (f & QFLAG_STRIPE) count_stripe++;

        unsigned char hc = h_health_class[i];
        if (hc == HEALTH_DEAD) count_dead++;
        else if (hc == HEALTH_DISEASED) count_diseased++;
        else if (hc == HEALTH_HEALTHY) count_healthy++;
        else count_uncertain++;
    }

    printf("--- Results for %s ---\n", scene.name);
    printf("Average NDVI: %f\n", sum_ndvi / num_pixels);
    printf("Average EVI:  %f\n", sum_evi / num_pixels);
    printf("Average NDVI local mean (%dx%d): %f\n", window_size, window_size, sum_ndvi_mean / num_pixels);
    printf("Average NDVI local var  (%dx%d): %f\n", window_size, window_size, sum_ndvi_var / num_pixels);
    printf("Average NDVI Sobel-like grad: %f\n", sum_ndvi_grad / num_pixels);
    printf("Average Quality Score: %f\n", sum_quality_score / num_pixels);

    printf("Quality Flags Summary:\n");
    printf("  Black pixels:     %d (%.2f%%)\n", count_black, 100.0 * (double)count_black / (double)num_pixels);
    printf("  Saturated pixels: %d (%.2f%%)\n", count_saturated, 100.0 * (double)count_saturated / (double)num_pixels);
    printf("  Outlier pixels:   %d (%.2f%%)\n", count_outlier, 100.0 * (double)count_outlier / (double)num_pixels);
    printf("  Flat pixels:      %d (%.2f%%)\n", count_flat, 100.0 * (double)count_flat / (double)num_pixels);
    printf("  Invalid pixels:   %d (%.2f%%)\n", count_invalid, 100.0 * (double)count_invalid / (double)num_pixels);
    printf("  Stripe-like:      %d (%.2f%%)\n", count_stripe, 100.0 * (double)count_stripe / (double)num_pixels);

    printf("Health Class Summary:\n");
    printf("  Dead:      %d (%.2f%%)\n", count_dead, 100.0 * (double)count_dead / (double)num_pixels);
    printf("  Diseased:  %d (%.2f%%)\n", count_diseased, 100.0 * (double)count_diseased / (double)num_pixels);
    printf("  Healthy:   %d (%.2f%%)\n", count_healthy, 100.0 * (double)count_healthy / (double)num_pixels);
    printf("  Uncertain: %d (%.2f%%)\n", count_uncertain, 100.0 * (double)count_uncertain / (double)num_pixels);

    printf("Timings for %s:\n", scene.name);
    printf("  Raw Load Time: %.6f seconds\n", decode_seconds);
    printf("  Quality Check Time: %.6f seconds\n", quality_seconds);
    printf("  H2D Copy Time: %.6f seconds\n", h2d_seconds);
    printf("  Kernel Phase Time: %.6f seconds\n", kernel_phase_seconds);
    printf("  D2H Copy Time: %.6f seconds\n", d2h_seconds);

    // Cleanup
    cudaFree(d_red); cudaFree(d_nir); cudaFree(d_blue);
    cudaFree(d_result_ndvi); cudaFree(d_result_evi);
    cudaFree(d_ndvi_mean); cudaFree(d_ndvi_var); cudaFree(d_ndvi_grad);
    cudaFree(d_quality_score); cudaFree(d_quality_flags);
    cudaFree(d_health_class);
    
    free(h_red); free(h_nir); free(h_blue);
    free(h_result_ndvi); free(h_result_evi);
    free(h_ndvi_mean); free(h_ndvi_var); free(h_ndvi_grad);
    free(h_quality_score); free(h_quality_flags);
    free(h_health_class);
}

// NOTE: RAW meta loader removed — this build only supports JP2/TIFF via GDAL.

// Load RAW uint16 image using companion _meta.txt (width height).
unsigned short* load_raster_image(const char* filename, int* width, int* height) {
    // Only JP2/TIFF via GDAL are supported in this build.
    std::string s(filename);
    std::string ext;
    size_t pos = s.find_last_of('.');
    if (pos != std::string::npos) ext = s.substr(pos);
    if (ext == ".jp2" || ext == ".JP2" || ext == ".tif" || ext == ".tiff" || ext == ".TIF" || ext == ".TIFF") {
        // use GDAL-based loader
        GDALDataset* ds = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
        if (!ds) {
            std::cerr << "Error: GDAL failed to open " << filename << std::endl;
            return nullptr;
        }
        *width = ds->GetRasterXSize();
        *height = ds->GetRasterYSize();
        GDALRasterBand* band = ds->GetRasterBand(1);
        size_t num_pixels = static_cast<size_t>(*width) * static_cast<size_t>(*height);
        size_t num_bytes = num_pixels * sizeof(unsigned short);
        unsigned short* data = static_cast<unsigned short*>(malloc(num_bytes));
        if (!data) {
            std::cerr << "Error: Failed to allocate memory for " << filename << std::endl;
            GDALClose(ds);
            return nullptr;
        }

        CPLErr err = band->RasterIO(GF_Read, 0, 0, *width, *height,
                                    data, *width, *height, GDT_UInt16,
                                    0, 0);
        GDALClose(ds);
        if (err != CE_None) {
            std::cerr << "Error: GDAL RasterIO failed for " << filename << std::endl;
            free(data);
            return nullptr;
        }

        printf("  Loaded %s via GDAL: %dx%d = %zu pixels\n", filename, *width, *height, num_pixels);
        return data;
    }

    std::cerr << "Error: Unsupported file type for " << filename << ". Only JP2/TIFF is supported in this build." << std::endl;
    return nullptr;
}

void report_band_quality(const unsigned short* band, int num_pixels, const char* band_name, float black_ratio_threshold) {
    if (!band || num_pixels <= 0) {
        std::cerr << "Error: " << band_name << " band is empty or null." << std::endl;
        return;
    }

    unsigned short min_v = band[0];
    unsigned short max_v = band[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    int black_pixels = 0;

    for (int i = 0; i < num_pixels; i++) {
        unsigned short v = band[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v <= 1) black_pixels++;

        double dv = static_cast<double>(v);
        sum += dv;
        sum_sq += dv * dv;
    }

    double mean = sum / static_cast<double>(num_pixels);
    double variance = (sum_sq / static_cast<double>(num_pixels)) - (mean * mean);
    if (variance < 0.0) variance = 0.0; // Numerical guard
    double stddev = std::sqrt(variance);

    float black_ratio = static_cast<float>(black_pixels) / static_cast<float>(num_pixels);

    printf(
        "%s quality | min=%u, max=%u, std=%.2f, black=%.2f%%\n",
        band_name,
        static_cast<unsigned>(min_v),
        static_cast<unsigned>(max_v),
        stddev,
        black_ratio * 100.0f
    );

    if (black_ratio >= black_ratio_threshold) {
        printf(
            "WARNING: %s is mostly black (%.2f%% <= 1 value pixels). Possible nodata/black band.\n",
            band_name,
            black_ratio * 100.0f
        );
    }
    if (max_v == min_v || stddev < 1e-9) {
        printf("WARNING: %s appears constant. Possible corrupted or empty information band.\n", band_name);
    }
}

// ==========================================
// HOST MAIN FUNCTION
// ==========================================
int main(int argc, char** argv) {
    // Initialize GDAL once from the main thread to avoid race conditions
    GDALAllRegister();

    // 1. PROCESS IMAGES
    auto total_start = std::chrono::high_resolution_clock::now();
    double cpu_start = cpu_seconds_now();

    int window_size = 3;

    if (argc == 2) {
        window_size = std::atoi(argv[1]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " [window_size(3|5)]" << std::endl;
        std::cerr << "Configure scenes in the SCENES array at the top of the file." << std::endl;
        return -1;
    }

    if (window_size != 3 && window_size != 5) {
        std::cerr << "Error: window_size must be 3 or 5." << std::endl;
        return -1;
    }

    printf("Processing %d scenes...\n", NUM_SCENES);
    for (int i = 0; i < NUM_SCENES; i++) {
        try {
            process_scene(SCENES[i], window_size);
        } catch (const std::exception& e) {
            std::cerr << "Error processing scene " << i+1 << ": " << e.what() << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(total_end - total_start).count();
    printf("Total Execution Time: %.6f seconds\n", total_seconds);

    return 0;
}