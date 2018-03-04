#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <cblas.h>
#include <halide_blas.h>

#include "inner_product_layer.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide;
using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char *argv[]) {
    int M = 512, N = 512, K = 512;
    Buffer<float> bot(K, M), w(K, N), top(N, M);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++)
            bot(j, i) = rand();
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++)
            w(j, i) = rand();
    }

    inner_product_layer(bot, w, top);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += bot(k, i) * w(k, j);
            }
            assert(top(j, i) == sum);
        }
    }

    std::cout << "Inner Product: PASSED.\n";

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(10, 10, [&]() {
        inner_product_layer(bot, w, top);
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    double min_t_hblas = benchmark(10, 10, [&]() {
        hblas_sgemm(HblasRowMajor, HblasNoTrans, HblasTrans, M, N, K, 1.0f, bot.begin(), K, w.begin(), K, 1.0f, top.begin(), N);
    });
    printf("HBlas time: %gms\n", min_t_hblas * 1e3);

    double min_t_cblas = benchmark(10, 10, [&]() {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, bot.begin(), K, w.begin(), K, 1.0f, top.begin(), N);
    });
    printf("CBlas time: %gms\n", min_t_cblas * 1e3);

    return 0;
}
