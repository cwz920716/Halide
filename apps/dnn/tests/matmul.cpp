// Halide tutorial lesson: Using fixed point math.

// This lesson demonstrates how to inspect what the Halide compiler is producing.

// On linux, you can compile and run it like so:
// g++ matmul.cpp -I./include -L./bin -lHalide -lpthread -ldl -std=c++11

#include "Halide.h"
#include <stdio.h>
#include <iostream>

// This time we'll just import the entire Halide namespace
using namespace Halide;

inline
void blis_gemm(Func &A, Func &B, Func &C, Expr K,
               bool transposeA = false, bool transposeB = false,
               TailStrategy guard = TailStrategy::GuardWithIf,
               int nc=32,
               int kc=16,
               int mc=8,
               int mr=4,
               int nr=4) {
    Func Aref("Aref"), Bref("Bref");
    Func Bp("Bp"), Btmp("Btmp"), Ap("Ap"), Atmp("Atmp");
    Var i("i"), j("j"), k("k");
    Var ji("ji"), jo("jo"), ko("ko"), ki("ki");
    Var ii("ii"), io("io"), iio("iio"), iii("iii"), jio("jio"), jii("jii"), t("t");
    RVar rv_i("rv_i"), rv_o("rv_o");

    if (transposeA) {
        Aref(j, i) = A(i, j);
    } else {
        Aref(j, i) = A(j, i);
    }

    if (transposeB) {
        Bref(j, i) = B(i, j);
    } else {
        Bref(j, i) = B(j, i);
    }

    Bp(j, ii, io) = Bref(j, io*kc+ii);
    Btmp(j, i) = Bp(j, i % kc, i / kc);

    Ap(ji, jo, ii, io) = Aref(jo*kc+ji, io*mc+ii);
    Atmp(j, i) = Ap(j % kc, j / kc, i % mc, i / mc);

    RDom rv(0, K);
    Func prod("prod");
    prod(k, j, i) = cast<float>(Atmp(k, i) * Btmp(j, k));
    C(j, i) += prod(rv, j, i);

    // Schedule
    Btmp.compute_at(C, rv_o);
    Atmp.compute_at(C, io);
    C.update(0).split(j, jo, ji, nc, guard)
               .split(rv, rv_o, rv_i, kc, guard)
               .split(i, io, ii, mc, guard)
               .split(ji, jio, jii, mr, guard)
               .split(ii, iio, iii, nr, guard)
               .reorder(jii, iii, rv_i, iio, jio, io, rv_o, jo)
               .unroll(iii, 2)
               .vectorize(jii)
               .rename(jo, t).parallel(t);
}

int main(int argc, char **argv) {

    Func A("A"), B("B"), C("C"), Bp("Bp"), Btmp("Btmp"), Ap("Ap"), Atmp("Atmp");
    Var i("i"), j("j"), k("k");
    Var ji("ji"), jo("jo"), ko("ko"), ki("ki");
    Var ii("ii"), io("io"), iio("iio"), iii("iii"), jio("jio"), jii("jii"), t("t");
    RVar rv_i("rv_i"), rv_o("rv_o");

    int M = 256, N = 64, K = 128;

    // i-th row, j-th column
    A(j, i) = cast<float>(i + j);
    B(j, i) = cast<float>(i - j);
    A.compute_root();
    B.compute_root();

    // Realize the function to produce an output image. We'll keep it
    // very small for this lesson.
    Buffer<float> Adata = A.realize(K, M);
    Buffer<float> Bdata = B.realize(K, N);

    blis_gemm(A, B, C, K, false, true, TailStrategy::Auto);

    // That line compiled and ran the pipeline. Try running this
    // lesson with the environment variable HL_DEBUG_CODEGEN set to
    // 1. It will print out the various stages of compilation, and a
    // pseudocode representation of the final pipeline.
    // Click to show output ...

    // If you set HL_DEBUG_CODEGEN to a higher number, you can see
    // more and more details of how Halide compiles your pipeline.
    // Setting HL_DEBUG_CODEGEN=2 shows the Halide code at each stage
    // of compilation, and also the llvm bitcode we generate at the
    // end.

    // Halide will also output an HTML version of this output, which
    // supports syntax highlighting and code-folding, so it can be
    // nicer to read for large pipelines. Open gradient.html with your
    // browser after running this tutorial.
    Buffer<float> Cdata = C.realize(N, M);
    C.compile_to_lowered_stmt("C.html", {}, HTML);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) {
                s += Adata(k, i) * Bdata(k, j);
            }
            assert(s == Cdata(j, i));
        }
    }

    // You can usually figure out what code Halide is generating using
    // this pseudocode. In the next lesson we'll see how to snoop on
    // Halide at runtime.

    printf("Success!\n");
    return 0;
}
