// Halide tutorial lesson: Using fixed point math.

// This lesson demonstrates how to inspect what the Halide compiler is producing.

// On linux, you can compile and run it like so:
// g++ fixmath_test.cpp -I./include -L./bin -lHalide -lpthread -ldl -std=c++11

#include "Halide.h"
#include <stdio.h>
#include <iostream>

// This time we'll just import the entire Halide namespace
using namespace Halide;

int main(int argc, char **argv) {

    // We'll start by defining the simple single-stage imaging
    // pipeline from lesson 1.

    // This lesson will be about debugging, but unfortunately in C++,
    // objects don't know their own names, which makes it hard for us
    // to understand the generated code. To get around this, you can
    // pass a string to the Func and Var constructors to give them a
    // name for debugging purposes.
    Func gradient("gradient");
    Var x("x"), y("y");
    gradient(x, y) = fixed_cast(y) * x + fixed_cast(x - y) / (fixed_cast(x+1) * (y+1));

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
    gradient.compile_to_lowered_stmt("gradient.html", {}, HTML);
    gradient.compile_to_c("gradient.cpp", {});

    // Realize the function to produce an output image. We'll keep it
    // very small for this lesson.
    assert(gradient.output_types()[0] == Fix16());
    Buffer<fix16_t> res = gradient.realize(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float s = float(j) * i + float(i - j) / (float(i+1) * (j+1));
            // std::cout << "(" << i << ", " << j << "):\t" << fix16_from_float(s) << " ==? " << res(i, j) << "\n";
            std::cout << "(" << i << ", " << j << "):\t" << s << " ==? " << fix16_to_float(res(i, j)) << "\n";
            assert(fabs(fix16_to_float(res(i, j)) - s) <= (1.0 / (1 << 16)));
        }
    }

    // You can usually figure out what code Halide is generating using
    // this pseudocode. In the next lesson we'll see how to snoop on
    // Halide at runtime.

    printf("Success!\n");
    return 0;
}
