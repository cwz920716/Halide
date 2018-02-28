#include "Halide.h"

namespace {

using namespace Halide;

class InnerProductLayer : public Halide::Generator<InnerProductLayer> {
public:
    typedef Generator<InnerProductLayer> Base;
    using Base::target;
    using Base::get_target;
    using Base::natural_vector_size;
    template<typename T2> using Input = typename Base::template Input<T2>;
    template<typename T2> using Output = typename Base::template Output<T2>;

    Input<Buffer<float>>  bottom_ = {"bottom", 2};
    Input<Buffer<float>>  weight_ = {"weight", 2};

    Output<Buffer<float>> top_ = {"top", 2};

    void generate() {
        const Expr num_rows = bottom_.height();
        const Expr num_cols = weight_.width();
        const Expr sum_size = bottom_.width();

        int block_size = natural_vector_size<float>();

        Func bottom_tmp("bottom_tmp");
        Input<Buffer<float>> *bottom_in = &bottom_;
        bottom_tmp(i, j) =
            BoundaryConditions::constant_exterior(*bottom_in, cast<T>(0))(i, j);
        

        Func weightT("weightT");

        Var i("i"), j("j"), k("k");
        Func prod("prod");
        prod(k, i, j) = bottom(i, k) * weightT(k, j);

        RDom rv(0, sum_size);
        top(i, j) += prod(rv, i, j);
    }
};

}  // namespace
