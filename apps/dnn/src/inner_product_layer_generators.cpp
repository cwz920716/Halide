#include "Halide.h"

namespace {

using namespace Halide;

inline
void blis_gemm(Func &A, Func &B, Func &C, Expr K,
               bool transposeA = false, bool transposeB = false,
               TailStrategy guard = TailStrategy::GuardWithIf,
               int nc=16,
               int kc=32,
               int mc=32,
               int mr=8,
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
        const Expr num_cols = weight_.height();
        const Expr sum_size = bottom_.width();

        Var i("i"), j("j");
        Func bottom_tmp("bottom_tmp"), weight_tmp("weight_tmp");
        Input<Buffer<float>> *bottom_in = &bottom_;
        Input<Buffer<float>> *weight_in = &weight_;
        bottom_tmp(i, j) =
            BoundaryConditions::constant_exterior(*bottom_in, 0.0f)(i, j);
        weight_tmp(i, j) =
            BoundaryConditions::constant_exterior(*weight_in, 0.0f)(i, j);

        Func C;
        blis_gemm(bottom_tmp, weight_tmp, C, sum_size, false, true, TailStrategy::Auto);
        C.compute_root();
        top_(j, i) = C(j, i);

        bottom_.dim(0).set_min(0).dim(1).set_min(0);
        weight_.dim(0).set_bounds(0, sum_size).dim(1).set_min(0);
        top_.dim(0).set_bounds(0, num_cols).dim(1).set_bounds(0, num_rows);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(InnerProductLayer, inner_product_layer)
