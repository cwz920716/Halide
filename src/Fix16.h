#ifndef HALIDE_FIX16_H
#define HALIDE_FIX16_H
#include "runtime/HalideRuntime.h"
#include "runtime/fixmath.h"
#include <stdint.h>
#include <string>
#include "Util.h"

namespace Halide {

/** 
 *  This is a Fixed 16.16 type to represent rational numbers in 32 bits
 *  with fixed decimal points. The exponential part is 2^-16, i.e.,
 *  16 bits for fractional part.
 * */
class Fix16_t {
 public:
    /// \name Constructors
    /// @{

    /** Construct from a float, double, or int using
     * round-to-nearest-ties-to-even. Out-of-range values become +/-
     * infinity.
     */
    // @{
    EXPORT explicit Fix16_t(float value) : data_(fix16_from_float(value)) {}
    EXPORT explicit Fix16_t(double value) : data_(fix16_from_float(value)) {}
    EXPORT explicit Fix16_t(int value) : data_(fix16_from_float(value)) {}
    // @}

    /** Construct a Fix16_t with the bits initialised to 0. This represents
     * positive zero.*/
    EXPORT Fix16_t() : data_(0) {}

    /// @}

    // Use explicit to avoid accidently raising the precision
    /** Cast to float */
    EXPORT explicit operator float() const {
      return fix16_to_float(data_);
    }

    /** Cast to double */
    EXPORT explicit operator double() const {
      return fix16_to_float(data_);
    }

    // Be explicit about how the copy constructor is expected to behave
    EXPORT Fix16_t(const Fix16_t&) = default;

    // Be explicit about how assignment is expected to behave
    EXPORT Fix16_t& operator=(const Fix16_t&) = default;

    /** \name Convenience "constructors"
     */
    /**@{*/

    /** Get a new Fix16_t with the given raw bits
     *
     * \param bits The bits conformant to IEEE754 binary16
     */
    EXPORT static Fix16_t make_from_bits(uint32_t bits) {
      Fix16_t fp;
      fp.data_ = bits;
      return fp;
    }

    /**@}*/

    /** Return a new Fix16_t with a negated sign bit*/
    EXPORT Fix16_t operator-() const {
      return make_from_bits(-data_);
    }

    /** Arithmetic operators. */
    // @{
    EXPORT Fix16_t operator+(Fix16_t rhs) const {
      return make_from_bits(fix16_add(data_, rhs.data_));
    }

    EXPORT Fix16_t operator-(Fix16_t rhs) const {
      return make_from_bits(fix16_sub(data_, rhs.data_));
    }

    EXPORT Fix16_t operator*(Fix16_t rhs) const {
      return make_from_bits(fix16_mul(data_, rhs.data_));
    }

    EXPORT Fix16_t operator/(Fix16_t rhs) const {
      return make_from_bits(fix16_div(data_, rhs.data_));
    }

    // @}

    /** Comparison operators */
    // @{
    EXPORT bool operator==(Fix16_t rhs) const {
      return data_ == rhs.data_;
    }

    EXPORT bool operator!=(Fix16_t rhs) const { return !(*this == rhs); }

    EXPORT bool operator>(Fix16_t rhs) const {
      return data_ > rhs.data_;
    }

    EXPORT bool operator<(Fix16_t rhs) const {
      return data_ < rhs.data_;
    }

    EXPORT bool operator>=(Fix16_t rhs) const { return (*this > rhs) || (*this == rhs); }

    EXPORT bool operator<=(Fix16_t rhs) const { return (*this < rhs) || (*this == rhs); }
    // @}

    /** Returns the bits that represent this Fix16_t.
     *
     *  An alternative method to access the bits is to cast a pointer
     *  to this instance as a pointer to a uint16_t.
     **/
    EXPORT uint32_t to_bits() const {
      return data_;
    }

 private:
  // the raw bits.
  fix16_t data_;
};

}  // namespace Halide

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<Halide::Fix16_t>() {
    return halide_type_t(halide_type_fix16, 32);
}

#endif  // HALIDE_FIX16_H
