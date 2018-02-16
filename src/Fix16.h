#ifndef HALIDE_FIX16_H
#define HALIDE_FIX16_H
#include "runtime/HalideRuntime.h"
#include <stdint.h>
#include <string>
#include "Util.h"

namespace Halide {

namespace {

inline float fix16_to_float(fix16_t a) {
  return (float)a / fix16_one;
}

inline fix16_t fix16_from_float(float a) {
  float temp = a * fix16_one;
#ifndef FIXMATH_NO_ROUNDING
  temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
  return (fix16_t)temp;
}

inline fix16_t fix16_add(fix16_t a, fix16_t b) {
  // Use unsigned integers because overflow with signed integers is
  // an undefined operation (http://www.airs.com/blog/archives/120).
  uint32_t _a = a, _b = b;
  uint32_t sum = _a + _b;

  // Overflow can only happen if sign of a == sign of b, and then
  // it causes sign of sum != sign of a.
  if (!((_a ^ _b) & 0x80000000) && ((_a ^ sum) & 0x80000000))
    return fix16_overflow;

  return sum;
}

inline fix16_t fix16_sub(fix16_t a, fix16_t b) {
  uint32_t _a = a, _b = b;
  uint32_t diff = _a - _b;

  // Overflow can only happen if sign of a != sign of b, and then
  // it causes sign of diff != sign of a.
  if (((_a ^ _b) & 0x80000000) && ((_a ^ diff) & 0x80000000))
    return fix16_overflow;

  return diff;
}

inline fix16_t fix16_mul(fix16_t inArg0, fix16_t inArg1) {
  int64_t product = (int64_t)inArg0 * inArg1;

  #ifndef FIXMATH_NO_OVERFLOW
    // The upper 17 bits should all be the same (the sign).
    uint32_t upper = (product >> 47);
  #endif  // FIXMATH_NO_OVERFLOW

  if (product < 0) {
    #ifndef FIXMATH_NO_OVERFLOW
      if (~upper)
        return fix16_overflow;
    #endif  // FIXMATH_NO_OVERFLOW

    #ifndef FIXMATH_NO_ROUNDING
      // This adjustment is required in order to round -1/2 correctly
      product--;
    #endif  // FIXMATH_NO_ROUNDING
  } else {
    #ifndef FIXMATH_NO_OVERFLOW
      if (upper)
        return fix16_overflow;
    #endif  // FIXMATH_NO_OVERFLOW
  }

  #ifdef FIXMATH_NO_ROUNDING
    return product >> 16;
  #else
    fix16_t result = product >> 16;
    result += (product & 0x8000) >> 15;
    return result;
  #endif  // FIXMATH_NO_ROUNDING
}

inline fix16_t fix16_div(fix16_t a, fix16_t b) {
  // This uses a hardware 32/32 bit division multiple times, until we have
  // computed all the bits in (a<<17)/b. Usually this takes 1-3 iterations.

  if (b == 0)
    return fix16_minimum;

  uint32_t remainder = (a >= 0) ? a : (-a);
  uint32_t divider = (b >= 0) ? b : (-b);
  uint32_t quotient = 0;
  int bit_pos = 17;

  // Kick-start the division a bit.
  // This improves speed in the worst-case scenarios where N and D are large
  // It gets a lower estimate for the result by N/(D >> 17 + 1).
  if (divider & 0xFFF00000) {
    uint32_t shifted_div = ((divider >> 17) + 1);
    quotient = remainder / shifted_div;
    remainder -= ((uint64_t)quotient * divider) >> 17;
  }

  // If the divider is divisible by 2^n, take advantage of it.
  while (!(divider & 0xF) && bit_pos >= 4) {
    divider >>= 4;
    bit_pos -= 4;
  }

  while (remainder && bit_pos >= 0) {
    // Shift remainder as much as we can without overflowing
    int shift = __builtin_clz(remainder);
    if (shift > bit_pos) shift = bit_pos;
    remainder <<= shift;
    bit_pos -= shift;

    uint32_t div = remainder / divider;
    remainder = remainder % divider;
    quotient += div << bit_pos;

    #ifndef FIXMATH_NO_OVERFLOW
      if (div & ~(0xFFFFFFFF >> bit_pos))
        return fix16_overflow;
    #endif  // FIXMATH_NO_OVERFLOW

    remainder <<= 1;
    bit_pos--;
  }

  #ifndef FIXMATH_NO_ROUNDING
    // Quotient is always positive so rounding is easy
    quotient++;
  #endif  // FIXMATH_NO_ROUNDING

  fix16_t result = quotient >> 1;

  // Figure out the sign of the result
  if ((a ^ b) & 0x80000000) {
    #ifndef FIXMATH_NO_OVERFLOW
      if (result == fix16_minimum)
        return fix16_overflow;
    #endif  // FIXMATH_NO_OVERFLOW

    result = -result;
  }

  return result;
}

}

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
