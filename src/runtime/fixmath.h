#ifndef RUNTIME_FIXMATH_H
#define RUNTIME_FIXMATH_H

#include "HalideRuntime.h"

typedef int32_t fix16_t;

const fix16_t FOUR_DIV_PI  = 0x145F3;            /*!< Fix16 value of 4/PI */
const fix16_t _FOUR_DIV_PI2 = 0xFFFF9840;        /*!< Fix16 value of -4/PI² */
const fix16_t X4_CORRECTION_COMPONENT = 0x399A; 	/*!< Fix16 value of 0.225 */
const fix16_t PI_DIV_4 = 0x0000C90F;             /*!< Fix16 value of PI/4 */
const fix16_t THREE_PI_DIV_4 = 0x00025B2F;       /*!< Fix16 value of 3PI/4 */

const fix16_t fix16_maximum  = 0x7FFFFFFF; /*!< the maximum value of fix16_t */
const fix16_t fix16_minimum  = 0x80000000; /*!< the minimum value of fix16_t */
const fix16_t fix16_overflow = 0x80000000; /*!< the value used to indicate overflows when FIXMATH_NO_OVERFLOW is not specified */

const fix16_t fix16_pi  = 205887;     /*!< fix16_t value of pi */
const fix16_t fix16_e   = 178145;     /*!< fix16_t value of e */
const fix16_t fix16_one = 0x00010000; /*!< fix16_t value of 1 */

HALIDE_ALWAYS_INLINE float fix16_to_float(fix16_t a) {
  return (float)a / fix16_one;
}

HALIDE_ALWAYS_INLINE fix16_t fix16_from_float(float a) {
  float temp = a * fix16_one;
#ifndef FIXMATH_NO_ROUNDING
  temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
  return (fix16_t)temp;
}

HALIDE_ALWAYS_INLINE fix16_t fix16_add(fix16_t a, fix16_t b) {
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

HALIDE_ALWAYS_INLINE fix16_t fix16_sub(fix16_t a, fix16_t b) {
  uint32_t _a = a, _b = b;
  uint32_t diff = _a - _b;

  // Overflow can only happen if sign of a != sign of b, and then
  // it causes sign of diff != sign of a.
  if (((_a ^ _b) & 0x80000000) && ((_a ^ diff) & 0x80000000))
    return fix16_overflow;

  return diff;
}

HALIDE_ALWAYS_INLINE fix16_t fix16_mul(fix16_t inArg0, fix16_t inArg1) {
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

HALIDE_ALWAYS_INLINE fix16_t fix16_div(fix16_t a, fix16_t b) {
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

#endif  // RUNTIME_FIXMATH_H
