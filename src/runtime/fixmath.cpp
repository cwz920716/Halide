#include "HalideRuntime.h"

WEAK extern "C" float halide_fix16_to_float(fix16_t a) {
  return (float)a / fix16_one;
}

WEAK extern "C" fix16_t halide_fix16_from_float(float a) {
  float temp = a * fix16_one;
#ifndef FIXMATH_NO_ROUNDING
  temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
  return (fix16_t)temp;
}

WEAK extern "C" fix16_t halide_fix16_add(fix16_t a, fix16_t b) {
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

WEAK extern "C" fix16_t halide_fix16_sub(fix16_t a, fix16_t b) {
  uint32_t _a = a, _b = b;
  uint32_t diff = _a - _b;

  // Overflow can only happen if sign of a != sign of b, and then
  // it causes sign of diff != sign of a.
  if (((_a ^ _b) & 0x80000000) && ((_a ^ diff) & 0x80000000))
    return fix16_overflow;

  return diff;
}

WEAK extern "C" fix16_t halide_fix16_mul(fix16_t inArg0, fix16_t inArg1) {
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

WEAK extern "C" fix16_t halide_fix16_div(fix16_t a, fix16_t b) {
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
