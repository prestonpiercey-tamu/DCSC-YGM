#pragma once
#include <cstdint>
#include <limits>

/**
 * @brief Reproducible, bijective, format-preserving permutation over [min_id, max_id].
 *
 * Uses a power-of-two bijection + cycle walking:
 *   - Choose m = ceil(log2(R)), where R = max_id - min_id + 1
 *   - Build a bijection on {0..2^m-1} parameterized by 'seed'
 *   - Cycle-walk until the permuted value falls in [0, R)
 *
 * Properties:
 *   - Reproducible given the same (min_id, max_id, seed)
 *   - O(1) memory, SPMD-friendly (no comms)
 *   - True permutation of [min_id, max_id] (no collisions)
 */
class FppPermuter {
public:
  using u32 = uint32_t;
  using u64 = uint64_t;

  FppPermuter(u32 min_id, u32 max_id, u64 seed)
      : min_id_(min_id), max_id_(max_id), seed_(seed) {
    // Handle empty/degenerate ranges defensively.
    if (max_id_ <= min_id_) {
      // Make it a no-op permutation to be safe.
      min_id_ = 0;
      max_id_ = 0;
    }
    const u64 R64 = static_cast<u64>(max_id_) - static_cast<u64>(min_id_) + 1ull;

    // R==0 means full 2^32 range.
    full_32bit_range_ = (R64 >= (1ull << 32));
    if (!full_32bit_range_) {
      R_ = static_cast<u32>(R64);
      m_ = (R_ <= 1u) ? 1u : ceil_log2_u64(R_);
      mask_ = (m_ == 32u) ? 0xFFFFFFFFu : ((1u << m_) - 1u);
    } else {
      R_ = 0u;
      m_ = 32u;
      mask_ = 0xFFFFFFFFu;
    }

    key_ = mix_key64_to_32(seed_);
    // Derive two odd round constants from key (cached)
    k1_ = key_ * 0x9E3779B1u + 0x85EBCA77u; k1_ |= 1u;
    k2_ = key_ * 0xC2B2AE3Du + 0x27D4EB2Fu; k2_ |= 1u;
  }

  /// Permute a single id. If it's outside [min_id, max_id], returns it unchanged.
  inline u32 operator()(u32 id) const {
    if (id < min_id_ || id > max_id_) return id;

    if (full_32bit_range_) {
      // Full 2^32: direct bijection (no cycle walking).
      const u32 x = id - min_id_;
      const u32 y = permute_pow2(x);
      return y + min_id_;
    } else {
      const u32 x0 = id - min_id_;
      return fpp_permute_in_range(x0) + min_id_;
    }
  }

  /// Accessors
  inline u32 min_id() const { return min_id_; }
  inline u32 max_id() const { return max_id_; }
  inline u64 seed()   const { return seed_;   }

private:
  // --- Helpers ---------------------------------------------------------------

  static inline u32 mix_key64_to_32(u64 seed) {
    // SplitMix64 finalizer, then fold to 32 bits.
    u64 z = seed + 0x9E3779B97F4A7C15ull;
    z ^= (z >> 30); z *= 0xBF58476D1CE4E5B9ull;
    z ^= (z >> 27); z *= 0x94D049BB133111EBull;
    z ^= (z >> 31);
    return static_cast<u32>((z ^ (z >> 32)) & 0xFFFFFFFFu);
  }

  static inline unsigned ceil_log2_u64(u64 n) {
    // Precondition: n >= 1
    unsigned l = 0; u64 v = n - 1;
    while (v > 0) { v >>= 1; ++l; }
    return l;
  }

  inline u32 permute_pow2(u32 x) const {
    // Bijective on {0..2^m-1}; all ops masked by 'mask_' (mod 2^m).
    u32 v = x & mask_;
    v ^= key_;                     v &= mask_;
    v ^= (v >> (m_/2 ? m_/2 : 1)); v &= mask_;
    v = mul_masked(v, k1_);        v &= mask_;
    v ^= (v >> ((m_+1)/3 ? (m_+1)/3 : 1)); v &= mask_;
    v = mul_masked(v, k2_);        v &= mask_;
    v ^= (v >> (m_/2 ? m_/2 : 1)); v &= mask_;
    v += key_;                     v &= mask_;
    return v;
  }

  inline u32 fpp_permute_in_range(u32 x_in_range) const {
    // Cycle-walk until the result falls in [0, R_)
    u32 x = x_in_range;
    do {
      x = permute_pow2(x);
    } while (x >= R_);
    return x;
  }

  inline u32 mul_masked(u32 x, u32 k) const {
    // Multiplication modulo 2^m is equivalent to normal mul then mask.
    // k is odd by construction.
    return (x * k) & mask_;
  }

  // --- Data ------------------------------------------------------------------
  u32 min_id_{0}, max_id_{0};
  u64 seed_{0};

  bool full_32bit_range_{false};
  u32  R_{0};        // range size, or 0 meaning 2^32
  unsigned m_{32};   // bits of the pow2 domain
  u32  mask_{0xFFFFFFFFu};

  u32  key_{0};      // mixed 32-bit key from seed
  u32  k1_{0}, k2_{0}; // odd round constants
};
