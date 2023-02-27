/**
 * SHA1 calculation class template.
 *
 * @license %, public domain
 * @author Steve Reid <steve@edmweb.com> (original C source)
 * @author Volker Grabsch <vog@notjusthosting.com> (Small changes to fit into bglibs)
 * @author Bruce Guenter <bruce@untroubled.org> (Translation to simpler C++ Code)
 * @author Stefan Wilhelm <cerbero s@atwillys.de> (class template rewrite, types, endianess)
 *
 * @file sha1.hh
 * @ccflags
 * @ldflags
 * @platform linux, bsd, windows
 * @standard >= c++98
 *
 */
#ifndef NOESIS_FRAMEWORK_UTILS_CHECKSUM_HPP_
#define NOESIS_FRAMEWORK_UTILS_CHECKSUM_HPP_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace noesis {
namespace utils {

/**
 * @class CheckSumSha1
 * @template
 */
template <typename _CharType=char>
class CheckSumSha1
{
public:

  // Alias
  using str_t = std::basic_string<_CharType>;

public:

  inline CheckSumSha1() {
    buf_.reserve(64); clear();
  }

  virtual ~CheckSumSha1() = default;

public:

  /**
   * Clear/reset all internal buffers and states.
   */
  void clear() {
    sum_[0] = 0x67452301; sum_[1] = 0xefcdab89; sum_[2] = 0x98badcfe; sum_[3] = 0x10325476;
    sum_[4] = 0xc3d2e1f0; iterations_ = 0; buf_.clear();
  }

  /**
   * Push new binary data into the internal buf_ and recalculate the checksum.
   * @param const void* data
   * @param size_t size
   */
  void update(const void* data, size_t size) {
    if (!data || !size) return;
    const char* p = (const char*) data;
    uint32_t block[16];
    if (!buf_.empty()) { // Deal with the remaining buf_ data
      while (size && buf_.length() < 64) { buf_ += *p++; --size; } // Copy bytes
      if (buf_.length() < 64) return; // Not enough data
      const char* pp = (const char*) buf_.data();
      for (unsigned i = 0; i < 16; ++i) {
        #if (defined (BYTE_ORDER)) && (defined (BIG_ENDIAN)) && ((BYTE_ORDER == BIG_ENDIAN))
        block[i] = (pp[0] << 0) | (pp[1] << 8) | (pp[2] << 16) | (pp[3] << 24);
        #else
        block[i] = (pp[3] << 0) | (pp[2] << 8) | (pp[1] << 16) | (pp[0] << 24);
        #endif
        pp += 4;
      }
      buf_.clear();
      transform(block);
    }
    while (size >= 64) { // Transform full blocks
      for (unsigned i = 0; i < 16; ++i) {
        #if (defined (BYTE_ORDER)) && (defined (BIG_ENDIAN)) && ((BYTE_ORDER == BIG_ENDIAN))
        block[i] = (p[0] << 0) | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
        #else
        block[i] = (p[3] << 0) | (p[2] << 8) | (p[1] << 16) | (p[0] << 24);
        #endif
        p += 4;
      }
      transform(block);
      size -= 64;
    }
    while (size--) {
      buf_ += *p++; // Transfer remaining bytes into the buf_
    }
  }

  /**
   * Finanlise checksum, return hex string.
   * @return str_t
   */
  str_t final() {
    uint64_t total_bits = (iterations_ * 64 + buf_.size()) * 8;
    buf_ += (char) 0x80;
    typename std::string::size_type sz = buf_.size();
    while (buf_.size() < 64) buf_ += (char) 0;
    uint32_t block[16];
    for (unsigned i = 0; i < 16; i++) {
      #if (defined (BYTE_ORDER)) && (defined (BIG_ENDIAN)) && ((BYTE_ORDER == BIG_ENDIAN))
      block[i] = ((buf_[4*i+0] & 0xff) << 0) | ((buf_[4*i+1] & 0xff) << 8) |
         ((buf_[4*i+2] & 0xff) << 16) | ((buf_[4*i+3] & 0xff) << 24);
      #else
      block[i] = ((buf_[4*i+3] & 0xff) << 0) | ((buf_[4*i+2] & 0xff) << 8) |
         ((buf_[4*i+1] & 0xff) << 16) | ((buf_[4*i+0] & 0xff) << 24);
      #endif
    }
    if (sz > 56) {
      transform(block);
      for (unsigned i=0; i<14; ++i) block[i] = 0;
    }
    block[15] = (total_bits >>  0);
    block[14] = (total_bits >> 32);
    transform(block);
    std::basic_stringstream<_CharType> ss; // hex string
    for (unsigned i = 0; i < 5; ++i) { // stream hex includes endian conversion
      ss << std::hex << std::setfill('0') << std::setw(8) << (sum_[i] & 0xffffffff);
    }
    clear();
    return ss.str();
  }

public:

  /**
   * Calculates the SHA1 for a given string.
   * @param const str_t & s
   * @return str_t
   */
  static str_t calculate(const str_t & s)
  { CheckSumSha1 r; r.update(s.data(), s.length()); return r.final(); }

  /**
   * Calculates the SHA1 for a given C-string.
   * @param const char* s
   * @return str_t
   */
  static str_t calculate(const void* data, size_t size)
  { CheckSumSha1 r; r.update(data, size); return r.final(); }

  /**
   * Calculates the SHA1 for a stream. Returns an empty string on error.
   * @param std::istream & is
   * @return str_t
   */
  static str_t calculate(std::istream & is) {
    CheckSumSha1 r;
    char data[64];
    while (is.good() && is.read(data, sizeof(data)).good()) {
      r.update(data, sizeof(data));
    }
    if (!is.eof()) return str_t();
    if (is.gcount()) r.update(data, is.gcount());
    return r.final();
  }

  /**
   * Calculates the SHA1 checksum for a given file, either read binary or as text.
   * @param const str_t & path
   * @param bool binary = true
   * @return str_t
   */
  static str_t file(const str_t & path, bool binary=true) {
    std::ifstream fs;
    fs.open(path.c_str(), binary ? (std::ios::in|std::ios::binary) : (std::ios::in));
    str_t s = calculate(fs);
    fs.close();
    return s;
  }

private:

  /**
   * Performs the SHA1 transformation on a given block
   * @param uint32_t *block
   */
  void transform(uint32_t *block) {
    #define rol(value, bits) (((value) << (bits)) | (((value) & 0xffffffff) >> (32-(bits))))
    #define blk(i) (block[i&15]=rol(block[(i+13)&15]^block[(i+8)&15]^block[(i+2)&15]^block[i&15],1))
    #define R0(v,w,x,y,z,i) z += ((w&(x^y))^y) + block[i] + 0x5a827999 + rol(v,5); w=rol(w,30);
    #define R1(v,w,x,y,z,i) z += ((w&(x^y))^y) + blk(i) + 0x5a827999 + rol(v,5); w=rol(w,30);
    #define R2(v,w,x,y,z,i) z += (w^x^y) + blk(i) + 0x6ed9eba1 + rol(v,5); w=rol(w,30);
    #define R3(v,w,x,y,z,i) z += (((w|x)&y)|(w&x)) + blk(i) + 0x8f1bbcdc + rol(v,5); w=rol(w,30);
    #define R4(v,w,x,y,z,i) z += (w^x^y) + blk(i) + 0xca62c1d6 + rol(v,5); w=rol(w,30);
    uint32_t a = sum_[0], b = sum_[1], c = sum_[2], d = sum_[3], e = sum_[4];
    R0(a,b,c,d,e, 0); R0(e,a,b,c,d, 1); R0(d,e,a,b,c, 2); R0(c,d,e,a,b, 3); R0(b,c,d,e,a, 4);
    R0(a,b,c,d,e, 5); R0(e,a,b,c,d, 6); R0(d,e,a,b,c, 7); R0(c,d,e,a,b, 8); R0(b,c,d,e,a, 9);
    R0(a,b,c,d,e,10); R0(e,a,b,c,d,11); R0(d,e,a,b,c,12); R0(c,d,e,a,b,13); R0(b,c,d,e,a,14);
    R0(a,b,c,d,e,15); R1(e,a,b,c,d,16); R1(d,e,a,b,c,17); R1(c,d,e,a,b,18); R1(b,c,d,e,a,19);
    R2(a,b,c,d,e,20); R2(e,a,b,c,d,21); R2(d,e,a,b,c,22); R2(c,d,e,a,b,23); R2(b,c,d,e,a,24);
    R2(a,b,c,d,e,25); R2(e,a,b,c,d,26); R2(d,e,a,b,c,27); R2(c,d,e,a,b,28); R2(b,c,d,e,a,29);
    R2(a,b,c,d,e,30); R2(e,a,b,c,d,31); R2(d,e,a,b,c,32); R2(c,d,e,a,b,33); R2(b,c,d,e,a,34);
    R2(a,b,c,d,e,35); R2(e,a,b,c,d,36); R2(d,e,a,b,c,37); R2(c,d,e,a,b,38); R2(b,c,d,e,a,39);
    R3(a,b,c,d,e,40); R3(e,a,b,c,d,41); R3(d,e,a,b,c,42); R3(c,d,e,a,b,43); R3(b,c,d,e,a,44);
    R3(a,b,c,d,e,45); R3(e,a,b,c,d,46); R3(d,e,a,b,c,47); R3(c,d,e,a,b,48); R3(b,c,d,e,a,49);
    R3(a,b,c,d,e,50); R3(e,a,b,c,d,51); R3(d,e,a,b,c,52); R3(c,d,e,a,b,53); R3(b,c,d,e,a,54);
    R3(a,b,c,d,e,55); R3(e,a,b,c,d,56); R3(d,e,a,b,c,57); R3(c,d,e,a,b,58); R3(b,c,d,e,a,59);
    R4(a,b,c,d,e,60); R4(e,a,b,c,d,61); R4(d,e,a,b,c,62); R4(c,d,e,a,b,63); R4(b,c,d,e,a,64);
    R4(a,b,c,d,e,65); R4(e,a,b,c,d,66); R4(d,e,a,b,c,67); R4(c,d,e,a,b,68); R4(b,c,d,e,a,69);
    R4(a,b,c,d,e,70); R4(e,a,b,c,d,71); R4(d,e,a,b,c,72); R4(c,d,e,a,b,73); R4(b,c,d,e,a,74);
    R4(a,b,c,d,e,75); R4(e,a,b,c,d,76); R4(d,e,a,b,c,77); R4(c,d,e,a,b,78); R4(b,c,d,e,a,79);
    sum_[0] += a; sum_[1] += b; sum_[2] += c; sum_[3] += d; sum_[4] += e; iterations_++;
    #undef rol
    #undef blk
    #undef R0
    #undef R1
    #undef R2
    #undef R3
    #undef R4
  }

private:
  uint64_t iterations_; // Number of iterations
  uint32_t sum_[5];     // Intermediate checksum digest buffer
  std::string buf_;     // Intermediate buffer for remaining pushed data
};

} // namespace utils
} // namespace noesis

#endif // NOESIS_FRAMEWORK_UTILS_CHECKSUM_HPP_
