#include "tile/codegen/test/passes/manager.h"

#include <random>

#include "base/util/error.h"
#include "base/util/logging.h"
#include "tile/base/buffer.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {
namespace passes {

namespace {

std::random_device g_rand_device;
std::minstd_rand g_rand_engine{g_rand_device()};

// Writes the bytes of a single random value of type T to dst
template <class T>
T random_value() {
  throw error::Unimplemented("Unsupported type for random_value");
}

template <>
bool random_value<bool>() {
  static std::bernoulli_distribution b_dist;
  bool val = b_dist(g_rand_engine);
  IVLOG(5, "Making a random bool: " << val);
  return val;
}

template <>
int8_t random_value<int8_t>() {
  static std::uniform_int_distribution<short> i8_dist{-100, 100};
  int8_t val = static_cast<int8_t>(i8_dist(g_rand_engine));
  IVLOG(5, "Making a random i8: " << (int)val);
  return val;
}

template <>
int16_t random_value<int16_t>() {
  static std::uniform_int_distribution<int16_t> i16_dist{-30000, 30000};
  int16_t val = i16_dist(g_rand_engine);
  IVLOG(5, "Making a random i16: " << val);
  return val;
}

template <>
int32_t random_value<int32_t>() {
  static std::uniform_int_distribution<int32_t> i32_dist{-1000000, 1000000};
  int32_t val = i32_dist(g_rand_engine);
  IVLOG(5, "Making a random i32: " << val);
  return val;
}

template <>
int64_t random_value<int64_t>() {
  static std::uniform_int_distribution<int64_t> i64_dist{-10000000000, 10000000000};
  int64_t val = i64_dist(g_rand_engine);
  IVLOG(5, "Making a random i64: " << val);
  return val;
}

template <>
uint8_t random_value<uint8_t>() {
  static std::uniform_int_distribution<unsigned short> u8_dist{0, 200};
  uint8_t val = static_cast<uint8_t>(u8_dist(g_rand_engine));
  IVLOG(5, "Making a random u8: " << val);
  return val;
}

template <>
uint16_t random_value<uint16_t>() {
  static std::uniform_int_distribution<uint16_t> u16_dist{0, 60000};
  uint16_t val = u16_dist(g_rand_engine);
  IVLOG(5, "Making a random u16: " << val);
  return val;
}

template <>
uint32_t random_value<uint32_t>() {
  static std::uniform_int_distribution<uint32_t> u32_dist{0, 1000000};
  uint32_t val = u32_dist(g_rand_engine);
  IVLOG(5, "Making a random u32: " << val);
  return val;
}

template <>
uint64_t random_value<uint64_t>() {
  static std::uniform_int_distribution<uint64_t> u64_dist{0, 10000000000};
  uint64_t val = u64_dist(g_rand_engine);
  IVLOG(5, "Making a random u64: " << val);
  return val;
}

template <>
float random_value<float>() {
  // Two distributions to simulate mix of small & large magnitude values in networks
  static std::normal_distribution<float> small_f32_dist{0., 1e-4};
  static std::normal_distribution<float> large_f32_dist{0., 100};
  static std::bernoulli_distribution b_dist;
  float val = b_dist(g_rand_engine) ? small_f32_dist(g_rand_engine) : large_f32_dist(g_rand_engine);
  IVLOG(5, "Making a random f32: " << val);
  return val;
}

template <>
double random_value<double>() {
  // Two distributions to simulate mix of small & large magnitude values in networks
  static std::normal_distribution<double> small_f64_dist{0., 1e-4};
  static std::normal_distribution<double> large_f64_dist{0., 100};
  static std::bernoulli_distribution b_dist;
  double val = b_dist(g_rand_engine) ? small_f64_dist(g_rand_engine) : large_f64_dist(g_rand_engine);
  IVLOG(5, "Making a random f64: " << val);
  return val;
}

struct Comparator {
  double irtol;
  int iatol;
  double frtol;
  double fatol;

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, bool>::type  //
  cmp_values(size_t elem_size, const char* a_buf, const char* b_buf) {
    auto a = reinterpret_cast<const T*>(a_buf);
    auto b = reinterpret_cast<const T*>(b_buf);
    for (size_t i = 0; i < elem_size; i++) {
      if (std::abs(a[i] - b[i]) > frtol * std::abs(b[i]) + fatol) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, bool>::type  //
  cmp_values(size_t elem_size, const char* a_buf, const char* b_buf) {
    auto a = reinterpret_cast<const T*>(a_buf);
    auto b = reinterpret_cast<const T*>(b_buf);
    for (size_t i = 0; i < elem_size; i++) {
      if (std::abs(a[i] - b[i]) > irtol * std::abs(b[i]) + iatol) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value, bool>::type  //
  cmp_values(size_t elem_size, const char* a_buf, const char* b_buf) {
    auto a = reinterpret_cast<const T*>(a_buf);
    auto b = reinterpret_cast<const T*>(b_buf);
    for (size_t i = 0; i < elem_size; i++) {
      if (std::max(a[i], b[i]) - std::min(a[i], b[i]) > irtol * b[i] + iatol) {
        return false;
      }
    }
    return true;
  }

  bool cmp_values_bool(size_t elem_size, const char* a_buf, const char* b_buf) {
    auto a = reinterpret_cast<const bool*>(a_buf);
    auto b = reinterpret_cast<const bool*>(b_buf);
    for (size_t i = 0; i < elem_size; i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace

BufferManager::Entry::Entry(const Entry& rhs)
    : dtype(rhs.dtype),  //
      elem_size(rhs.elem_size),
      buffer(rhs.buffer->Clone()) {}

BufferManager::BufferManager(const BufferManager& src) : map_(src.map_) {}

std::map<std::string, void*> BufferManager::map_buffers() {
  context::Context ctx;
  std::map<std::string, void*> ret;
  for (const auto& kvp : map_) {
    auto view = kvp.second.buffer->MapCurrent(ctx).get();
    ret[kvp.first] = view->data();
  }
  return ret;
}

void BufferManager::add_random(const std::string& name, DataType dtype, uint64_t elem_size) {
  IVLOG(3, "Building buffer " << name << " with " << elem_size << " random values");
  if (map_.find(name) != map_.end()) {
    throw error::AlreadyExists("Attempted to add buffer that already exists to BufferManager");
  }
  uint64_t byte_size = elem_size * byte_width(dtype);
  if (byte_size == 0) {
    throw error::Unimplemented(
        "Attempt to initialize data for unsupported type");  // TODO: Unclear that we need to error here for all cases
  }
  Entry entry;
  entry.dtype = dtype;
  entry.elem_size = elem_size;
  entry.buffer = std::make_shared<SimpleBuffer>(byte_size);
  context::Context ctx;
  auto view = entry.buffer->MapDiscard(ctx);
  map_.emplace(name, std::move(entry));
  for (size_t i = 0; i < elem_size; ++i) {
    switch (dtype) {
      case DataType::BOOLEAN:
        reinterpret_cast<bool*>(view->data())[i] = random_value<bool>();
        break;
      case DataType::INT8:
        reinterpret_cast<int8_t*>(view->data())[i] = random_value<int8_t>();
        break;
      case DataType::INT16:
        reinterpret_cast<int16_t*>(view->data())[i] = random_value<int16_t>();
        break;
      case DataType::INT32:
        reinterpret_cast<int32_t*>(view->data())[i] = random_value<int32_t>();
        break;
      case DataType::INT64:
        reinterpret_cast<int64_t*>(view->data())[i] = random_value<int64_t>();
        break;
      case DataType::UINT8:
        reinterpret_cast<uint8_t*>(view->data())[i] = random_value<uint8_t>();
        break;
      case DataType::UINT16:
        reinterpret_cast<uint16_t*>(view->data())[i] = random_value<uint16_t>();
        break;
      case DataType::UINT32:
        reinterpret_cast<uint32_t*>(view->data())[i] = random_value<uint32_t>();
        break;
      case DataType::UINT64:
        reinterpret_cast<uint64_t*>(view->data())[i] = random_value<uint64_t>();
        break;
      case DataType::FLOAT32:
        reinterpret_cast<float*>(view->data())[i] = random_value<float>();
        break;
      case DataType::FLOAT64:
        reinterpret_cast<double*>(view->data())[i] = random_value<double>();
        break;
      case DataType::INVALID:
        throw error::InvalidArgument("Can't construct value of 'INVALID' type");
      case DataType::PRNG:
        throw error::InvalidArgument("Can't construct value of 'PRNG' type");
      case DataType::INT128:
        throw error::Unimplemented("Can't yet construct value of 'INT128' type");
      case DataType::FLOAT16:
        throw error::Unimplemented("Can't yet construct value of 'FLOAT16' type");
      default:
        throw error::InvalidArgument("Unrecognized data type when generating random values");
    }
  }
  return;
}

bool BufferManager::is_close(const BufferManager& rhs, double frtol, double fatol, double irtol, int iatol) const {
  context::Context ctx;
  IVLOG(5, "Starting comparison of buffers");
  if (map_.size() != rhs.map_.size()) {
    IVLOG(2, "Unequal numbers of buffers (lhs: " << map_.size() << "rhs: " << rhs.map_.size() << ")");
    return false;
  }
  Comparator cmp{irtol, iatol, frtol, fatol};
  for (const auto& kvp : map_) {
    IVLOG(2, "Comparing buffer " << kvp.first);
    const Entry* lhs_entry;
    const Entry* rhs_entry;
    try {
      std::tie(lhs_entry, rhs_entry) = setup_buffer_cmp(rhs, kvp.first);
    } catch (const error::InvalidArgument& e) {
      IVLOG(2, "Buffers unequal: " << e.what());
      return false;
    }
    auto lhs_view = lhs_entry->buffer->MapCurrent(ctx).get();
    auto rhs_view = rhs_entry->buffer->MapCurrent(ctx).get();
    bool ret = false;
    switch (lhs_entry->dtype) {
      case DataType::BOOLEAN:
        ret = cmp.cmp_values_bool(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::INT8:
        ret = cmp.cmp_values<int8_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::INT16:
        ret = cmp.cmp_values<int16_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::INT32:
        ret = cmp.cmp_values<int32_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::INT64:
        ret = cmp.cmp_values<int64_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::UINT8:
        ret = cmp.cmp_values<uint8_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::UINT16:
        ret = cmp.cmp_values<uint16_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::UINT32:
        ret = cmp.cmp_values<uint32_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::UINT64:
        ret = cmp.cmp_values<uint64_t>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::FLOAT32:
        ret = cmp.cmp_values<float>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      case DataType::FLOAT64:
        ret = cmp.cmp_values<double>(lhs_entry->elem_size, lhs_view->data(), rhs_view->data());
        break;
      default:
        throw error::InvalidArgument("Unrecognized data type");
    }
    if (!ret) {
      return false;
    }
  }
  return true;
}

std::tuple<const BufferManager::Entry*, const BufferManager::Entry*>  //
BufferManager::setup_buffer_cmp(const BufferManager& cmp, const std::string& name) const {
  auto it_cur = map_.find(name);
  if (it_cur == map_.end()) {
    throw error::InvalidArgument("Buffer not found in cur");
  }
  auto it_cmp = cmp.map_.find(name);
  if (it_cmp == map_.end()) {
    throw error::InvalidArgument("Buffer not found in cmp");
  }
  if (it_cur->second.dtype != it_cmp->second.dtype) {
    throw error::InvalidArgument("Buffer dtype mismatch");
  }
  if (it_cur->second.elem_size != it_cmp->second.elem_size) {
    throw error::InvalidArgument("Buffer elem_size mismatch");
  }
  const Entry* lhs = &it_cur->second;
  const Entry* rhs = &it_cmp->second;
  return std::make_tuple(lhs, rhs);
}

}  // namespace passes
}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
