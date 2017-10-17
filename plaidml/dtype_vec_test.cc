// Copyright Vertex.AI.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <half.hpp>

#include "base/config/config.h"
#include "base/util/type_url.h"
#include "plaidml/base/base.h"
#include "plaidml/base/context.h"
#include "plaidml/plaidml++.h"
#include "plaidml/plaidml.h"
#include "plaidml/plaidml.pb.h"
#include "testing/matchers.h"
#include "testing/plaidml_config.h"
#include "tile/platform/local_machine/local_machine.pb.h"

using ::testing::NotNull;
using ::testing::IsVaiStatus;

extern "C" void vai_internal_set_vlog(size_t);

namespace half_float {

void PrintTo(const half h, ::std::ostream* os) { *os << static_cast<float>(h); }

}  // namespace half_float

namespace {

namespace plaidml = vertexai::plaidml;

template <class T>
void Fill(char* buffer, std::initializer_list<T> values) {
  T* base = reinterpret_cast<T*>(buffer);
  for (auto value : values) {
    *base++ = value;
  }
}

template <class T>
void Fill16(char* buffer) {
  Fill<T>(buffer,
          {T(1), T(2), T(3), T(4), T(5), T(6), T(7), T(8), T(9), T(10), T(11), T(12), T(13), T(14), T(15), T(16)});
}

template <class T>
void Expect(char* buffer, std::initializer_list<T> values) {
  T* base = reinterpret_cast<T*>(buffer);
  for (auto value : values) {
    EXPECT_EQ(*base++, value);
  }
}

template <>
void Expect<float>(char* buffer, std::initializer_list<float> values) {
  float* base = reinterpret_cast<float*>(buffer);
  for (auto value : values) {
    EXPECT_FLOAT_EQ(*base++, value);
  }
}

template <>
void Expect<double>(char* buffer, std::initializer_list<double> values) {
  double* base = reinterpret_cast<double*>(buffer);
  for (auto value : values) {
    EXPECT_DOUBLE_EQ(*base++, value);
  }
}

template <class T>
void ExpectOutput(char* buffer) {
  Expect<T>(buffer, {T(276), T(336), T(404), T(480)});
}

std::string ConfigWithVectorWidth(std::size_t vec_width) {
  // return vertexai::testing::PlaidMLConfig();
  auto confstr = vertexai::testing::PlaidMLConfig();
  auto config = vertexai::ParseConfig<vertexai::plaidml::proto::Config>(confstr);

  vertexai::tile::local_machine::proto::Platform platform;
  if (!config.platform().UnpackTo(&platform)) {
    return confstr;
  }

  auto vec_config = platform.add_hardware_configs();
  vec_config->mutable_sel()->set_value(true);
  vec_config->mutable_settings()->set_vec_size(vec_width);

  config.mutable_platform()->PackFrom(platform, vertexai::kTypeVertexAIPrefix);
  return vertexai::SerializeConfig(config, true);
}

class DTypeTest : public ::testing::TestWithParam<std::tuple<plaidml_datatype, std::size_t>> {};

TEST_P(DTypeTest, MulSum) {
  plaidml_datatype dtype = std::get<0>(GetParam());
  std::size_t vec_width = std::get<1>(GetParam());

  vai_clear_status();

  // N.B. This is a rather odd convolution, but it does permit vectorization.
  std::unique_ptr<plaidml_function> f{
      plaidml_build_coded_function("function (B[X,Y], C[Y,X]) -> (A) { A[y : Y] = +(C[y, z] * B[z,y]); }", nullptr)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  auto config = ConfigWithVectorWidth(vec_width);
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), config.c_str(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev{
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::size_t elem_size;
  switch (dtype) {
    case PLAIDML_DATA_INT8:
      elem_size = sizeof(std::int8_t);
      break;
    case PLAIDML_DATA_INT16:
      elem_size = sizeof(std::int16_t);
      break;
    case PLAIDML_DATA_INT32:
      elem_size = sizeof(std::int32_t);
      break;
    case PLAIDML_DATA_INT64:
      elem_size = sizeof(std::int64_t);
      break;
    case PLAIDML_DATA_UINT8:
      elem_size = sizeof(std::uint8_t);
      break;
    case PLAIDML_DATA_UINT16:
      elem_size = sizeof(std::uint16_t);
      break;
    case PLAIDML_DATA_UINT32:
      elem_size = sizeof(std::uint32_t);
      break;
    case PLAIDML_DATA_UINT64:
      elem_size = sizeof(std::uint64_t);
      break;
    case PLAIDML_DATA_FLOAT16:
      elem_size = sizeof(half_float::half);
      break;
    case PLAIDML_DATA_FLOAT32:
      elem_size = sizeof(float);
      break;
    case PLAIDML_DATA_FLOAT64:
      elem_size = sizeof(double);
      break;
    default:
      FAIL() << "Unsupported dtype";
  }

  std::unique_ptr<plaidml_buffer> a_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 4 * elem_size)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> in_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 16 * elem_size)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> inmap{plaidml_map_buffer_discard(ctx.get(), in_buf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    char* base = plaidml_get_mapping_base(ctx.get(), inmap.get());
    ASSERT_THAT(base, NotNull());

    switch (dtype) {
      case PLAIDML_DATA_INT8:
        Fill16<std::int8_t>(base);
        break;
      case PLAIDML_DATA_INT16:
        Fill16<std::int16_t>(base);
        break;
      case PLAIDML_DATA_INT32:
        Fill16<std::int32_t>(base);
        break;
      case PLAIDML_DATA_INT64:
        Fill16<std::int64_t>(base);
        break;
      case PLAIDML_DATA_UINT8:
        Fill16<std::uint8_t>(base);
        break;
      case PLAIDML_DATA_UINT16:
        Fill16<std::uint16_t>(base);
        break;
      case PLAIDML_DATA_UINT32:
        Fill16<std::uint32_t>(base);
        break;
      case PLAIDML_DATA_UINT64:
        Fill16<std::uint64_t>(base);
        break;
      case PLAIDML_DATA_FLOAT16:
        Fill16<half_float::half>(base);
        break;
      case PLAIDML_DATA_FLOAT32:
        Fill16<float>(base);
        break;
      case PLAIDML_DATA_FLOAT64:
        Fill16<double>(base);
        break;
      default:
        FAIL() << "Unsupported dtype";
    }

    plaidml_writeback_mapping(ctx.get(), inmap.get());
  }

  std::unique_ptr<plaidml_shape> a_shape{plaidml_alloc_shape(ctx.get(), dtype)};
  plaidml_add_dimension(ctx.get(), a_shape.get(), 4, 1);

  std::unique_ptr<plaidml_shape> b_shape{plaidml_alloc_shape(ctx.get(), dtype)};
  plaidml_add_dimension(ctx.get(), b_shape.get(), 4, 4);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 4, 1);

  std::unique_ptr<plaidml_shape> c_shape{plaidml_alloc_shape(ctx.get(), dtype)};
  plaidml_add_dimension(ctx.get(), c_shape.get(), 4, 1);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 4, 4);

  std::unique_ptr<plaidml_var> a{plaidml_alloc_tensor(ctx.get(), a_buf.get(), a_shape.get())};
  std::unique_ptr<plaidml_var> b{plaidml_alloc_tensor(ctx.get(), in_buf.get(), b_shape.get())};
  std::unique_ptr<plaidml_var> c{plaidml_alloc_tensor(ctx.get(), in_buf.get(), c_shape.get())};

  std::unique_ptr<plaidml_invoker> invoker{plaidml_alloc_invoker(ctx.get(), f.get())};
  plaidml_set_invoker_input(invoker.get(), "B", b.get());
  plaidml_set_invoker_input(invoker.get(), "C", c.get());
  plaidml_set_invoker_output(invoker.get(), "A", a.get());
  std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx.get(), invoker.get())};

  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> outmap{plaidml_map_buffer_current(a_buf.get(), nullptr, nullptr)};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    char* base = plaidml_get_mapping_base(ctx.get(), outmap.get());
    ASSERT_THAT(base, NotNull());

    switch (dtype) {
      case PLAIDML_DATA_INT8:
        ExpectOutput<std::int8_t>(base);
        break;
      case PLAIDML_DATA_INT16:
        ExpectOutput<std::int16_t>(base);
        break;
      case PLAIDML_DATA_INT32:
        ExpectOutput<std::int32_t>(base);
        break;
      case PLAIDML_DATA_INT64:
        ExpectOutput<std::int64_t>(base);
        break;
      case PLAIDML_DATA_UINT8:
        ExpectOutput<std::uint8_t>(base);
        break;
      case PLAIDML_DATA_UINT16:
        ExpectOutput<std::uint16_t>(base);
        break;
      case PLAIDML_DATA_UINT32:
        ExpectOutput<std::uint32_t>(base);
        break;
      case PLAIDML_DATA_UINT64:
        ExpectOutput<std::uint64_t>(base);
        break;
      case PLAIDML_DATA_FLOAT16:
        ExpectOutput<half_float::half>(base);
        break;
      case PLAIDML_DATA_FLOAT32:
        ExpectOutput<float>(base);
        break;
      case PLAIDML_DATA_FLOAT64:
        ExpectOutput<double>(base);
        break;
      default:
        FAIL() << "Unsupported dtype";
    }
  }
}

// TODO: Add a separate test for PLAIDML_DATA_BOOLEAN.
INSTANTIATE_TEST_CASE_P(ShortTypes, DTypeTest,
                        ::testing::Combine(::testing::Values(PLAIDML_DATA_INT8, PLAIDML_DATA_INT16, PLAIDML_DATA_INT32,
                                                             PLAIDML_DATA_UINT8, PLAIDML_DATA_UINT16,
                                                             PLAIDML_DATA_UINT32, PLAIDML_DATA_FLOAT16,
                                                             PLAIDML_DATA_FLOAT32),
                                           ::testing::Values(1, 2, 4)));

// Some of our systems use a sufficiently small memory width setting that four 64-bit values won't fit.  So we test
// those separately, omiting vector width four.
INSTANTIATE_TEST_CASE_P(LongTypes, DTypeTest,
                        ::testing::Combine(::testing::Values(PLAIDML_DATA_INT64, PLAIDML_DATA_UINT64,
                                                             PLAIDML_DATA_FLOAT64),
                                           ::testing::Values(1, 2)));

}  // namespace
