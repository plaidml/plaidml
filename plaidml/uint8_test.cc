// Copyright 2018 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "plaidml/base/base.h"
#include "plaidml/plaidml.h"
#include "testing/matchers.h"
#include "testing/plaidml_config.h"

using ::testing::IsVaiStatus;
using ::testing::NotNull;

namespace {

TEST(PlaidMLTest, Uints) {
  std::unique_ptr<plaidml_function> dotprod{
      plaidml_build_coded_function("function (A[N], B[N]) -> (C) { C[n : N] = +(A[n] * B[n]); }", NULL)};

  std::unique_ptr<vai_ctx> ctx(vai_alloc_ctx());
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), vertexai::testing::PlaidMLConfig(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev(
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0)));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  // Three-element vector of uint8
  std::unique_ptr<plaidml_shape> shape_a(plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_INT32));
  plaidml_add_dimension(ctx.get(), shape_a.get(), 3, 1);
  std::unique_ptr<plaidml_buffer> input_a(plaidml_alloc_buffer(ctx.get(), dev.get(), 3 * sizeof(int32_t)));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  {
    std::unique_ptr<plaidml_mapping> map_a{plaidml_map_buffer_discard(ctx.get(), input_a.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    int32_t* buf_a = reinterpret_cast<int32_t*>(plaidml_get_mapping_base(ctx.get(), map_a.get()));
    ASSERT_THAT(buf_a, NotNull());
    buf_a[0] = 1;
    buf_a[1] = 2;
    buf_a[2] = 3;
    plaidml_writeback_mapping(ctx.get(), map_a.get());
  }
  std::unique_ptr<plaidml_var> val_a(plaidml_alloc_tensor(ctx.get(), input_a.get(), shape_a.get()));

  // Three-element vector of uint8 which would go negative if cast to int8
  std::unique_ptr<plaidml_shape> shape_b(plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_UINT8));
  plaidml_add_dimension(ctx.get(), shape_b.get(), 3, 1);
  std::unique_ptr<plaidml_buffer> input_b(plaidml_alloc_buffer(ctx.get(), dev.get(), 3 * sizeof(uint8_t)));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  {
    std::unique_ptr<plaidml_mapping> map_b{plaidml_map_buffer_discard(ctx.get(), input_b.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    uint8_t* buf_b = reinterpret_cast<uint8_t*>(plaidml_get_mapping_base(ctx.get(), map_b.get()));
    ASSERT_THAT(buf_b, NotNull());
    buf_b[0] = 0xC8;  // 200 if inteprreted as uint8, -56 if cast to int8
    buf_b[1] = 0xC8;
    buf_b[2] = 0xC8;
    plaidml_writeback_mapping(ctx.get(), map_b.get());
  }
  std::unique_ptr<plaidml_var> val_b(plaidml_alloc_tensor(ctx.get(), input_b.get(), shape_b.get()));

  // Output: multiply each element of A by the corresponding element of B
  std::unique_ptr<plaidml_shape> shape_c(plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_INT32));
  plaidml_add_dimension(ctx.get(), shape_c.get(), 3, 1);
  std::unique_ptr<plaidml_buffer> output_c(plaidml_alloc_buffer(ctx.get(), dev.get(), 3 * sizeof(int32_t)));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  std::unique_ptr<plaidml_var> val_c(plaidml_alloc_tensor(ctx.get(), output_c.get(), shape_c.get()));

  std::unique_ptr<plaidml_invoker> invoker(plaidml_alloc_invoker(ctx.get(), dotprod.get()));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  plaidml_set_invoker_input(invoker.get(), "A", val_a.get());
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  plaidml_set_invoker_input(invoker.get(), "B", val_b.get());
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
  plaidml_set_invoker_output(invoker.get(), "C", val_c.get());
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_invocation> invocation(plaidml_schedule_invocation(ctx.get(), invoker.get()));
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> map_c{plaidml_map_buffer_current(output_c.get(), nullptr, nullptr)};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    int32_t* buf_c = reinterpret_cast<int32_t*>(plaidml_get_mapping_base(ctx.get(), map_c.get()));
    ASSERT_THAT(buf_c, NotNull());
    EXPECT_EQ(buf_c[0], 1 * 200);  // fail == -56
    EXPECT_EQ(buf_c[1], 2 * 200);  // fail == -112
    EXPECT_EQ(buf_c[2], 3 * 200);  // fail == -168
  }
}

}  // namespace
