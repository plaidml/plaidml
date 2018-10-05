// Copyright 2018 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/base/base.h"
#include "plaidml/base/context.h"
#include "plaidml/plaidml++.h"
#include "plaidml/plaidml.h"
#include "testing/matchers.h"
#include "testing/plaidml_config.h"

using ::testing::Gt;
using ::testing::IsVaiStatus;
using ::testing::Not;
using ::testing::NotNull;

extern "C" void vai_internal_set_vlog(size_t);

namespace {

namespace plaidml = vertexai::plaidml;

TEST(PlaidML_C_API, VersionDefined) {
  EXPECT_THAT(plaidml_get_version(), NotNull());
  EXPECT_THAT(strlen(plaidml_get_version()), Gt(4));
}

TEST(PlaidML_C_API, BroadcastFailure) {
  vai_clear_status();

  std::unique_ptr<plaidml_function> add{plaidml_build_coded_function("function (A, B) -> (C) { C = A + B; }", nullptr)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), vertexai::testing::PlaidMLConfig(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev{
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> a_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 2 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> b_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> c_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 18 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_shape> a_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), a_shape.get(), 1, 0);
  plaidml_add_dimension(ctx.get(), a_shape.get(), 2, 1);

  std::unique_ptr<plaidml_shape> b_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 3);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 1, 0);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 1);

  std::unique_ptr<plaidml_shape> c_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 6);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 2, 3);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 1);

  std::unique_ptr<plaidml_var> a{plaidml_alloc_tensor(ctx.get(), a_buf.get(), a_shape.get())};
  std::unique_ptr<plaidml_var> b{plaidml_alloc_tensor(ctx.get(), b_buf.get(), b_shape.get())};
  std::unique_ptr<plaidml_var> c{plaidml_alloc_tensor(ctx.get(), c_buf.get(), c_shape.get())};

  std::unique_ptr<plaidml_invoker> invoker{plaidml_alloc_invoker(ctx.get(), add.get())};
  plaidml_set_invoker_input(invoker.get(), "A", a.get());
  plaidml_set_invoker_input(invoker.get(), "B", b.get());
  plaidml_set_invoker_output(invoker.get(), "C", c.get());
  std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx.get(), invoker.get())};

  EXPECT_THAT(vai_last_status(), Not(IsVaiStatus(VAI_STATUS_OK)));
}

TEST(PlaidML_C_API, BroadcastOne) {
  vai_clear_status();

  std::unique_ptr<plaidml_function> add{plaidml_build_coded_function("function (A, B) -> (C) { C = A + B; }", nullptr)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), vertexai::testing::PlaidMLConfig(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev{
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> a_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 3 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> b_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> c_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> a_map{plaidml_map_buffer_discard(ctx.get(), a_buf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), a_map.get()));
    ASSERT_THAT(base, NotNull());

    base[0] = 10.0;
    base[1] = 20.0;
    base[2] = 30.0;

    plaidml_writeback_mapping(ctx.get(), a_map.get());
  }

  {
    std::unique_ptr<plaidml_mapping> b_map{plaidml_map_buffer_discard(ctx.get(), b_buf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), b_map.get()));
    ASSERT_THAT(base, NotNull());

    base[0] = 1.0;
    base[1] = 2.0;
    base[2] = 3.0;
    base[3] = 4.0;
    base[4] = 5.0;
    base[5] = 6.0;
    base[6] = 7.0;
    base[7] = 8.0;
    base[8] = 9.0;

    plaidml_writeback_mapping(ctx.get(), b_map.get());
  }

  std::unique_ptr<plaidml_shape> a_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), a_shape.get(), 3, 1);
  plaidml_add_dimension(ctx.get(), a_shape.get(), 1, 0);

  std::unique_ptr<plaidml_shape> b_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 3);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 1);

  std::unique_ptr<plaidml_shape> c_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 3);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 1);

  std::unique_ptr<plaidml_var> a{plaidml_alloc_tensor(ctx.get(), a_buf.get(), a_shape.get())};
  std::unique_ptr<plaidml_var> b{plaidml_alloc_tensor(ctx.get(), b_buf.get(), b_shape.get())};
  std::unique_ptr<plaidml_var> c{plaidml_alloc_tensor(ctx.get(), c_buf.get(), c_shape.get())};

  std::unique_ptr<plaidml_invoker> invoker{plaidml_alloc_invoker(ctx.get(), add.get())};
  plaidml_set_invoker_input(invoker.get(), "A", a.get());
  plaidml_set_invoker_input(invoker.get(), "B", b.get());
  plaidml_set_invoker_output(invoker.get(), "C", c.get());
  std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx.get(), invoker.get())};

  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> c_map{plaidml_map_buffer_current(c_buf.get(), nullptr, nullptr)};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), c_map.get()));
    ASSERT_THAT(base, NotNull());

    EXPECT_FLOAT_EQ(base[0], 11.0);
    EXPECT_FLOAT_EQ(base[1], 12.0);
    EXPECT_FLOAT_EQ(base[2], 13.0);
    EXPECT_FLOAT_EQ(base[3], 24.0);
    EXPECT_FLOAT_EQ(base[4], 25.0);
    EXPECT_FLOAT_EQ(base[5], 26.0);
    EXPECT_FLOAT_EQ(base[6], 37.0);
    EXPECT_FLOAT_EQ(base[7], 38.0);
    EXPECT_FLOAT_EQ(base[8], 39.0);
  }
}

TEST(PlaidML_C_API, BroadcastBoth) {
  vai_clear_status();

  std::unique_ptr<plaidml_function> add{plaidml_build_coded_function("function (A, B) -> (C) { C = A + B; }", nullptr)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), vertexai::testing::PlaidMLConfig(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev{
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> a_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 2 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> b_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> c_buf{plaidml_alloc_buffer(ctx.get(), dev.get(), 18 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> a_map{plaidml_map_buffer_discard(ctx.get(), a_buf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), a_map.get()));
    ASSERT_THAT(base, NotNull());

    base[0] = 10.0;
    base[1] = 20.0;

    plaidml_writeback_mapping(ctx.get(), a_map.get());
  }

  {
    std::unique_ptr<plaidml_mapping> b_map{plaidml_map_buffer_discard(ctx.get(), b_buf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), b_map.get()));
    ASSERT_THAT(base, NotNull());

    base[0] = 1.0;
    base[1] = 2.0;
    base[2] = 3.0;
    base[3] = 4.0;
    base[4] = 5.0;
    base[5] = 6.0;
    base[6] = 7.0;
    base[7] = 8.0;
    base[8] = 9.0;

    plaidml_writeback_mapping(ctx.get(), b_map.get());
  }

  std::unique_ptr<plaidml_shape> a_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), a_shape.get(), 2, 1);
  plaidml_add_dimension(ctx.get(), a_shape.get(), 1, 0);

  std::unique_ptr<plaidml_shape> b_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 3);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 1, 0);
  plaidml_add_dimension(ctx.get(), b_shape.get(), 3, 1);

  std::unique_ptr<plaidml_shape> c_shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 6);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 2, 3);
  plaidml_add_dimension(ctx.get(), c_shape.get(), 3, 1);

  std::unique_ptr<plaidml_var> a{plaidml_alloc_tensor(ctx.get(), a_buf.get(), a_shape.get())};
  std::unique_ptr<plaidml_var> b{plaidml_alloc_tensor(ctx.get(), b_buf.get(), b_shape.get())};
  std::unique_ptr<plaidml_var> c{plaidml_alloc_tensor(ctx.get(), c_buf.get(), c_shape.get())};

  std::unique_ptr<plaidml_invoker> invoker{plaidml_alloc_invoker(ctx.get(), add.get())};
  plaidml_set_invoker_input(invoker.get(), "A", a.get());
  plaidml_set_invoker_input(invoker.get(), "B", b.get());
  plaidml_set_invoker_output(invoker.get(), "C", c.get());
  std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx.get(), invoker.get())};

  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> c_map{plaidml_map_buffer_current(c_buf.get(), nullptr, nullptr)};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), c_map.get()));
    ASSERT_THAT(base, NotNull());

    EXPECT_FLOAT_EQ(base[0], 11.0);
    EXPECT_FLOAT_EQ(base[1], 12.0);
    EXPECT_FLOAT_EQ(base[2], 13.0);
    EXPECT_FLOAT_EQ(base[3], 21.0);
    EXPECT_FLOAT_EQ(base[4], 22.0);
    EXPECT_FLOAT_EQ(base[5], 23.0);
    EXPECT_FLOAT_EQ(base[6], 14.0);
    EXPECT_FLOAT_EQ(base[7], 15.0);
    EXPECT_FLOAT_EQ(base[8], 16.0);
    EXPECT_FLOAT_EQ(base[9], 24.0);
    EXPECT_FLOAT_EQ(base[10], 25.0);
    EXPECT_FLOAT_EQ(base[11], 26.0);
    EXPECT_FLOAT_EQ(base[12], 17.0);
    EXPECT_FLOAT_EQ(base[13], 18.0);
    EXPECT_FLOAT_EQ(base[14], 19.0);
    EXPECT_FLOAT_EQ(base[15], 27.0);
    EXPECT_FLOAT_EQ(base[16], 28.0);
    EXPECT_FLOAT_EQ(base[17], 29.0);
  }
}

TEST(PlaidML_C_API, MatMul) {
  vai_clear_status();

  std::unique_ptr<plaidml_function> matmul{
      plaidml_build_coded_function("function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }", nullptr)};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<vai_ctx> ctx{vai_alloc_ctx()};
  std::unique_ptr<plaidml_device_enumerator> dev_enum{
      plaidml_alloc_device_enumerator_with_config(ctx.get(), vertexai::testing::PlaidMLConfig(), nullptr, nullptr)};
  std::unique_ptr<plaidml_device> dev{
      plaidml_open_device(ctx.get(), plaidml_get_devconf(ctx.get(), dev_enum.get(), 0))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> inbuf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  std::unique_ptr<plaidml_buffer> outbuf{plaidml_alloc_buffer(ctx.get(), dev.get(), 9 * sizeof(float))};
  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> inmap{plaidml_map_buffer_discard(ctx.get(), inbuf.get())};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), inmap.get()));
    ASSERT_THAT(base, NotNull());

    base[0] = 1.0;
    base[1] = 2.0;
    base[2] = 3.0;
    base[3] = 4.0;
    base[4] = 5.0;
    base[5] = 6.0;
    base[6] = 7.0;
    base[7] = 8.0;
    base[8] = 9.0;

    plaidml_writeback_mapping(ctx.get(), inmap.get());
  }

  std::unique_ptr<plaidml_shape> shape{plaidml_alloc_shape(ctx.get(), PLAIDML_DATA_FLOAT32)};
  plaidml_add_dimension(ctx.get(), shape.get(), 3, 3);
  plaidml_add_dimension(ctx.get(), shape.get(), 3, 1);

  std::unique_ptr<plaidml_var> a{plaidml_alloc_tensor(ctx.get(), outbuf.get(), shape.get())};
  std::unique_ptr<plaidml_var> b{plaidml_alloc_tensor(ctx.get(), inbuf.get(), shape.get())};
  std::unique_ptr<plaidml_var> c{plaidml_alloc_tensor(ctx.get(), inbuf.get(), shape.get())};

  std::unique_ptr<plaidml_invoker> invoker{plaidml_alloc_invoker(ctx.get(), matmul.get())};
  plaidml_set_invoker_input(invoker.get(), "B", b.get());
  plaidml_set_invoker_input(invoker.get(), "C", c.get());
  plaidml_set_invoker_output(invoker.get(), "A", a.get());
  std::unique_ptr<plaidml_invocation> invocation{plaidml_schedule_invocation(ctx.get(), invoker.get())};

  EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));

  {
    std::unique_ptr<plaidml_mapping> outmap{plaidml_map_buffer_current(outbuf.get(), nullptr, nullptr)};
    EXPECT_THAT(vai_last_status(), IsVaiStatus(VAI_STATUS_OK));
    float* base = reinterpret_cast<float*>(plaidml_get_mapping_base(ctx.get(), outmap.get()));
    ASSERT_THAT(base, NotNull());

    EXPECT_FLOAT_EQ(base[0], 1.0 + 8.0 + 21.0);
    EXPECT_FLOAT_EQ(base[1], 2.0 + 10.0 + 24.0);
    EXPECT_FLOAT_EQ(base[2], 3.0 + 12.0 + 27.0);
    EXPECT_FLOAT_EQ(base[3], 4.0 + 20.0 + 42.0);
    EXPECT_FLOAT_EQ(base[4], 8.0 + 25.0 + 48.0);
    EXPECT_FLOAT_EQ(base[5], 12.0 + 30.0 + 54.0);
    EXPECT_FLOAT_EQ(base[6], 7.0 + 32.0 + 63.0);
    EXPECT_FLOAT_EQ(base[7], 14.0 + 40.0 + 72.0);
    EXPECT_FLOAT_EQ(base[8], 21.0 + 48.0 + 81.0);
  }
}

TEST(PlaidML_C_API, Save) {
  vai_clear_status();
  auto ctx = std::make_shared<vertexai::ctx>();
  auto devices = plaidml::enumerate_devices(ctx, vertexai::testing::PlaidMLConfig());
  plaidml::device dev = devices[0].open();
  plaidml::function matmul("function (B[X,Z], C[Z,Y]) -> (A) { A[x,y : X,Y] = +(B[x,z] * C[z,y]); }");

  // Setup a tensor of data
  plaidml::tensor<float> fixed = dev.allocate(plaidml::shape<float>(ctx, {1000, 1000}));
  {
    plaidml::mapping<float> data = fixed.map(plaidml::map_for_write);
    for (size_t i = 0; i < 1000; i++) {
      for (size_t j = 0; j < 1000; j++) {
        data(i, j) = 7;
      }
    }
  }

  // Compose a new function
  plaidml::placeholder var(2);
  plaidml::variable out = matmul(fixed, var);
  plaidml::function fixed_mul = plaidml::compose().input("C", var).output("A", out);

  // Save the composed function
  fixed_mul.save("test.plaidml");

  // Reload it, bind, and execute
  plaidml::function fixed_mul_2;
  plaidml::tensor<float> output = dev.allocate(plaidml::shape<float>(ctx, {1000, 1000}));
  fixed_mul_2.load(ctx, dev, "test.plaidml");

  plaidml::invoker(ctx, fixed_mul_2).set_input("C", fixed).set_output("A", output).invoke();

  {
    plaidml::mapping<float> data = output.map(plaidml::map_for_read);
    EXPECT_FLOAT_EQ(data(0u, 0u), 49000);
    EXPECT_FLOAT_EQ(data(9u, 9u), 49000);
  }
}

}  // namespace
