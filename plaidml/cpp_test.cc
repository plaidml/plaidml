// Copyright 2018 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "plaidml/base/context.h"
#include "plaidml/plaidml++.h"
#include "testing/plaidml_config.h"

using ::testing::Ne;

namespace {

using namespace vertexai::plaidml;  // NOLINT

TEST(PlaidML_CPP_API, Composition) {
  vai_clear_status();
  auto ctx = std::make_shared<vertexai::ctx>();
  std::vector<device_config> configs = enumerate_devices(ctx, vertexai::testing::PlaidMLConfig());

  for (size_t i = 0; i < configs.size(); i++) {
    std::cout << i << ": " << configs[i].id() << "->" << configs[i].description() << std::endl;
  }

  ASSERT_THAT(configs.size(), Ne(0));

  device dev = configs[0].open();

  function weight_mul("function (A[I, K], B[J, K]) -> (C) { C[i,j : I, J] = +(A[i, k] * B[j, k]); }");
  function add_bias("function (A[N, I], B[I]) -> (C) { C[n, i : N, I] = +(A[n, i] + B[i]); }");
  function relu("function (A) -> (R) { R = (A > 0 ? A : 0); }");
  function sq_err("function (A, B) -> (R) { D = A - B; S = D * D; R[] = +(S[i, j]); }");

  placeholder input(2);

  tensor<float> weights = dev.allocate(shape<float>(ctx, {50, 100}));
  tensor<float> biases = dev.allocate(shape<float>(ctx, {50}));

  variable output = relu(add_bias(weight_mul(input, weights), biases));

  placeholder goal(2);
  variable err = sq_err(output, goal);

  function forward = compose("forward")  //
                         .input("X", input)
                         .output("Y", output);

  function error = compose("error")  //
                       .input("X", input)
                       .input("Yt", goal)
                       .output("E", err);

  gradient derr(err);
  placeholder alpha(0);

  function add_update("function (X, Alpha, DX) -> (R) { R = X + Alpha * DX; }");

  function learn = compose("learn")
                       .input("X", input)
                       .input("Yt", goal)
                       .input("Alpha", alpha)
                       .output("E", err)
                       .update(biases, add_update(biases, alpha, derr(biases)))
                       .update(weights, add_update(weights, alpha, derr(weights)));

  tensor<float> X = dev.allocate(shape<float>(ctx, {64, 100}));
  tensor<float> Y = dev.allocate(shape<float>(ctx, {64, 50}));
  tensor<float> E = dev.allocate(shape<float>(ctx));

  invoker(ctx, learn)  //
      .set_input("X", X)
      .set_input("Yt", Y)
      .set_input("Alpha", .001)
      .set_output("E", E)
      .invoke();

  {
    auto _E = E.map(map_for_read);
    std::cout << _E() << std::endl;
  }
}

TEST(PlaidML_CPP_API, MultiDep) {
  const std::size_t N0 = 100;
  const std::size_t N1 = 6;
  const std::size_t N2 = 100;

  vai_clear_status();
  auto ctx = std::make_shared<vertexai::ctx>();

  auto devices = enumerate_devices(ctx, vertexai::testing::PlaidMLConfig());
  device dev = devices[0].open();
  function multidep(R"(
    function (I0[N0, A0, N2], I1[N0, A1, N2], I2[N0, A2, N2]) -> (O) {
      T0[n0, a, n2: N0, 6, N2] = +(I0[n0, a, n2]);
      T1[n0, a+2, n2: N0, 6, N2] = +(I1[n0, a, n2]);
      T2[n0, a+3, n2: N0, 6, N2] = +(I2[n0, a, n2]);
      O = T0 + T1 + T2;
    }
  )");

  tensor<float> in = dev.allocate(shape<float>(ctx, {N0, N1, N2}));
  {
    mapping<float> view = in.map(map_for_write);
    for (size_t i = 0; i < N0; i++) {
      for (size_t j = 0; j < N1; j++) {
        for (size_t k = 0; k < N2; k++) {
          view(i, j, k) = 7;
        }
      }
    }
  }

  tensor<float> out = dev.allocate(shape<float>(ctx, {N0, N1, N2}));

  invoker(ctx, multidep)  //
      .set_input("I0", in)
      .set_input("I1", in)
      .set_input("I2", in)
      .set_output("O", out)
      .invoke();

  {
    mapping<float> view = out.map(map_for_read);
    for (size_t i = 0; i < N0; i++) {
      for (size_t j = 0; j < N1; j++) {
        for (size_t k = 0; k < N2; k++) {
          EXPECT_THAT(view(i, j, k), Ne(0));
        }
      }
    }
  }
}

}  // namespace
