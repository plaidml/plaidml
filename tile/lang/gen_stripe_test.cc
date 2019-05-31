// Copyright 2017-2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/edsl/edsl.h"
#include "testing/matchers.h"
#include "tile/lang/gen_stripe.h"

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace lang {

using namespace plaidml::edsl;  // NOLINT

namespace {

lang::RunInfo Evaluate(const std::string& name, const std::vector<Tensor>& vars) {
  plaidml::edsl::Program program(name, vars);
  return *static_cast<const tile::lang::RunInfo*>(program.runinfo());
}

Tensor ContractPlusElementwise(const Tensor& A, const Tensor& B) {
  TensorDim M, N, K;
  A.bind_dims(M, K);
  B.bind_dims(K, N);
  auto C = TensorOutput(M, N);
  TensorIndex k, m, n;
  C(m, n) += A(m, k) * B(k, n);
  return Call("tanh", C);
}

TEST(GenStripeTest, ContractPlusElementwise) {
  using plaidml::edsl::TensorShape;
  TensorShape shape(PLAIDML_DATA_FLOAT32, {10, 10});
  Tensor A(shape), B(shape);
  auto runinfo = Evaluate("ContractPlusElementwise", {ContractPlusElementwise(A, B)});
  auto program = GenerateStripe(runinfo);
  LOG(INFO) << "Block: " << *program->entry;
  EXPECT_THAT(IntoProto(*program->entry), EqualsProtoText(R"***(
    name: "ContractPlusElementwise"
    loc {}
    refs [
      {
        key: "_X0"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          attrs: { key: "user" value {} }
        }
      },
      {
        key: "_X1"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          attrs: { key: "user" value {} }
        }
      },
      {
        key: "_X2"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          attrs: { key: "tmp" value {} }
        }
      },
      {
        key: "_X3"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          attrs: { key: "user" value {} }
        }
      }
    ]
    stmts [{
      attrs: { key: "main" value {} }
      block {
        name:"main" loc {}
        refs [
          {
            key:"_X0"
            value: {
              from:"_X0" dir:In access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          },
          {
            key:"_X1"
            value: {
              from:"_X1" dir:In access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          },
          {
            key:"_X2"
            value: {
              from:"_X2" dir:InOut access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
              attrs: { key: "tmp" value {} }
            }
          },
          {
            key:"_X3"
            value: {
              from:"_X3" dir:Out access [{}, {}] loc {} agg_op:"assign"
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          }
        ]
        stmts [{
          attrs: { key: "agg_op_add" value {} }
          attrs: { key: "comb_op_mul" value {} }
          attrs: { key: "contraction" value {} }
          attrs: { key: "kernel" value {} }
          block {
            name: "kernel_0(_X0,_X1)" loc {}
            comments: "_X2[x0, x2 : 10, 10] = +(_X0[x0, x1] * _X1[x1, x2])"
            idxs [{name:"x0" range:10 affine {}}, {name:"x1" range:10 affine {}}, {name:"x2", range: 10, affine {}}]
            refs [
              {
                key: "_X0"
                value: {
                  from:"_X0" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x0" value:1}}, {terms:{key:"x1" value:1}}]
                  attrs: { key: "contraction" value {} }
                }
              },
              {
                key: "_X1"
                value: {
                  from:"_X1" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x1" value:1}}, {terms:{key:"x2" value:1}}]
                  attrs: { key: "contraction" value {} }
                }
              },
              {
                key: "_X2"
                value: {
                  from:"_X2" dir:Out loc {} agg_op:"add"
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x0" value:1}}, {terms:{key:"x2" value:1}}]
                }
              }
            ]
            stmts [
              {load {from:"_X0" into:"$_X0"}},
              {load {from:"_X1" into:"$_X1"}},
              {intrinsic {name:"mul" inputs:["$_X0", "$_X1"] outputs:"$_X2" type:FLOAT32}},
              {store {from:"$_X2" into:"_X2"}}
            ]
          }
        }, {
          attrs: { key: "eltwise" value {} }
          attrs: { key: "eltwise_tanh" value {} }
          attrs: { key: "kernel" value {} }
          block {
            name: "kernel_1(_X2)" loc {}
            comments: "_X3 = tanh(_X2)"
            idxs [{name:"i1" range:10 affine {}}, {name:"i2", range:10 affine {}}]
            refs [
              {
                key:"_X2"
                value: {
                  from:"_X2" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
                  attrs: { key: "eltwise_tanh" value {} }
                }
              },
              {
                key:"_X3"
                value: {
                  from:"_X3" dir:Out loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
                }
              }
            ]
            stmts [
              {load {from:"_X2" into:"$_X2"}},
              {intrinsic {name:"tanh" inputs:"$_X2" outputs:"$_X3" type:FLOAT32}},
              {store {from:"$_X3" into:"_X3"}}
            ]
          }
        }]
      }
    }]
  )***"));
}

}  // namespace
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
