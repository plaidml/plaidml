// Copyright 2017-2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <utility>

#include "testing/matchers.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/parser.h"

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace lang {
namespace {

class GenStripeTest : public ::testing::Test {
 protected:
  std::shared_ptr<stripe::Block> ToStripe(
      const char* tile_code, std::initializer_list<std::pair<std::string, std::shared_ptr<Value>>> inputs,
      std::initializer_list<std::pair<std::string, std::shared_ptr<TensorValue>>> outputs) {
    auto func = std::make_shared<BoundFunction>(tile_code);

    FunctionApplication a(func);
    for (const auto& input : inputs) {
      a.SetInput(input.first, input.second);
    }

    std::vector<std::shared_ptr<Value>> outvals;
    outvals.reserve(outputs.size());
    for (const auto& output : outputs) {
      outvals.emplace_back(a.GetOutput(output.first));
    }

    BoundFunction f;
    f.AddDependency(a);
    auto ovit = outvals.begin();
    for (const auto& output : outputs) {
      f.AddUpdate(output.second, *ovit++);
    }

    RunInfo r = f.PrepareToRun();
    return GenerateStripe(r).program;
  }
};

TEST_F(GenStripeTest, ContractPlusElementwise) {
  auto block =
      ToStripe(R"***(
      function (A[M, K], B[K, N]) -> (X) {
        C[m, n : M, N] = +(A[m, k] * B[k, n]);
        X = tanh(C);
      }
  )***",
               {{"A", TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {10, 10}))},
                {"B", TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {10, 10}))}},
               {{"X", TensorValue::make(std::make_shared<BufferBase>(), SimpleShape(DataType::FLOAT32, {10, 10}))}});

  LOG(INFO) << "Block: " << *block;

  EXPECT_THAT(IntoProto(*block), EqualsProtoText(R"***(
    loc { unit { } }
    refs [
      {
        into:"X_I_0" access [{}, {}] loc {unit {}}
        interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        exterior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
      },
      {
        into:"X_I_1" access [{}, {}] loc {unit {}}
        interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        exterior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
      },
      {
        into:"X_T1" access [{}, {}] loc {unit {}}
        interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        exterior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
      },
      {
        into:"X_T2" access [{}, {}] loc {unit {}}
        interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        exterior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
      }
    ]
    stmts [{
      tags:["main"] block {
        name:"main" loc {unit {}}
        refs [
          {
            from:"X_I_0" into:"X_I_0" dir:In access [{}, {}] loc {unit {}}
            interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          },
          {
            from:"X_I_1" into:"X_I_1" dir:In access [{}, {}] loc {unit {}}
            interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          },
          {
            from:"X_T1" into:"X_T1" dir:InOut access [{}, {}] loc {unit {}}
            interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          },
          {
            from:"X_T2" into:"X_T2" dir:Out access [{}, {}] loc {unit {}} agg_op:"assign"
            interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
          }
        ]
        stmts [{
          tags:["agg_op_add", "comb_op_mul", "contraction", "kernel"] block {
            name:"kernel_0(X_I_0,X_I_1)" loc {unit {}}
            comments:"X_T1[m, n : _T0, _T1] = +(X_I_0[m, k] * X_I_1[k, n])"
            idxs [{name:"k" range:10 affine {}}, {name:"m" range:10 affine {}}, {name:"n", range: 10, affine {}}]
            refs [
              {
                from:"X_I_0" into:"X_I_0" dir:In loc {unit {}}
                interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
                access [{terms:{key:"m" value:1}}, {terms:{key:"k" value:1}}]
              },
              {
                from:"X_I_1" into:"X_I_1" dir:In loc {unit {}}
                interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
                access [{terms:{key:"k" value:1}}, {terms:{key:"n" value:1}}]
              },
              {
                from:"X_T1" into:"X_T1" dir:Out loc {unit {}} agg_op:"add"
                interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
                access [{terms:{key:"m" value:1}}, {terms:{key:"n" value:1}}]
              }
            ]
            stmts [
              {load {from:"X_I_0" into:"$X_I_0"}},
              {load {from:"X_I_1" into:"$X_I_1"}},
              {intrinsic {name:"mul" inputs:["$X_I_0", "$X_I_1"] outputs:"$X_T1" type:FLOAT32}},
              {store {from:"$X_T1" into:"X_T1"}}
            ]
          }
        }, {
          tags:["eltwise", "eltwise_tanh", "kernel"] block {
            name:"kernel_1(X_T1)" loc {unit {}}
            comments:"X_T2 = tanh(X_T1)"
            idxs [{name:"i1" range:10 affine {}}, {name:"i2", range:10 affine {}}]
            refs [
              {
                from:"X_T1" into:"X_T1" dir:In loc {unit {}}
                interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
                access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
              },
              {
                from:"X_T2" into:"X_T2" dir:Out loc {unit {}}
                interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                exterior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
                access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
              }
            ]
            stmts [
              {load {from:"X_T1" into:"$X_T1"}},
              {intrinsic {name:"tanh" inputs:"$X_T1" outputs:"$X_T2" type:FLOAT32}},
              {store {from:"$X_T2" into:"X_T2"}}
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
