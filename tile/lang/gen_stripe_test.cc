// Copyright 2017-2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/tile_cc.h"

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace lang {
namespace {

Tensor ContractPlusElementwise(const Tensor& A, const Tensor& B) {
  Tensor C;
  auto M = A[0], N = B[1];
  Index k, m, n;
  C({m, n}, {M, N}) += A({m, k}) * B({k, n});
  return Call("tanh", {C});
}

TEST(GenStripeTest, ContractPlusElementwise) {
  auto shape = SimpleShape(DataType::FLOAT32, {10, 10});
  Tensor A(shape), B(shape);
  auto runinfo = Evaluate("ContractPlusElementwise", {ContractPlusElementwise(A, B)});
  auto block = GenerateStripe(runinfo).program;
  LOG(INFO) << "Block: " << *block;
  EXPECT_THAT(IntoProto(*block), EqualsProtoText(R"***(
    name: "ContractPlusElementwise"
    loc {}
    refs [
      {
        key:"X0"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        }
      },
      {
        key:"X1"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        }
      },
      {
        key:"X2"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        }
      },
      {
        key:"X3"
        value: {
          access [{}, {}] loc {}
          interior_shape {type:FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
        }
      }
    ]
    stmts [{
      tags:["main"] block {
        name:"main" loc {}
        refs [
          {
            key:"X0"
            value: {
              from:"X0" dir:In access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          },
          {
            key:"X1"
            value: {
              from:"X1" dir:In access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          },
          {
            key:"X2"
            value: {
              from:"X2" dir:InOut access [{}, {}] loc {}
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          },
          {
            key:"X3"
            value: {
              from:"X3" dir:Out access [{}, {}] loc {} agg_op:"assign"
              interior_shape {type: FLOAT32 dims [{size:10 stride:10}, {size:10 stride:1}]}
            }
          }
        ]
        stmts [{
          tags:["agg_op_add", "comb_op_mul", "contraction", "kernel"] block {
            name:"kernel_0(X0,X1)" loc {}
            comments:"X2[x0, x2 : 10, 10] = +(X0[x0, x1] * X1[x1, x2])"
            idxs [{name:"x0" range:10 affine {}}, {name:"x1" range:10 affine {}}, {name:"x2", range: 10, affine {}}]
            refs [
              {
                key:"X0"
                value: {
                  from:"X0" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x0" value:1}}, {terms:{key:"x1" value:1}}]
                }
              },
              {
                key:"X1"
                value: {
                  from:"X1" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x1" value:1}}, {terms:{key:"x2" value:1}}]
                }
              },
              {
                key:"X2"
                value: {
                  from:"X2" dir:Out loc {} agg_op:"add"
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"x0" value:1}}, {terms:{key:"x2" value:1}}]
                }
              }
            ]
            stmts [
              {load {from:"X0" into:"$X0"}},
              {load {from:"X1" into:"$X1"}},
              {intrinsic {name:"mul" inputs:["$X0", "$X1"] outputs:"$X2" type:FLOAT32}},
              {store {from:"$X2" into:"X2"}}
            ]
          }
        }, {
          tags:["eltwise", "eltwise_tanh", "kernel"] block {
            name:"kernel_1(X2)" loc {}
            comments:"X3 = tanh(X2)"
            idxs [{name:"i1" range:10 affine {}}, {name:"i2", range:10 affine {}}]
            refs [
              {
                key:"X2"
                value: {
                  from:"X2" dir:In loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
                }
              },
              {
                key:"X3"
                value: {
                  from:"X3" dir:Out loc {}
                  interior_shape {type: FLOAT32 dims [{size:1 stride:10}, {size:1 stride:1}]}
                  access [{terms:{key:"i1" value:1}}, {terms:{key:"i2" value:1}}]
                }
              }
            ]
            stmts [
              {load {from:"X2" into:"$X2"}},
              {intrinsic {name:"tanh" inputs:"$X2" outputs:"$X3" type:FLOAT32}},
              {store {from:"$X3" into:"X3"}}
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
