// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>

#include "tile/codegen/tile.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"
#include "tile/targets/cpu/jit.h"

namespace gp = google::protobuf;

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {
namespace test {

TEST(Jit, JitIntrinsicMUL_F32) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    refs [
      {
        key: "b1"
        value {
          loc {}
          dir: 3
          interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          access { }
        }
      },
      {
        key: "b2"
        value {
          loc {}
          dir: 3
          interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          access { }
        }
      }
    ]
    stmts { load { from:"b1" into:"$1" } }
    stmts { load { from:"b2" into:"$2" } }
    stmts { intrinsic { name:"mul" type:FLOAT32 inputs:"$1" inputs:"$2" outputs:"$3"} }
    stmts { store { from:"$3" into:"b2"} }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> b1{3.0};
  std::vector<float> b2{2.0};
  std::map<std::string, void*> buffers{{"b1", b1.data()}, {"b2", b2.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(b2[0], Eq(6.0));
}

TEST(Jit, JitIntrinsicADD_F32) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    refs [
      {
        key: "b1"
        value {
          loc {}
          dir: 3
          interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          access { }
        }
      },
      {
        key: "b2"
        value {
          loc {}
          dir: 3
          interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          access { }
        }
      }
    ]
    stmts { load { from:"b1" into:"$1" } }
    stmts { load { from:"b2" into:"$2" } }
    stmts { intrinsic { name:"add" type:FLOAT32 inputs:"$1" inputs:"$2" outputs:"$3"} }
    stmts { store { from:"$3" into:"b2"} }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> b1{1.0};
  std::vector<float> b2{2.0};
  std::map<std::string, void*> buffers{{"b1", b1.data()}, {"b2", b2.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(b2[0], Eq(3.0));
}

TEST(Jit, JitIntrinsicEQ) {}

TEST(Jit, JitIntrinsicCOND) {}

TEST(Jit, JitSimpleLoop) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    idxs {
      name: "i"
      range: 5
      },
    refs [
      {
        key: "bufA"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key: "i" value: 1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:1} }
        }
      },
      {
        key: "bufB"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key: "i" value: 1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:1} }
        }
      }
    ]
    stmts { load { from:"bufA" into:"$1" } }
    stmts { store { from:"$1" into:"bufB"} }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> bufA = {
      1, 2, 3, 4, 5,
  };
  std::vector<float> bufB = {
      0, 0, 0, 0, 0,
  };
  std::vector<float> expected = {
      1, 2, 3, 4, 5,
  };

  std::map<std::string, void*> buffers{{"bufA", bufA.data()}, {"bufB", bufB.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(bufB, ContainerEq(expected));
}

TEST(Jit, JitCopy2D) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    idxs { name: "i" range: 5 }
    idxs { name: "j" range: 5 }
    refs [
      {
        key: "bufA"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:1} }
        }
      },
      {
        key: "bufB"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key:"i" value:1} }
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:5} dims: {size:5 stride:1} }
        }
      }
    ]
    stmts { load { from:"bufA" into:"$1" } }
    stmts { store { from:"$1" into:"bufB"} }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> bufA = {
      1, 2, 3, 4, 5,
  };
  std::vector<float> bufB = {
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
  };
  std::vector<float> expected = {
      1, 2, 3, 4, 5,  //
      1, 2, 3, 4, 5,  //
      1, 2, 3, 4, 5,  //
      1, 2, 3, 4, 5,  //
      1, 2, 3, 4, 5,  //
  };

  std::map<std::string, void*> buffers{{"bufA", bufA.data()}, {"bufB", bufB.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(bufB, ContainerEq(expected));
}

TEST(Jit, JitAggSum2D) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    idxs { name: "i" range: 5 }
    idxs { name: "j" range: 5 }
    refs [
      {
        key: "bufA"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:1} }
        }
      },
      {
        key: "bufB"
        value {
          loc {}
          dir: 3
          agg_op: "add"
          access { offset: 0 terms {key:"i" value:1} }
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:5} dims: {size:5 stride:1} }
        }
      }
    ]
    stmts { load { from:"bufA" into:"$1" } }
    stmts { store { from:"$1" into:"bufB"} }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> bufA = {
      1, 2, 3, 4, 5,
  };
  std::vector<float> bufB = {
      1,  2,  3,  4,  5,   //
      6,  7,  8,  9,  10,  //
      11, 12, 13, 14, 15,  //
      16, 17, 18, 19, 20,  //
      21, 22, 23, 24, 25,  //
  };
  std::vector<float> expected = {
      2,  4,  6,  8,  10,  //
      7,  9,  11, 13, 15,  //
      12, 14, 16, 18, 20,  //
      17, 19, 21, 23, 25,  //
      22, 24, 26, 28, 30   //
  };

  std::map<std::string, void*> buffers{{"bufA", bufA.data()}, {"bufB", bufB.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(bufB, ContainerEq(expected));
}

TEST(Jit, JitMatMul) {
  std::vector<float> bufA = {
      1, 2, 3, 4, 5,  //
      4, 5, 6, 7, 8,  //
      7, 8, 9, 7, 8,  //
      1, 2, 3, 1, 2,  //
      1, 2, 3, 1, 2,  //
  };

  std::vector<float> bufB = {
      1, 2, 3, 1, 2,  //
      1, 2, 3, 1, 2,  //
      1, 2, 3, 1, 2,  //
      1, 2, 3, 1, 2,  //
      1, 2, 3, 1, 2,  //
  };

  std::vector<float> bufC = {
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0,  //
  };

  std::vector<float> expected = {
      15, 30, 45,  15, 30,  //
      30, 60, 90,  30, 60,  //
      39, 78, 117, 39, 78,  //
      9,  18, 27,  9,  18,  //
      9,  18, 27,  9,  18,  //
  };

  lang::RunInfo runinfo;
  runinfo.program_name = "matmul";
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  size_t dim = sqrt(expected.size());
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {dim, dim}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {dim, dim}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {dim, dim}));

  auto stripe = GenerateStripe(runinfo);
  auto main = stripe.program->SubBlock(0);

  IVLOG(2, "Before>\n" << *main);

  std::map<std::string, void*> data = {
      {"A", bufA.data()},
      {"B", bufB.data()},
      {"C", bufC.data()},
  };
  JitExecute(*main, data);

  IVLOG(2, "A: " << bufA);
  IVLOG(2, "B: " << bufB);
  IVLOG(2, "C: " << bufC);
  EXPECT_THAT(bufC, ContainerEq(expected));
}

TEST(Jit, JitNestedAlloc) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc {}
    idxs { name: "i" range: 5 }
    idxs { name: "j" range: 5 }
    refs [
      {
        key: "bufA"
        value {
          loc {}
          dir: 3
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:1} }
        }
      },
      {
        key: "bufB"
        value {
          loc {}
          dir: 3
          agg_op: "add"
          access { offset: 0 terms {key:"i" value:1} }
          access { offset: 0 terms {key:"j" value:1} }
          interior_shape { type: FLOAT32 dims: {size:5 stride:5} dims: {size:5 stride:1} }
        }
      }
    ]
    stmts { block {
      refs [
        {
          key: "bufA"
          value {
            loc {}
            dir: 3
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
            access { }
          }
        },
        {
          key: "bufB"
          value {
            loc {}
            dir: 3
            agg_op: "add"
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
            access { }
          }
        },
        {
          key: "bufTemp"
          value {
            dir: 0
            interior_shape { type: INT32 dims: {size:5 stride:1} }
            access { }
          }
        }
      ]
      stmts { load { from:"bufA" into:"$1" } }
      stmts { store { from:"$1" into:"bufB"} }
    } }
  )",
                                  &input_proto);
  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  std::vector<float> bufA = {
      1, 2, 3, 4, 5,
  };
  std::vector<float> bufB = {
      1,  2,  3,  4,  5,   //
      6,  7,  8,  9,  10,  //
      11, 12, 13, 14, 15,  //
      16, 17, 18, 19, 20,  //
      21, 22, 23, 24, 25,  //
  };
  std::vector<float> expected = {
      2,  4,  6,  8,  10,  //
      7,  9,  11, 13, 15,  //
      12, 14, 16, 18, 20,  //
      17, 19, 21, 23, 25,  //
      22, 24, 26, 28, 30   //
  };

  std::map<std::string, void*> buffers{{"bufA", bufA.data()}, {"bufB", bufB.data()}};
  JitExecute(*block, buffers);

  EXPECT_THAT(bufB, ContainerEq(expected));
}

}  // namespace test
}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
