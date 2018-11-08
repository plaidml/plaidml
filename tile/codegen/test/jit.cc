// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>

#include "tile/codegen/jit.h"
#include "tile/codegen/tile.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

namespace gp = google::protobuf;

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

TEST(Codegen, JitIntrinsicMUL) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    location { unit { } }
    refs {
      location { unit { } }
      into: "b1"
      shape { type: FLOAT32 dimensions: {size:1 stride:1} }
    }
    refs {
      location { unit { } }
      into: "b2"
      shape { type: FLOAT32 dimensions: {size:1 stride:1} }
    }
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

TEST(Codegen, JitIntrinsicADD) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    location { unit { } }
    refs {
      location { unit { } }
      into: "b1"
      shape { type: FLOAT32 dimensions: {size:1 stride:1} }
    }
    refs {
      location { unit { } }
      into: "b2"
      shape { type: FLOAT32 dimensions: {size:1 stride:1} }
    }
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

TEST(Codegen, JitIntrinsicEQ) {}

TEST(Codegen, JitIntrinsicCOND) {}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
