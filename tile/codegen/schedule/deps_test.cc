// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/codegen/schedule/deps.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

namespace gp = google::protobuf;

using ::testing::EqualsProto;

namespace vertexai {
namespace tile {
namespace codegen {
namespace schedule {

TEST(DepsTest, SmallDepMix) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    location { unit { } }
    refs {
      location { unit { } }
      into: "b1"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    refs {
      location { unit { } }
      into: "b2"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } }
    stmts { load { from:"b2" into:"$2" } deps: 0}
    stmts { constant { name:"$3" iconst: 0 } }
    stmts { intrinsic { name:"ADD" inputs:"$2" inputs:"$3" outputs:"$4"} }
    stmts { special { name:"COPY" inputs:"b1" outputs:"b2"} }
    stmts { store { from:"$4" into:"b2"} }
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  ComputeDepsForTree(block.get());

  stripe::proto::Block output_proto{IntoProto(*block)};

  EXPECT_THAT(output_proto, EqualsProto(R"(
    location { unit { } }
    refs {
      location { unit { } }
      into: "b1"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    refs {
      location { unit { } }
      into: "b2"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { load { from:"b2" into:"$2" } deps: 1}
    stmts { constant { name:"$3" iconst: 0 } }
    stmts { intrinsic { name:"ADD" inputs:"$2" inputs:"$3" outputs:"$4"} deps: 2 deps: 3}
    stmts { special { name:"COPY" inputs:"b1" outputs:"b2"} deps: 2}
    stmts { store { from:"$4" into:"b2"} deps: 4 deps: 5}
  )"));
}

}  // namespace schedule
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
