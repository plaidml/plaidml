// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/codegen/placer.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

namespace gp = google::protobuf;

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace codegen {

TEST(PlacerTest, TemporalSeparationCausesSpatialReuse) {
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
    stmts { store { from:"$1" into:"b2" } deps: 0 }
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  PlaceRefinements(block.get());

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
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
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(PlacerTest, TemporalOverlapCausesSpacialSeparation) {
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
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  PlaceRefinements(block.get());

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
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
      offset: 16
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(PlacerTest, DistinctLocationCausesSpacialReuse) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    location { unit { } }
    refs {
      location { name: "loc_1" unit { } }
      into: "b1"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    refs {
      location { name: "loc_2" unit { } }
      into: "b2"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  PlaceRefinements(block.get());

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
    location { unit { } }
    refs {
      location { name: "loc_1" unit { } }
      into: "b1"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    refs {
      location { name: "loc_2" unit { } }
      into: "b2"
      shape {
        type: FLOAT32
        dimensions: {size:1 stride:1}
      }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
