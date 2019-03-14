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
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  proto::MemoryPlacementPass options;
  options.add_locs()->set_name("loc_1");

  PlaceRefinements(block.get(), options);

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(PlacerTest, TemporalOverlapCausesSpacialSeparation) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  proto::MemoryPlacementPass options;
  options.add_locs()->set_name("loc_1");
  options.set_alignment(16);

  PlaceRefinements(block.get(), options);

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      offset: 16
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(PlacerTest, DistinctlocCausesSpacialReuse) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  proto::MemoryPlacementPass options;
  options.add_locs()->set_name("loc_1");
  options.add_locs()->set_name("loc_2");

  PlaceRefinements(block.get(), options);

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(PlacerTest, LocationSubsetCanBePlaced) {
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b3"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
    stmts { special { name:"COPY" inputs:"b2" outputs:"b3"} deps: 1}
  )",
                                  &input_proto);

  std::shared_ptr<stripe::Block> block{stripe::FromProto(input_proto)};

  proto::MemoryPlacementPass options;
  options.add_locs()->set_name("loc_1");

  PlaceRefinements(block.get(), options);

  stripe::proto::Block output_proto{IntoProto(*block)};

  const char* expected = R"(
    loc { unit { } }
    refs {
      loc { name: "loc_1" unit { } }
      into: "b1"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b2"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    refs {
      loc { name: "loc_2" unit { } }
      into: "b3"
      interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
      exterior_shape { type: FLOAT32 dims: {size:1 stride:1} }
    }
    stmts { load { from:"b1" into:"$1" } }
    stmts { store { from:"$1" into:"b2" } deps: 0 }
    stmts { special { name:"COPY" inputs:"b2" outputs:"b1"} deps: 1}
    stmts { special { name:"COPY" inputs:"b2" outputs:"b3"} deps: 1}
  )";

  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
