// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/codegen/deps.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

namespace gp = google::protobuf;

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace codegen {

TEST(DepsTest, SmallDepMix) {
  auto input_text = R"(
    loc {}
    stmts {
      attrs: { key: "main" value {} }
      block {
        loc {}
        refs: [{
          key: "b1"
          value: {
            loc {}
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} } 
          }
        }, {
          key: "b2"
          value: {
            loc {}
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          }
        }]
        stmts { load { from:"b1" into:"$1" } }
        stmts { store { from:"$1" into:"b2" } }
        stmts { load { from:"b2" into:"$2" } deps: 0}
        stmts { constant { name:"$3" iconst: 0 } }
        stmts { intrinsic { name:"ADD" type:FLOAT32 inputs:"$2" inputs:"$3" outputs:"$4"} }
        stmts { special { name:"COPY" inputs:"b1" outputs:"b2"} }
        stmts { store { from:"$4" into:"b2"} }
      }
    }
  )";
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(input_text, &input_proto);

  auto prog = std::make_shared<stripe::Program>();
  prog->entry = stripe::FromProto(input_proto);
  CompilerState state(prog);
  proto::ComputeDepsPass options;
  options.add_reqs("main");
  ComputeDepsPass(options).Apply(&state);

  const char* expected = R"(
    loc {}
    stmts {
      attrs: { key: "main" value {} }
      block {
        loc {}
        refs [{
          key: "b1"
          value: {
            loc {}
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} } 
          }
        }, {
          key: "b2"
          value: {
            loc {}
            interior_shape { type: FLOAT32 dims: {size:1 stride:1} }
          }
        }]
        stmts { load { from:"b1" into:"$1" } }
        stmts { store { from:"$1" into:"b2" } deps: 0 }
        stmts { load { from:"b2" into:"$2" } deps: 1}
        stmts { constant { name:"$3" iconst: 0 } }
        stmts { intrinsic { name:"ADD" type:FLOAT32 inputs:"$2" inputs:"$3" outputs:"$4"} deps: 2 deps: 3}
        stmts { special { name:"COPY" inputs:"b1" outputs:"b2"} deps: 2}
        stmts { store { from:"$4" into:"b2"} deps: 4 deps: 5}
      }
    }
  )";

  auto output_proto = IntoProto(*prog->entry);
  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

TEST(DepsTest, Subregion) {
  auto input_text = R"(
    refs [{
      key: "buf"
      value: {
        access [ { offset: 0 }, { offset: 0 } ]
        interior_shape { type: FLOAT32 dims: { size:2 stride:10 } dims: { size:10 stride:1 } }
        loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
      }
    }]
    stmts {
      attrs: { key: "main" value {} }
      block {
        idxs { name: "i" range: 10 affine { } }
        refs [{
          key: "b1"
          value: {
            from: "buf" dir: In
            access [ { offset: 0 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
          }
        }, {
          key: "b2",
          value: {
            from: "buf" dir: In
            access [ { offset: 1 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
          }
        }, {
          key: "b3",
          value: {
            from: "buf" dir: In
            access [ { offset: 1 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 1}]}] }
          }
        }]
        stmts { constant { name:"$1" iconst: 0 } }
        stmts { store { from:"$1" into:"b1" } }
        stmts { store { from:"$1" into:"b2" } }
        stmts { store { from:"$1" into:"b3" } }
      }
    }
  )";
  stripe::proto::Block input_proto;
  gp::TextFormat::ParseFromString(input_text, &input_proto);

  auto prog = std::make_shared<stripe::Program>();
  prog->entry = stripe::FromProto(input_proto);
  CompilerState state(prog);
  proto::ComputeDepsPass options;
  options.add_reqs("main");
  ComputeDepsPass(options).Apply(&state);

  const char* expected = R"(
    loc {}
    refs [{
      key: "buf"
      value: {
        access [ { offset: 0 }, { offset: 0 } ]
        interior_shape { type: FLOAT32 dims: { size:2 stride:10 } dims: { size:10 stride:1 } }
        loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
      }
    }]
    stmts {
      attrs: { key: "main" value {} }
      block {
        loc {}
        idxs { name: "i" range: 10 affine { } }
        refs [{
          key: "b1"
          value: {
            from: "buf" dir: In
            access [ { offset: 0 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
          }
        }, {
          key: "b2"
          value: {
            from: "buf" dir: In
            access [ { offset: 1 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 0}]}] }
          }
        }, {
          key: "b3"
          value: {
            from: "buf" dir: In
            access [ { offset: 1 }, { terms [ { key: "i" value: 1 } ] } ]
            interior_shape { type: FLOAT32 dims: { size:1 stride:1 } dims: { size:1 stride:1 } }
            loc { devs: [{name: "RAM", units: [{offset: 1}]}] }
          }
        }]
        stmts { constant { name:"$1" iconst: 0 } }
        stmts { store { from:"$1" into:"b1" } deps: 0 }
        stmts { store { from:"$1" into:"b2" } deps: 0 }
        stmts { store { from:"$1" into:"b3" } deps: 0 }
      }
    }
  )";

  auto output_proto = IntoProto(*prog->entry);
  EXPECT_THAT(output_proto, EqualsProtoText(expected));
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
