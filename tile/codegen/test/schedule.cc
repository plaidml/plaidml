// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/codegen/schedule.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

namespace gp = google::protobuf;

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace codegen {

class ScheduleTest : public ::testing::Test {
 public:
  void SetUp() override {
    SetUpBlock();
    SetUpOptions();
    main_ = std::dynamic_pointer_cast<stripe::Block>(block_->stmts.front());
  }

  template <typename P>
  P ParseProtoText(const char* txt) {
    P proto;
    gp::TextFormat::ParseFromString(txt, &proto);
    return proto;
  }

  virtual void SetUpBlock() {
    block_ = stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
      name: "program" location {unit {}}
      refs [{into: "i1" location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
            {into: "i2" location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
            {into: "o1" location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
      stmts [{
        tags: ["main"] block {
          name: "main" location {unit {}}
          refs [{from: "i1" into: "i1" dir: In location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                {from: "i2" into: "i2" dir: In location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                {from: "o1" into: "o1" dir: Out location {name: "RAM" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
        }
      }]
    )"));
  }

  virtual void SetUpOptions() {
    options_ = ParseProtoText<proto::SchedulePass>(R"(
      reqs: ["main"],
      mem_loc: { name: "CACHE" },
      mem_KiB: 1024,
      alignment: 16,
      xfer_loc: { name: "DMA" }
    )");
  }

  void AddTmpRefinement(const char* name, TensorDimension dim) {
    stripe::Refinement ref;
    ref.dir = stripe::RefDir::None;
    ref.into = name;
    ref.shape.type = DataType::FLOAT32;
    ref.shape.dims.emplace_back(std::move(dim));
    ref.location.name = "RAM";
    main_->refs.emplace_back(std::move(ref));
  }

 protected:
  std::shared_ptr<stripe::Block> block_;
  std::shared_ptr<stripe::Block> main_;
  proto::SchedulePass options_;
};

TEST_F(ScheduleTest, EmptyMain) {
  SchedulePass(block_.get(), options_);
  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    location { unit { } }
    refs [{into: "i1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "i2" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "o1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
    stmts [{
      tags: ["main"] block {
        name: "main" location {unit {}}
        refs [{dir: In from: "i1" into: "i1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {dir: In from: "i2" into: "i2" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {dir: Out from: "o1" into: "o1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
      }
    }]
  )"));
}

TEST_F(ScheduleTest, DISABLED_CachesIO) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" location {unit {}}
    refs [{from: "i1" into: "i1" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "i2" into: "i2" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "o1" into: "o1" dir: Out location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
  )")));
  SchedulePass(block_.get(), options_);
  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    location { unit { } }
    refs [{into: "i1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "i2" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "o1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
    stmts [{
      tags: ["main"] block {
        name: "main" location {unit {}}
        refs [{from: "i1" into: "i1" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "i1_0" offset: 128 location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {from: "i2" into: "i2" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "i2_0" offset: 64 location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {from: "o1" into: "o1" dir: Out location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "o1_0" location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
        stmts [{
          block {
            name: "swap_in_i2_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i2" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "i2_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "swap_in_i1_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i1" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "i1_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "sub_block_1" location {unit {}}
            refs [{from: "i1_0" into: "i1" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "i2_0" into: "i2" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "o1_0" into: "o1" dir: Out location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
          }
          deps: [0, 1]
        }, {
          block {
            name: "swap_out_o1_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "o1_0" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "o1" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
          deps: [2]
        }]
      }
    }]
  )"));
}

TEST_F(ScheduleTest, DISABLED_UsesTmps) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" location {unit {}}
    refs [{from: "i1" into: "i1" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "i2" into: "i2" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "t1" into: "t1" dir: Out location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
  )")));

  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_2" location {unit {}}
    refs [{from: "t1" into: "t1" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "i2" into: "i2" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {from: "o1" into: "o1" dir: Out location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
  )")));

  AddTmpRefinement("t1", TensorDimension{1, 16});

  SchedulePass(block_.get(), options_);

  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    location { unit { } }
    refs [{into: "i1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "i2" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
          {into: "o1" location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
    stmts [{
      tags: ["main"] block {
        name: "main" location {unit {}}
        refs [{from: "i1" into: "i1" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "i1_0" offset: 64 location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {from: "i2" into: "i2" dir: In location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "i2_0" offset: 128 location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {from: "o1" into: "o1" dir: Out location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "o1_0" offset: 64 location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
              {into: "t1_0" location {name: "CACHE" unit {}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
        stmts [{
          block {
            name: "swap_in_i1_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i1" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "i1_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "swap_in_i2_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i2" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "i2_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "sub_block_1" location {unit {}}
            refs [{from: "i1_0" into: "i1" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "i2_0" into: "i2" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "t1_0" into: "t1" dir: Out location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
          }
          deps: [0, 1]
        }, {
          block {
            name: "sub_block_2" location {unit {}}
            refs [{from: "t1_0" into: "t1" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "i2_0" into: "i2" dir: In location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}},
                  {from: "o1_0" into: "o1" dir: Out location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:16 stride:1}}}]
          }
          deps: [2]
        }, {
          block {
            name: "swap_out_o1_0" location {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "o1_0" into: "src" dir: In is_const: true access [{terms [{key: "i0" value: 1}]}] location {name: "CACHE" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}},
                  {from: "o1" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] location {name: "RAM" unit{}} shape {type: FLOAT32 dimensions: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
          deps: [3]
        }]
      }
    }]
  )"));
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
