// Copyright 2018 Intel Corporation.

#include <gtest/gtest.h>

#include "testing/matchers.h"
#include "tile/codegen/schedule.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

using ::testing::EqualsProtoText;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

template <typename P>
P ParseProtoText(const char* txt) {
  P proto;
  google::protobuf::TextFormat::ParseFromString(txt, &proto);
  return proto;
}

class ScheduleTest : public ::testing::Test {
 public:
  void SetUp() override {
    SetUpBlock();
    SetUpOptions();
    main_ = block_->SubBlock(0);
  }

  virtual void SetUpBlock() {
    block_ = stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
      name: "program" loc {unit {}}
      refs [{into: "i1" loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
            {into: "i2" loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
            {into: "o1" loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
      stmts [{
        tags: ["main"] block {
          name: "main" loc {unit {}}
          refs [{from: "i1" into: "i1" dir: In loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                {from: "i2" into: "i2" dir: In loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                {from: "o1" into: "o1" dir: Out loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
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

  void AddTmpRefinement(const char* name, const TensorDimension& dim) {
    main_->refs.emplace_back(stripe::Refinement{
        stripe::RefDir::None,                   // dir
        "",                                     // from
        name,                                   // into
        {stripe::Affine{}},                     // access
        TensorShape(DataType::FLOAT32, {dim}),  // shape
        "",                                     // agg_op
        stripe::Location{"RAM"},                // location
    });
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
    loc { unit { } }
    refs [{into: "i1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "i2" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "o1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
    stmts [{
      tags: ["main"] block {
        name: "main" loc {unit {}}
        refs [{dir: In from: "i1" into: "i1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {dir: In from: "i2" into: "i2" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {dir: Out from: "o1" into: "o1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
      }
    }]
  )"));
}

TEST_F(ScheduleTest, CachesIO) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" loc {unit {}}
    refs [{from: "i1" into: "i1" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "i2" into: "i2" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "o1" into: "o1" dir: Out loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
  )")));
  SchedulePass(block_.get(), options_);
  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    loc { unit { } }
    refs [{into: "i1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "i2" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "o1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
    stmts [{
      tags: ["main"] block {
        name: "main" loc {unit {}}
        refs [{from: "i1" into: "i1" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "i1_whole_0" offset: 128 loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {from: "i2" into: "i2" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "i2_whole_0" offset: 64 loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {from: "o1" into: "o1" dir: Out loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "o1_whole_0" loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
        stmts [{
          block {
            name: "swap_in_i2_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i2" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "i2_whole_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "swap_in_i1_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i1" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "i1_whole_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "sub_block_1" loc {unit {}}
            refs [{from: "i1_whole_0" into: "i1" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "i2_whole_0" into: "i2" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "o1_whole_0" into: "o1" dir: Out loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
          }
          deps: [0, 1]
        }, {
          block {
            name: "swap_out_o1_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "o1_whole_0" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "o1" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
          deps: [2]
        }]
      }
    }]
  )"));
}

TEST_F(ScheduleTest, UsesTmps) {
  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_1" loc {unit {}}
    refs [{from: "i1" into: "i1" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "i2" into: "i2" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "t1" into: "t1" dir: Out loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
  )")));

  main_->stmts.emplace_back(stripe::FromProto(ParseProtoText<stripe::proto::Block>(R"(
    name: "sub_block_2" loc {unit {}}
    refs [{from: "t1" into: "t1" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "i2" into: "i2" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {from: "o1" into: "o1" dir: Out loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
  )")));

  AddTmpRefinement("t1", TensorDimension{1, 16});

  SchedulePass(block_.get(), options_);

  EXPECT_THAT(IntoProto(*block_), EqualsProtoText(R"(
    name: "program"
    loc { unit { } }
    refs [{into: "i1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "i2" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
          {into: "o1" loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
    stmts [{
      tags: ["main"] block {
        name: "main" loc {unit {}}
        refs [{from: "i1" into: "i1" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "i1_whole_0" offset: 64 loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {from: "i2" into: "i2" dir: In loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "i2_whole_0" offset: 128 loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {from: "o1" into: "o1" dir: Out loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "o1_whole_0" offset: 64 loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "t1" loc {name: "RAM" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
              {into: "t1_whole_0" loc {name: "CACHE" unit {}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
        stmts [{
          block {
            name: "swap_in_i1_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i1" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "i1_whole_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "swap_in_i2_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "i2" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "i2_whole_0" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
        }, {
          block {
            name: "sub_block_1" loc {unit {}}
            refs [{from: "i1_whole_0" into: "i1" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "i2_whole_0" into: "i2" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "t1_whole_0" into: "t1" dir: Out loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
          }
          deps: [0, 1]
        }, {
          block {
            name: "sub_block_2" loc {unit {}}
            refs [{from: "t1_whole_0" into: "t1" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "i2_whole_0" into: "i2" dir: In loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}},
                  {from: "o1_whole_0" into: "o1" dir: Out loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:16 stride:1}} access {}}]
          }
          deps: [2]
        }, {
          block {
            name: "swap_out_o1_whole_0" loc {name: "DMA" unit {}}
            idxs [{name: "i0" range: 16 affine {}}]
            refs [{from: "o1_whole_0" into: "src" dir: In access [{terms [{key: "i0" value: 1}]}] loc {name: "CACHE" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}},
                  {from: "o1" into: "dst" dir: Out access [{terms [{key: "i0" value: 1}]}] loc {name: "RAM" unit{}} shape {type: FLOAT32 dims: {size:1 stride:1}}}]
            stmts [{load: {from: "src" into: "$X"}}, {store: {from: "$X" into: "dst"}}]
          }
          deps: [3]
        }]
      }
    }]
  )"));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
