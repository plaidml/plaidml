// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>

#include "base/proto/proto.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT
using ::testing::Eq;
using ::testing::Ne;

static proto::Stage GenerateStage() {
  auto cfg_tmpl = R"(
    passes: [
      {
        name: "loc_prog"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
            reqs: ["program"]
            loc: { devs: [{name: "DRAM"}] }
          }
        }
      }, {
        name: "loc_main"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
            reqs: ["main"]
            loc: { devs: [{name: "DRAM"}] }
          }
        }
      }, {
        name: "loc_initial"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateBlockPass] {
            reqs: ["kernel"]
            loc: { devs: [{name: "PPE"}] }
          }
        }
      }, {
        name: "compute_deps"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.ComputeDepsPass] {
            reqs: ["all"]
          }
        }
      }, {
        name: "dead_code_elimination"
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass] {
            reqs: ["all"]
          }
        }
      }
    ]
  )";
  return ParseProtoText<proto::Stage>(cfg_tmpl);
}

static std::string RemoveComments(std::string src) {
  size_t pos = 0;
  std::string dest = "";
  while (pos < src.size()) {
    size_t next = src.find("//", pos);
    if (next == std::string::npos) {
      dest = dest + src.substr(pos);
      return dest;
    }
    dest = dest + src.substr(pos, next - pos);
    pos = src.find('\n', next + 2);
    if (pos == std::string::npos) {
      return dest;
    }
  }
  return dest;
}

TEST(DCETest, RemoveBlock) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (XA[W, X, Y, Z]) -> (XB) {
      XC = XA + XA;
      XB = index(XC, 2);
    }
  )***";
  runinfo.input_shapes.emplace("XA", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  runinfo.output_shapes.emplace("XB", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::Optimize(program->entry.get(), stage.passes(), options);
  // Expected result should not contain XA or XC
  std::string actual_after = RemoveComments(to_string(*program->entry));
  IVLOG(1, "After stripe optimization: " << actual_after);

  EXPECT_THAT(actual_after.find("XA"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("XC"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("XB"), Ne(std::string::npos));
}

TEST(DCETest, SimpleTest) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (X_T0[D0, D1]) -> (X_T5) {
      X_T1 = X_T0 + X_T0;
      X_T2[x0: D0] = >(X_T1[x0, x1]);
      X_T3[x1: D1] = >(X_T1[x0, x1]);
      X_T4[x0, x1: D0, D1] = =(X_T2[x0] + X_T3[x1]);
      X_T5 = index(X_T4, 0);
    }
  )***";
  runinfo.input_shapes.emplace("X_T0", SimpleShape(DataType::FLOAT32, {3, 3}));
  runinfo.output_shapes.emplace("X_T5", SimpleShape(DataType::FLOAT32, {3, 3}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::Optimize(program->entry.get(), stage.passes(), options);

  // Expected result should not contain X_T1, X_T2, X_T3, X_T4
  std::string actual_after = RemoveComments(to_string(*program->entry));
  IVLOG(1, "After stripe optimization: " << actual_after);

  EXPECT_THAT(actual_after.find("X_T1"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T2"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T3"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T4"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T5"), Ne(std::string::npos));
}

TEST(DCETest, MultiUseTest) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (X_T0[D0, D1]) -> (X_T3, X_T5) {
      X_T1 = X_T0 + X_T0;
      X_T2[] = +(X_T0[x0, x1]);
      X_T3 = X_T1 * X_T2;
      X_T4 = X_T3 * X_T2;
      X_T5 = index(X_T4, 0); 
    }
  )***";

  runinfo.input_shapes.emplace("X_T0", SimpleShape(DataType::FLOAT32, {3, 3}));
  runinfo.output_shapes.emplace("X_T3", SimpleShape(DataType::FLOAT32, {3, 3}));
  runinfo.output_shapes.emplace("X_T5", SimpleShape(DataType::FLOAT32, {3, 3}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::Optimize(program->entry.get(), stage.passes(), options);

  // Expected result should not contain X_T4
  std::string actual_after = RemoveComments(to_string(*program->entry));
  IVLOG(1, "After stripe optimization: " << actual_after);

  EXPECT_THAT(actual_after.find("X_T1"), Ne(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T2"), Ne(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T3"), Ne(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T4"), Eq(std::string::npos));
  EXPECT_THAT(actual_after.find("X_T5"), Ne(std::string::npos));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
