// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "base/proto/proto.h"
#include "base/util/throw.h"
#include "testing/matchers.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/driver.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;
using ::testing::LinesEq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT

static std::string EraseSpace(const std::string& src) {
  std::string str = src;
  size_t i = 0;
  while (i < str.size()) {
    size_t j = i;
    while (j < str.size() && str[j] == ' ') {
      ++j;
    }
    str.erase(i, j - i);
    if (str.size() == 0) {
      break;
    }
    i = str.find('\n', i);
    if (i == std::string::npos) {
      i = str.size();
    }
    j = i - 1;
    while (j >= 0 && str[j] == ' ') {
      --j;
    }
    if (i - j > 1) {
      str.erase(j + 1, i - j - 1);
    }
    i = j + 2;
  }
  return str;
}

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
        name: "loc_initial",
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.LocateBlockPass] {
            reqs: ["kernel"]
            loc: { devs: [{name: "PPE"}] }
          }
        }
      }, {
        name: "stencil_mac",
        pass: {
          [type.vertex.ai/vertexai.tile.codegen.proto.StencilPass] {
            reqs: ["agg_op_add", "comb_op_mul"],
            outer_set: ["dpu"],
            inner_set: ["dpu_inner"],
            stencils: [
                {
                    startup_cost: 32,
                    idxs: [
                        { name: "k", size: 16, outs: [-1], ins: [-1,  0] },
                        { name: "x", size: 16, outs: [-1], ins: [ 0, -1] },
                        { name: "c", size: -1, outs: [ 0], ins: [-1, -1] }
                    ]
                },
                {
                    startup_cost: 32,
                    idxs: [
                        { name: "k", size: 16, outs: [-1], ins: [ 0, -1] },
                        { name: "x", size: 16, outs: [-1], ins: [-1,  0] },
                        { name: "c", size: -1, outs: [ 0], ins: [-1, -1] }
                    ]
                },
                {
                    startup_cost: 32,
                    idxs: [
                        { name: "k", size: 16, outs: [-1], ins: [-1,  0] },
                        { name: "x", size:  4, outs: [-1], ins: [ 0, -1] },
                        { name: "y", size:  4, outs: [-1], ins: [ 0, -1] },
                        { name: "c", size: -1, outs: [ 0], ins: [-1, -1] }
                    ]
                },
                {
                    startup_cost: 32,
                    idxs: [
                        { name: "k", size: 16, outs: [-1], ins: [ 0, -1] },
                        { name: "x", size:  4, outs: [-1], ins: [-1,  0] },
                        { name: "y", size:  4, outs: [-1], ins: [-1,  0] },
                        { name: "c", size: -1, outs: [ 0], ins: [-1, -1] }
                    ]
                }
            ]
        }
      }
    }, {
      name: "fuse_dpu_add"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          a_reqs: ["dpu"]
          b_reqs: ["eltwise_add"]
          fused_set: ["dpu"]
        }
      }
    }, {
      name: "fuse_dpu_mul"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          a_reqs: ["dpu"]
          b_reqs: ["eltwise_mul"]
          fused_set: ["dpu"]
        }
      }
    }, {
      name: "fuse_dpu_shift"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          a_reqs: ["dpu"]
          b_reqs: ["eltwise_bit_right"]
          fused_set: ["dpu"]
        }
      }
    }, {
      name: "fuse_dpu_zelu"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          a_reqs: ["dpu"]
          b_reqs: ["eltwise_zelu"]
          fused_set: ["dpu"]
        }
      }
    }, {
      name: "fuse_eltwise"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          a_reqs: ["eltwise"]
          b_reqs: ["eltwise"]
          fused_set: ["dpu"]
        }
      }
    }, {
      name: "fuse_dpu"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.FusionPass] {
          parent_reqs: ["dpu"]
          a_reqs: ["eltwise"]
          fused_set: ["dpu_fusion"]
        }
      }
    }, {
      name: "light_cstr_reduction"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LightConstraintReductionPass] {
          reqs: ["all"]
        }
      }
    }, {
      name: "localize_main"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass] {
          reqs: ["main"]
        }
      }
    }, {
      name: "scalarize_main"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.ScalarizePass] {
          reqs: ["main"]
        }
      }
    }, {
      name: "fit_into_cmx",
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass] {
          reqs: ["kernel"],
          outer_set: ["fit_part"],
          skip_1d: true,
          only_po2: true,
          max_total_size : 1024,
          input_cost: 1.0,
          output_cost: 1.0,
          copy_tags: true,
          clear_outer: true,
          acc_idxs: false
        }
      }
    }, {
      name: "loc_dpu"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateBlockPass] {
          reqs: ["dpu"]
          loc: { devs: [{name: "DPU"}] }
        }
      }
    }, {
      name: "loc_mpe"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateInnerBlockPass] {
          reqs: ["dpu"]
          loc: { devs: [{name: "MPE"}] }
        }
      }
    }, {
      name: "loc_dpu_mem"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
          reqs: ["dpu"]
          loc: { devs: [{name: "ACC"}] }
        }
      }
    }, {
      name: "loc_dpu_fusion_mem"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass] {
          reqs: ["dpu_fusion"], loc: { devs: [{name: "ACC"}] }
        }
      }
    }, {
      name: "loc_dpu_fused"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateBlockPass] {
          reqs: ["eltwise", "dpu_fusion"]
          loc: { devs: [{name: "MPE"}] }
        }
      }
    }, {
      name: "loc_dpu_eltwise"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.LocateInnerBlockPass] {
          reqs: ["dpu"]
          inner_reqs: ["eltwise"], loc: { devs: [{name: "MPE"}] }
        }
      }
    }, {
      name: "schedule_main",
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.SchedulePass] {
          reqs: ["main"],
          mem_loc: { devs: [{name: "CMX"}] },
          mem_KiB: 128,
          alignment: 16,
          xfer_loc: { devs: [{name: "DMA"}] }
        }
      }
    }, {
      name: "prune_refs"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass] {
          reqs: ["program"]
        }
      }
    }, {
      name: "place_program"
      pass: {
        [type.vertex.ai/vertexai.tile.codegen.proto.MemoryPlacementPass] {
          reqs: ["program"]
          locs: [{ devs: [{name: "DRAM"}] }]
          alignment: 4
        }
      }
    }
  ]
  )";
  return ParseProtoText<proto::Stage>(cfg_tmpl);
}

static std::map<std::string, std::vector<float>> GenerateMatrix(const std::vector<size_t>& dim,
                                                                std::vector<std::string> vars) {
  size_t size = 1;
  for (size_t i = 0; i < dim.size(); ++i) {
    size *= dim[i];
  }
  // We don't care the contents in data
  std::map<std::string, std::vector<float>> data;
  for (const auto& var : vars) {
    data[var] = std::vector<float>(size);
  }
  return data;
}

static std::vector<float> GenerateExpected(const std::vector<size_t>& dim, size_t load_dim) {
  size_t size = 1;
  for (size_t i = 0; i < dim.size(); ++i) {
    size *= dim[i];
  }
  std::vector<float> result(size);
  size_t idx = 0;
  for (size_t w = 0; w < dim[0]; ++w) {
    for (size_t x = 0; x < dim[1]; ++x) {
      for (size_t y = 0; y < dim[2]; ++y) {
        for (size_t z = 0; z < dim[3]; ++z) {
          if (load_dim == 0) {
            result[idx++] = w;
          } else if (load_dim == 1) {
            result[idx++] = x;
          } else if (load_dim == 2) {
            result[idx++] = y;
          } else if (load_dim == 3) {
            result[idx++] = z;
          } else {
            throw_with_trace(std::runtime_error("Invalid load_dim in GenerateExpected"));
          }
        }
      }
    }
  }
  return result;
}

TEST(LoadIndexTest, SimpleIndex) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B) {
      B = index(A, 2);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::INT32, {4, 4, 4, 4}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  std::string expected_stripe = R"**(0: #program #total_macs=0
    block []:1 ( // load_index_simple
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] i32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB, E(4, 4, 4, 4):1 KiB
          out B[0, 0, 0, 0]:assign i32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB, E(4, 4, 4, 4):1 KiB
      ) {
        0: #eltwise #eltwise_index #kernel 
        block [i1:4, i2:4, i3:4, i4:4]:256 ( // kernel_0(A)
            // B = index(A, _T0)
            out B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(64, 16, 4, 1):4 B, E(4, 4, 4, 4):1 KiB
        ) {
          0: $B = load_index(i3)
          1: B = store($B)
        }
      }
    }
  )**";

  std::string actual_stripe = to_string(*program->entry);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::CompilerState state(program);
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(1, "After stripe optimization: " << *program->entry);

  std::vector<size_t> dim = {4, 4, 4, 4};
  std::vector<std::string> vars = {"A", "B"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_result = GenerateExpected(dim, 2);
  ExecuteProgram(*program->entry, &data);
  EXPECT_THAT(data["B"], Eq(expected_result));
}

TEST(LoadIndexTest, AffineIndex) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_affine";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B) {
      B = index(A, 2);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {8, 8, 256, 8}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::INT32, {8, 8, 256, 8}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  std::string expected_stripe = R"**(0: #program #total_macs=0
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] i32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB, E(8, 8, 256, 8):512 KiB
          out B[0, 0, 0, 0]:assign i32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB, E(8, 8, 256, 8):512 KiB
      ) {
        0: #eltwise #eltwise_index #kernel 
        block [i1:8, i2:8, i3:256, i4:8]:131072 ( // kernel_0(A)
            // B = index(A, _T0)
            out B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(16384, 2048, 8, 1):4 B, E(8, 8, 256, 8):512 KiB
        ) {
          0: $B = load_index(i3)
          1: B = store($B)
        }
      }    
    }
  )**";

  std::string actual_stripe = to_string(*program->entry);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::CompilerState state(program);
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(1, "After stripe optimization: " << *program->entry);

  std::vector<size_t> dim = {8, 8, 256, 8};
  std::vector<std::string> vars = {"A", "B"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_result = GenerateExpected(dim, 2);
  ExecuteProgram(*program->entry, &data);
  EXPECT_THAT(data["B"], Eq(expected_result));
}

TEST(LoadIndexTest, MultiLoadIndex) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_affine";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B, C, D) {
      B = index(A, 2);
      C = index(A, 0);
      D = index(A, 1);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 256, 8}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::INT32, {4, 4, 256, 8}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::INT32, {4, 4, 256, 8}));
  runinfo.output_shapes.emplace("D", SimpleShape(DataType::INT32, {4, 4, 256, 8}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  std::string expected_stripe = R"**(0: #program #total_macs=0
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 C[0, 0, 0, 0] i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 D[0, 0, 0, 0] i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out B[0, 0, 0, 0]:assign i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out C[0, 0, 0, 0]:assign i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out D[0, 0, 0, 0]:assign i32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
      ) {
        0: #eltwise #eltwise_index #kernel 
        block [i1:4, i2:4, i3:256, i4:8]:32768 ( // kernel_0(A)
            // B = index(A, _T0)
            out B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(8192, 2048, 8, 1):4 B, E(4, 4, 256, 8):128 KiB
        ) {
          0: $B = load_index(i3)
          1: B = store($B)
        }
        1: #eltwise #eltwise_index #kernel 
        block [i1:4, i2:4, i3:256, i4:8]:32768 ( // kernel_1(A)
            // C = index(A, _T2)
            out C[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(8192, 2048, 8, 1):4 B, E(4, 4, 256, 8):128 KiB
        ) {
          0: $C = load_index(i1)
          1: C = store($C)
        }
        2: #eltwise #eltwise_index #kernel 
        block [i1:4, i2:4, i3:256, i4:8]:32768 ( // kernel_2(A)
            // D = index(A, _T4)
            out D[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(8192, 2048, 8, 1):4 B, E(4, 4, 256, 8):128 KiB
        ) {
          0: $D = load_index(i2)
          1: D = store($D)
        }
      }
    }
  )**";

  std::string actual_stripe = to_string(*program->entry);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::CompilerState state(program);
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(1, "After stripe optimization: " << *program->entry);

  std::vector<size_t> dim = {4, 4, 256, 8};
  std::vector<std::string> vars = {"A", "B", "C", "D"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_B = GenerateExpected(dim, 2);
  auto expected_C = GenerateExpected(dim, 0);
  auto expected_D = GenerateExpected(dim, 1);
  ExecuteProgram(*program->entry, &data);
  EXPECT_THAT(data["B"], Eq(expected_B));
  EXPECT_THAT(data["C"], Eq(expected_C));
  EXPECT_THAT(data["D"], Eq(expected_D));
}

TEST(LoadIndexTest, FuseIndex) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_affine";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B, C) {
      B = index(A, 2);
      C = A + B;
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 4, 8}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::INT32, {4, 4, 4, 8}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {4, 4, 4, 8}));
  auto program = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *program->entry);

  std::string expected_stripe = R"**(0: #program #total_macs=0
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] i32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
        #user none new@0x00000000 C[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB, E(4, 4, 4, 8):2 KiB
          out B[0, 0, 0, 0]:assign i32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB, E(4, 4, 4, 8):2 KiB
          out C[0, 0, 0, 0]:assign fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB, E(4, 4, 4, 8):2 KiB
      ) {
        0: #eltwise #eltwise_index #kernel 
        block [i1:4, i2:4, i3:4, i4:8]:512 ( // kernel_0(A)
            // B = index(A, _T0)
            out B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(128, 32, 8, 1):4 B, E(4, 4, 4, 8):2 KiB
        ) {
          0: $B = load_index(i3)
          1: B = store($B)
        }
        1: #eltwise #eltwise_add #kernel 
        block [i1:4, i2:4, i3:4, i4:8]:512 ( // kernel_1(A,B)
            // C = add(A, B)
            #eltwise_add in A[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(128, 32, 8, 1):4 B, E(4, 4, 4, 8):2 KiB
            #eltwise_add in B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(128, 32, 8, 1):4 B, E(4, 4, 4, 8):2 KiB
            out C[i1, i2, i3, i4] fp32:I(1, 1, 1, 1):(128, 32, 8, 1):4 B, E(4, 4, 4, 8):2 KiB
        ) {
          0: $A = load(A)
          1: $B = load(B)
          2: $C = add($A, $B)
          3: C = store($C)
        }
      }
    }
  )**";

  std::string actual_stripe = to_string(*program->entry);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto stage = GenerateStage();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::CompilerState state(program);
  codegen::Optimize(&state, stage.passes(), options);
  IVLOG(1, "After stripe optimization: " << *program->entry);

  std::vector<size_t> dim = {4, 4, 4, 8};
  std::vector<std::string> vars = {"A", "B", "C"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_B = GenerateExpected(dim, 2);
  const auto& data_A = data["A"];
  size_t size = dim[0] * dim[1] * dim[2] * dim[3];
  std::vector<float> expected_C(size);
  for (size_t i = 0; i < data_A.size(); ++i) {
    expected_C[i] = data_A[i] + expected_B[i];
  }
  ExecuteProgram(*program->entry, &data);
  EXPECT_THAT(data["B"], Eq(expected_B));
  EXPECT_THAT(data["C"], Eq(expected_C));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
