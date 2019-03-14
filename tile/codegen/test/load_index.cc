// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "base/util/stream_container.h"
#include "testing/matchers.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/driver.h"
#include "tile/codegen/fuse.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/scalarize.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/semprinter.h"
#include "tile/lang/simplifier.h"
#include "tile/ocl_exec/emitsem.h"
#include "tile/stripe/stripe.h"
#include "tile/stripe/stripe.pb.h"

using ::testing::ContainerEq;
using ::testing::Eq;
using ::testing::LinesEq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT

template <typename P>
P ParseProtoText(const std::string& txt) {
  P proto;
  google::protobuf::TextFormat::ParseFromString(txt, &proto);
  return proto;
}

std::string EraseSpace(const std::string& src) {
  std::string str = src;
  int i = 0;
  while (i < str.size()) {
    int j = i;
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

proto::Config GenerateCFG() {
  auto cfg_tmpl = R"(
    arch: "test"
    passes: { name: "loc_prog" locate_memory: { reqs: ["program"] loc: { name: "DRAM" } } }
    passes: { name: "loc_main" locate_memory: { reqs: ["main"] loc: { name: "DRAM" } } }
    passes: { name: "loc_initial", locate_block: { reqs: ["kernel"], loc: { name: "PPE" } } }
    passes: {
            name: "stencil_mac",
            stencil: {
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
    passes: { name: "fuse_dpu_add", fusion: { a_reqs: ["dpu"], b_reqs: ["eltwise_add"], fused_set: ["dpu"] } }
    passes: { name: "fuse_dpu_mul", fusion: { a_reqs: ["dpu"], b_reqs: ["eltwise_mul"], fused_set: ["dpu"] } }
    passes: { name: "fuse_dpu_shift", fusion: { a_reqs: ["dpu"], b_reqs: ["eltwise_bit_right"], fused_set: ["dpu"] } }
    passes: { name: "fuse_dpu_zelu", fusion: { a_reqs: ["dpu"], b_reqs: ["eltwise_zelu"], fused_set: ["dpu"] } }
    passes: { name: "fuse_dpu", fusion: { parent_reqs: ["dpu"], a_reqs: ["eltwise"], fused_set: ["dpu_fusion"] } }
    passes: { name: "light_cstr_reduction", light_cstr_reduction: { reqs: ["all"] } }
    passes: { name: "localize_main", localize: { reqs: ["main"] } }
    passes: { name: "scalarize_main", scalarize: { reqs: ["main"] } }
    passes: {   
            name: "fit_into_cmx",
            autotile: {
                reqs: ["kernel"],
                outer_set: ["fit_part"],
                skip_1d: true,
                only_po2: true,
                max_total_size : 4096,
                input_cost: 1.0,
                output_cost: 1.0,
                copy_tags: true,
                clear_outer: true,
                acc_idxs: false
            }
    }
    passes: { name: "loc_dpu", locate_block: { reqs: ["dpu"], loc: { name: "DPU" } } }
    passes: { name: "loc_mpe", locate_inner_block: { reqs: ["dpu"], loc: { name: "MPE" } } }
    passes: { name: "loc_dpu_mem", locate_memory: { reqs: ["dpu"], loc: { name: "ACC" } } }
    passes: { name: "loc_dpu_fusion_mem", locate_memory: { reqs: ["dpu_fusion"], loc: { name: "ACC" } } }
    passes: { name: "cache_dpu_in", cache: { reqs: ["dpu"], dirs: [ In ], mem_loc: { name: "MRM" }, xfer_loc: { name: "IDU" } } }
    passes: { name: "cache_dpu_out", cache: { reqs: ["dpu"], dirs: [ Out ], mem_loc: { name: "ACC" }, xfer_loc: { name: "ODU" } } }
    passes: { name: "loc_dpu_fused", locate_block: { reqs: ["eltwise", "dpu_fusion"], loc: { name: "MPE" } } }
    passes: { name: "loc_dpu_eltwise", locate_inner_block: { reqs: ["dpu"], inner_reqs: ["eltwise"], loc: { name: "MPE" } } }
    passes: {
            name: "schedule_main",
            schedule: {
                reqs: ["main"],
                mem_loc: { name: "CMX" },
                mem_KiB: 128,
                alignment: 16,
                xfer_loc: { name: "DMA" }
            }
    },
    passes: { name: "prune_refs", prune_refs: { reqs: ["program"] } },
    passes: { name: "place_program", memory_placement: { reqs: ["program"], locs: [{ name: "DRAM" }], alignment: 4 } }
  )";
  auto cfg = ParseProtoText<proto::Config>(cfg_tmpl);
  // str(boost::format(cfg_tmpl) % num_banks % procs_per_bank % bank_size % bank_size_KiB));
  return cfg;
}

TEST(LoadIndexTest, SimpleIndex) {
  auto verbose = std::getenv("VERBOSE");
  if (verbose && strlen(verbose) > 0) {
    el::Loggers::setVerboseLevel(std::stoi(verbose));
  }

  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B) {
      B = index(A, 2);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_stripe = R"**(0: #program 
    block []:1 ( // load_index_simple
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB, E(4, 4, 4, 4):1 KiB
          out B[0, 0, 0, 0]:assign fp32:I(4, 4, 4, 4):(64, 16, 4, 1):1 KiB, E(4, 4, 4, 4):1 KiB
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

  std::string actual_stripe = to_string(*stripe.program);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto cfg = GenerateCFG();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  options.dump_code = false;
  options.dump_passes = false;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::Optimize(stripe.program.get(), cfg.passes(), options);
  IVLOG(1, "After stripe optimization: " << *stripe.program);

  std::string expected_kernel0 = R"**(void kernel_1(int* d2_B^0)
    {
      int d3_i3 = (get_group_id(0) >> 8);
      int d4_i4 = ((get_group_id(0) >> 6) & 3);
      int d4_i3 = ((get_group_id(0) >> 4) & 3);
      int d4_i2 = ((get_group_id(0) >> 2) & 3);
      int d4_i1 = (get_group_id(0) & 3);
      long s_4_B = ((4 * d3_i3) + d4_i3);
      d2_B^0[((((((((256 * d3_i1) + (64 * d3_i2)) + (16 * d3_i3)) + (4 * d3_i4)) + (64 * d4_i1)) + (16 * d4_i2)) + (4 * d4_i3)) + d4_i4)] = s_4_B;
    }
  )**";
  expected_kernel0 = EraseSpace(expected_kernel0);

  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe.program);
  lang::Simplify(emit.kernels_.kernels);

  sem::Print actual_kernel0(*(emit.kernels_.kernels[0].kfunc));
  auto actual_kernel0_str = actual_kernel0.str();
  IVLOG(1, actual_kernel0_str);
  actual_kernel0_str = EraseSpace(actual_kernel0_str);
  EXPECT_THAT(actual_kernel0_str, LinesEq(expected_kernel0));
}

TEST(LoadIndexTest, AffineIndex) {
  auto verbose = std::getenv("VERBOSE");
  if (verbose && strlen(verbose) > 0) {
    el::Loggers::setVerboseLevel(std::stoi(verbose));
  }

  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_affine";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B) {
      B = index(A, 2);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {16, 16, 1024, 16}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {16, 16, 1024, 16}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_stripe = R"**(0: #program
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(16, 16, 1024, 16):(262144, 16384, 16, 1):16384 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] fp32:I(16, 16, 1024, 16):(262144, 16384, 16, 1):16384 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(16, 16, 1024, 16):(262144, 16384, 16, 1):16384 KiB, E(16, 16, 1024, 16):16384 KiB
          out B[0, 0, 0, 0]:assign fp32:I(16, 16, 1024, 16):(262144, 16384, 16, 1):16384 KiB, E(16, 16, 1024, 16):16384 KiB
      ) {
        0: #eltwise #eltwise_index #kernel 
        block [i1:16, i2:16, i3:1024, i4:16]:4194304 ( // kernel_0(A)
            // B = index(A, _T0)
            out B[i1, i2, i3, i4] i32:I(1, 1, 1, 1):(262144, 16384, 16, 1):4 B, E(16, 16, 1024, 16):16384 KiB
        ) {
          0: $B = load_index(i3)
          1: B = store($B)
        }
      }
    }
  )**";

  std::string actual_stripe = to_string(*stripe.program);
  actual_stripe = EraseSpace(actual_stripe);
  expected_stripe = EraseSpace(expected_stripe);

  EXPECT_THAT(actual_stripe, LinesEq(expected_stripe));

  auto cfg = GenerateCFG();
  auto dbg_dir = std::getenv("DBG_DIR");
  OptimizeOptions options;
  options.dump_code = false;
  options.dump_passes = false;
  if (dbg_dir) {
    options.dump_passes = true;
    options.dbg_dir = dbg_dir;
    IVLOG(1, "Writing passes to: " << dbg_dir);
  }
  codegen::Optimize(stripe.program.get(), cfg.passes(), options);
  IVLOG(1, "After stripe optimization: " << *stripe.program);

  std::string expected_kernel0 = R"**(void kernel_1(int* d2_B^0)
    {
      int d3_i3 = (get_group_id(0) >> 10);
      int d4_i4 = ((get_group_id(0) >> 6) & 15);
      int d4_i3 = ((get_group_id(0) >> 1) & 31);
      int d4_i1 = (get_group_id(0) & 1);
      long s_4_B = ((32 * d3_i3) + d4_i3);
      d2_B^0[(((512 * d4_i1) + (16 * d4_i3)) + d4_i4)] = s_4_B;
    }
  )**";
  expected_kernel0 = EraseSpace(expected_kernel0);

  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe.program);
  lang::Simplify(emit.kernels_.kernels);

  sem::Print actual_kernel0(*(emit.kernels_.kernels[0].kfunc));
  auto actual_kernel0_str = actual_kernel0.str();
  IVLOG(1, actual_kernel0_str);
  actual_kernel0_str = EraseSpace(actual_kernel0_str);
  EXPECT_THAT(actual_kernel0_str, LinesEq(expected_kernel0));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
