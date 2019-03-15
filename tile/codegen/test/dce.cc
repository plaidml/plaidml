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

static proto::Config GenerateCFG() {
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
    passes: { name: "fuse_eltwise", fusion: { a_reqs: ["eltwise"], b_reqs: ["eltwise"], fused_set: ["dpu"] } }
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
                max_total_size : 1024,
                input_cost: 1.0,
                output_cost: 1.0,
                copy_tags: true,
                clear_outer: true,
                acc_idxs: false
            }
    }
    passes: { name: "dead_code_elimination", dead_code_elimination: { reqs: ["all"] } }
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
  return cfg;
}

TEST(DCETest, RemoveBlock) {
  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_simple";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B) {
      C = A + A;
      B = index(C, 2);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {4, 4, 4, 4}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_before = R"**(0: #program 
  )**";

  std::string actual_before = EraseSpace(to_string(*stripe.program));
  expected_before = EraseSpace(expected_before);

  EXPECT_THAT(actual_before, LinesEq(expected_before));

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

  std::string expected_after = R"**(0: #program
  )**";

  std::string actual_after = EraseSpace(to_string(*stripe.program));
  expected_after = EraseSpace(expected_after);

  EXPECT_THAT(actual_after, LinesEq(expected_after));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
