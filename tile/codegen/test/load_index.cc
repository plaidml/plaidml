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

std::map<std::string, std::vector<float>> GenerateMatrix(const size_t* dim, std::vector<std::string> vars) {
  size_t size = dim[0] * dim[1] * dim[2] * dim[3];
  // We don't care the contents in data
  std::map<std::string, std::vector<float>> data;
  for (const auto& var : vars) {
    data[var] = std::vector<float>(size);
  }
  return data;
}

std::vector<float> GenerateExpected(const size_t* dim, size_t load_dim) {
  size_t size = dim[0] * dim[1] * dim[2] * dim[3];
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

  size_t dim[4] = {4, 4, 4, 4};
  std::vector<std::string> vars = {"A", "B"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_result = GenerateExpected(dim, 2);
  ExecuteProgram(*stripe.program, &data);
  EXPECT_THAT(data["B"], Eq(expected_result));
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
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {8, 8, 256, 8}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {8, 8, 256, 8}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_stripe = R"**(0: #program
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB, E(8, 8, 256, 8):512 KiB
          out B[0, 0, 0, 0]:assign fp32:I(8, 8, 256, 8):(16384, 2048, 8, 1):512 KiB, E(8, 8, 256, 8):512 KiB
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

  size_t dim[4] = {8, 8, 256, 8};
  std::vector<std::string> vars = {"A", "B"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_result = GenerateExpected(dim, 2);
  ExecuteProgram(*stripe.program, &data);
  EXPECT_THAT(data["B"], Eq(expected_result));
}

TEST(LoadIndexTest, MultiLoadIndex) {
  auto verbose = std::getenv("VERBOSE");
  if (verbose && strlen(verbose) > 0) {
    el::Loggers::setVerboseLevel(std::stoi(verbose));
  }

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
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {4, 4, 256, 8}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {4, 4, 256, 8}));
  runinfo.output_shapes.emplace("D", SimpleShape(DataType::FLOAT32, {4, 4, 256, 8}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_stripe = R"**(0: #program
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 C[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
        #user none new@0x00000000 D[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out B[0, 0, 0, 0]:assign fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out C[0, 0, 0, 0]:assign fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
          out D[0, 0, 0, 0]:assign fp32:I(4, 4, 256, 8):(8192, 2048, 8, 1):128 KiB, E(4, 4, 256, 8):128 KiB
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

  size_t dim[4] = {4, 4, 256, 8};
  std::vector<std::string> vars = {"A", "B", "C", "D"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_B = GenerateExpected(dim, 2);
  auto expected_C = GenerateExpected(dim, 0);
  auto expected_D = GenerateExpected(dim, 1);
  ExecuteProgram(*stripe.program, &data);
  EXPECT_THAT(data["B"], Eq(expected_B));
  EXPECT_THAT(data["C"], Eq(expected_C));
  EXPECT_THAT(data["D"], Eq(expected_D));
}

TEST(LoadIndexTest, FuseIndex) {
  auto verbose = std::getenv("VERBOSE");
  if (verbose && strlen(verbose) > 0) {
    el::Loggers::setVerboseLevel(std::stoi(verbose));
  }

  lang::RunInfo runinfo;
  runinfo.program_name = "load_index_affine";
  runinfo.code = R"***(
    function (A[W, X, Y, Z]) -> (B, C) {
      B = index(A, 2);
      C = A + B;
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {4, 4, 4, 8}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {4, 4, 4, 8}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {4, 4, 4, 8}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, "Before stripe optimization: " << *stripe.program);

  std::string expected_stripe = R"**(0: #program
    block []:1 ( // load_index_affine
        #user none new@0x00000000 A[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
        #user none new@0x00000000 B[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
        #user none new@0x00000000 C[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB
    ) {
      0: #main 
      block []:1 ( // main
          in A[0, 0, 0, 0] fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB, E(4, 4, 4, 8):2 KiB
          out B[0, 0, 0, 0]:assign fp32:I(4, 4, 4, 8):(128, 32, 8, 1):2 KiB, E(4, 4, 4, 8):2 KiB
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

  size_t dim[4] = {4, 4, 4, 8};
  std::vector<std::string> vars = {"A", "B", "C"};
  auto data = GenerateMatrix(dim, vars);
  auto expected_B = GenerateExpected(dim, 2);
  const auto& data_A = data["A"];
  size_t size = dim[0] * dim[1] * dim[2] * dim[3];
  std::vector<float> expected_C(size);
  for (size_t i = 0; i < data_A.size(); ++i) {
    expected_C[i] = data_A[i] + expected_B[i];
  }
  ExecuteProgram(*stripe.program, &data);
  EXPECT_THAT(data["B"], Eq(expected_B));
  EXPECT_THAT(data["C"], Eq(expected_C));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
