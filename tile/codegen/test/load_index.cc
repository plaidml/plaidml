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
    passes: { name: "loc_proc"
      locate_block: {
        reqs: ["kernel"]
        loc: {
          name: "PROC"
          unit: {
            terms: { key: "#bank" value: %2% }
            terms: { key: "#proc" value: 1 }
          }
        }
      }
    }
    passes: { name: "partition_memory"
      partition_memory: {
        reqs: ["kernel"]
        num_parts: %1%
        set_tags: ["bank_part"]
        idx_tag: "bank"
      }
    }
    passes: { name: "unroll_bank_parts"
      unroll: {
        reqs: ["bank_part"]
        expand_idx: "#bank"
        part_name: "bank"
        make_views: true
      }
    }
    passes: { name: "fit_into_mem"
      autotile: {
        reqs: ["kernel"]
        outer_set: ["fit_part"]
        skip_1d: true
        only_po2: true
        max_total_size : %3%
        input_cost: 1.0
        output_cost: 1.0
        copy_tags: true
      }
    }
    passes: { name: "partition_compute"
      partition_compute: {
        reqs: ["kernel"]
        num_parts: %2%
        set_tags: ["compute_part"]
        idx_tag: "proc"
      }
    }
    passes: { name: "unroll_compute_parts"
      unroll: {
        reqs: ["compute_part"]
        expand_idx: "#proc"
        part_name: "proc"
      }
    }
    passes: { name: "schedule"
      schedule: {
        reqs: ["main"]
        mem_loc: { name: "SRAM" }
        mem_KiB: %4%
        alignment: 16
        xfer_loc: { name: "DMA" }
      }
    }
    passes: { name: "prune_refs" prune_refs: { reqs: ["program"] } }
  )";
  size_t num_banks = 2;
  size_t num_procs = 4;
  size_t procs_per_bank = num_procs / num_banks;
  size_t bank_size_KiB = 192;
  size_t bank_size = bank_size_KiB * 1024;
  auto cfg = ParseProtoText<proto::Config>(
      str(boost::format(cfg_tmpl) % num_banks % procs_per_bank % bank_size % bank_size_KiB));
  return cfg;
}

TEST(LoadIndexTest, SimpleIndex) {
  auto verbose = std::getenv("VERBOSE");
  if (verbose && strlen(verbose) > 0) {
    el::Loggers::setVerboseLevel(std::stoi(verbose));
  }

  lang::RunInfo runinfo;
  runinfo.program_name = "simple_fuse";
  runinfo.code = R"***(
    function (A[X, Y]) -> (B) {
      B = index(A, 1);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {16, 16}));
  runinfo.output_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {16, 16}));
  auto stripe = GenerateStripe(runinfo);
  IVLOG(1, *stripe.program);

  std::string expected_stripe = R"**(0: #program 
    block []:1 ( // simple_fuse
      #user none new@0x00000000 A[0, 0] fp32:I(16, 16):(16, 1):1 KiB
      #user none new@0x00000000 B[0, 0] fp32:I(16, 16):(16, 1):1 KiB
    ) {
      0: #main 
        block []:1 ( // main
          in A[0, 0] fp32:I(16, 16):(16, 1):1 KiB, E(16, 16):1 KiB
          out B[0, 0]:assign fp32:I(16, 16):(16, 1):1 KiB, E(16, 16):1 KiB
        ) {
            0: #eltwise #eltwise_index #kernel 
              block [i1:16, i2:16]:256 ( // kernel_0(A)
                // B = index(A, _T0)
                #eltwise_index in A[i1, i2] fp32:I(1, 1):(16, 1):4 B, E(16, 16):1 KiB
                out B[i1, i2] i32:I(1, 1):(16, 1):4 B, E(16, 16):1 KiB
              ) {
                0: $B = load_index(i2)
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
  IVLOG(1, *stripe.program);

  std::string expected_kernel0 = R"**(void kernel_1(int* d2_B%bank_i1_0^0)
  {
    for(int d4_i2 = 0; d4_i2 < 16; d4_i2 += 1)
    {
      for(int d4_i1 = 0; d4_i1 < 8; d4_i1 += 1)
      {
        d2_B%bank_i1_0^0[((16 * d4_i1) + d4_i2)] = d4_i2;
      }
    }
  }
  )**";
  expected_kernel0 = EraseSpace(expected_kernel0);

  std::string expected_kernel1 = R"**(void kernel_2(int* d2_B%bank_i1_1^0)
  {
    for(int d4_i2 = 0; d4_i2 < 16; d4_i2 += 1)
    {
      for(int d4_i1 = 0; d4_i1 < 8; d4_i1 += 1)
      {
        d2_B%bank_i1_1^0[((128 + (16 * d4_i1)) + d4_i2)] = d4_i2;
      }
    }
  }
  )**";
  expected_kernel1 = EraseSpace(expected_kernel1);

  codegen::SemtreeEmitter emit(codegen::AliasMap{}, 256);
  emit.Visit(*stripe.program);
  lang::Simplify(emit.kernels_.kernels);

  sem::Print actual_kernel0(*(emit.kernels_.kernels[0].kfunc));
  auto actual_kernel0_str = actual_kernel0.str();
  IVLOG(4, actual_kernel0_str);
  actual_kernel0_str = EraseSpace(actual_kernel0_str);
  EXPECT_THAT(actual_kernel0_str, LinesEq(expected_kernel0));

  sem::Print actual_kernel1(*(emit.kernels_.kernels[1].kfunc));
  auto actual_kernel1_str = actual_kernel1.str();
  IVLOG(4, actual_kernel1_str);
  actual_kernel1_str = EraseSpace(actual_kernel1_str);
  EXPECT_THAT(actual_kernel1_str, LinesEq(expected_kernel1));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
