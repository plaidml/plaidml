// Copyright 2018, Intel Corporation

#include "tile/codegen/driver.h"

#include "tile/codegen/autotile.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/deps.h"
#include "tile/codegen/fuse.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/placer.h"
#include "tile/codegen/scalarize.h"
#include "tile/codegen/schedule.h"
#include "tile/codegen/tile.h"

namespace vertexai {
namespace tile {
namespace codegen {

namespace {

void DumpProgram(const stripe::Block& program,    //
                 const OptimizeOptions& options,  //
                 const std::string& name,         //
                 size_t counter) {
  if (options.dump_passes) {
    boost::filesystem::create_directory(options.dbg_dir);
    auto filename = printstring("%02zu_%s.txt", counter, name.c_str());
    auto path = (options.dbg_dir / filename).string();
    std::ofstream fout(path);
    fout << program << std::endl;
  }
}

}  // namespace

void Optimize(stripe::Block* block, const proto::Config& cfg, const OptimizeOptions& options) {
  size_t counter = 0;
  DumpProgram(*block, options, "initial", counter++);
  for (const auto& pass : cfg.passes()) {
    IVLOG(2, "Optimization Pass " << pass.name());
    switch (pass.pass_case()) {
      case proto::Pass::kCache:
        CachePass(block, pass.cache());
        break;
      case proto::Pass::kComputeDeps:
        ComputeDepsPass(block, pass.compute_deps());
        break;
      case proto::Pass::kFusion:
        FusionPass(block, pass.fusion());
        break;
      case proto::Pass::kLocalize:
        LocalizePass(block, pass.localize());
        break;
      case proto::Pass::kLocateBlock:
        LocateBlockPass(block, pass.locate_block());
        break;
      case proto::Pass::kLocateInnerBlock:
        LocateInnerBlockPass(block, pass.locate_inner_block());
        break;
      case proto::Pass::kLocateMemory:
        LocateMemoryPass(block, pass.locate_memory());
        break;
      case proto::Pass::kMemoryPlacement:
        MemPlacementPass(block, pass.memory_placement());
        break;
      case proto::Pass::kScalarize:
        ScalarizePass(block, pass.scalarize());
        break;
      case proto::Pass::kSchedule:
        SchedulePass(block, pass.schedule());
        break;
      case proto::Pass::kStencil:
        StencilPass(block, pass.stencil());
        break;
      case proto::Pass::kAutotile:
        AutotilePass(block, pass.autotile());
        break;
      case proto::Pass::kCustom:
        // TODO
        break;
      default:
        break;
    }
    DumpProgram(*block, options, pass.name(), counter++);
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
