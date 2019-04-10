// Copyright 2018, Intel Corporation

#include "tile/codegen/driver.h"

#include <boost/format.hpp>

#include "base/config/config.h"
#include "base/util/throw.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/autotile.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/codec.h"
#include "tile/codegen/cstr_reduction.h"
#include "tile/codegen/dce.h"
#include "tile/codegen/deps.h"
#include "tile/codegen/emitc.h"
#include "tile/codegen/fuse.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/mem_rebase.h"
#include "tile/codegen/package.h"
#include "tile/codegen/pad.h"
#include "tile/codegen/partition.h"
#include "tile/codegen/placer.h"
#include "tile/codegen/reg_cache.h"
#include "tile/codegen/scalarize.h"
#include "tile/codegen/schedule.h"
#include "tile/codegen/thread_inner.h"
#include "tile/codegen/tidy.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/transpose.h"
#include "tile/codegen/unroll.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

void DumpProgram(const Block& program,            //
                 const OptimizeOptions& options,  //
                 const std::string& name,         //
                 size_t counter) {
  if (options.dump_passes || options.dump_code) {
    boost::filesystem::create_directory(options.dbg_dir);
    if (options.dump_passes) {
      auto filename = str(boost::format("%02zu_%s.txt") % counter % name);
      auto path = (options.dbg_dir / filename).string();
      std::ofstream fout(path);
      fout << program << std::endl;
    }
    if (options.dump_code) {
      auto filename = str(boost::format("%02zu_%s.c") % counter % name);
      auto path = (options.dbg_dir / filename).string();
      std::ofstream fout(path);
      fout << EmitC(program);
    }
  }
}

void ValidateBlock(Block* root) {
  RunOnBlocks(  //
      root, {},
      [&](auto map, auto block) {
        for (const auto& ref : block->refs) {
          if (ref.dir == RefDir::None && !ref.from.empty()) {
            throw_with_trace(std::runtime_error(
                str(boost::format("ref.dir == RefDir::None && !ref.from.empty(). ref: %1% in block: %2%") % ref.into() %
                    block->name)));
          }
          if (ref.from.empty() && ref.dir != RefDir::None) {
            throw_with_trace(std::runtime_error(
                str(boost::format("ref.from.empty() && ref.dir != RefDir::None. ref: %1% in block: %2%") % ref.into() %
                    block->name)));
          }
        }
      },
      true);
}

class ConfigsRegistry {
 public:
  static ConfigsRegistry* Instance() {
    static ConfigsRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, const std::string& cfg_bytes) {  //
    registry_[name] = cfg_bytes;
  }

  proto::Config Resolve(const std::string& name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      throw_with_trace(std::runtime_error(str(boost::format("Could not find config: %s") % name)));
    }
    return ParseConfig<proto::Config>(it->second);
  }

 private:
  std::unordered_map<std::string, std::string> registry_;
};

}  // namespace

void Optimize(Block* block, const Passes& passes, const OptimizeOptions& options) {
  size_t counter = 0;
  DumpProgram(*block, options, "initial", counter++);
  for (const auto& pass : passes) {
    IVLOG(1, "Optimization Pass " << pass.name());
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
      case proto::Pass::kTranspose:
        TransposePass(block, pass.transpose());
        break;
      case proto::Pass::kPartitionCompute:
        PartitionComputePass(block, pass.partition_compute());
        break;
      case proto::Pass::kPartitionMemory:
        PartitionMemoryPass(block, pass.partition_memory());
        break;
      case proto::Pass::kUnroll:
        UnrollPass(block, pass.unroll());
        break;
      case proto::Pass::kPruneIdxs:
        PruneIndexesPass(block, pass.prune_idxs());
        break;
      case proto::Pass::kPruneRefs:
        PruneRefinementsPass(block, pass.prune_refs());
        break;
      case proto::Pass::kThreadInner:
        ThreadInnerPass(block, pass.thread_inner());
        break;
      case proto::Pass::kAssignCodec:
        AssignCodecPass(block, pass.assign_codec());
        break;
      case proto::Pass::kMemRebase:
        MemRebasePass(block, pass.mem_rebase());
        break;
      case proto::Pass::kLightCstrReduction:
        LightCstrReductionPass(block, pass.light_cstr_reduction());
        break;
      case proto::Pass::kIlpCstrReduction:
        IlpCstrReductionPass(block, pass.ilp_cstr_reduction());
        break;
      case proto::Pass::kDeadCodeElimination:
        DeadCodeEliminationPass(block, pass.dead_code_elimination(), false);
        break;
      case proto::Pass::kPackageBlocks:
        PackagePass(block, pass.package_blocks());
        break;
      case proto::Pass::kPad:
        PadPass(block, pass.pad());
        break;
      case proto::Pass::kRegisterCache:
        RegisterCachePass(block, pass.register_cache());
        break;
      default:
        throw_with_trace(std::runtime_error(str(boost::format("Unsupported pass: %1%") % pass.name())));
    }
    DumpProgram(*block, options, pass.name(), counter++);
    ValidateBlock(block);
  }
}

void Configs::Register(const std::string& name, const std::string& pb_bytes) {
  ConfigsRegistry::Instance()->Register(name, pb_bytes);
}

proto::Config Configs::Resolve(const std::string& name) {  //
  return ConfigsRegistry::Instance()->Resolve(name);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
