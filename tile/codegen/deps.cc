// Copyright 2018, Intel Corporation

#include "tile/codegen/deps.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include <boost/format.hpp>

#include "base/util/throw.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct Tracker {
  // Tracks the state of a buffer as it's operated on by the Block's Statements.
  struct BufferInfo {
    // Keep track of a list of all currently active writers.
    // TODO: remove writers from the list when all written memory has been overwritten by subsequent writers.
    std::unordered_map<StatementIt, AliasInfo> writers;
    std::unordered_set<StatementIt> readers;
  };

  // For each scalar: the Statement that wrote the scalar.
  std::unordered_map<std::string, StatementIt> scalars;

  // The dataflow dependencies of the current Statement being processed.
  std::set<StatementIt> dataflow_deps;

  // For each buffer (keyed by the buffer base name from the alias map):
  // information about the buffer's accessors (as of the current Statement being processed).
  std::unordered_map<std::string, BufferInfo> buffers;

  void WriteScalar(const Block& block, StatementIt it, const std::string& name) {
    // Note that scalars are SSA, so there will only ever be one writer per block.
    auto it_inserted = scalars.insert(std::make_pair(name, it));
    if (!it_inserted.second) {
      throw_with_trace(
          std::logic_error(str(boost::format("Scalar %s written multiple times in %s") % name % block.name)));
    }
  }

  void ReadScalar(const Block& block, const std::string& name) {
    auto it = scalars.find(name);
    if (it == scalars.end()) {
      throw_with_trace(
          std::logic_error(str(boost::format("Scalar %s read before it was written in %s") % name % block.name)));
    }
    dataflow_deps.insert(it->second);
  }

  void WriteBuffer(StatementIt it, const std::string& name, const AliasMap& alias_map) {
    IVLOG(4, boost::format("    WriterBuffer> name: %1%, it: %2%") % name % *it);

    const AliasInfo& alias_info = alias_map.at(name);
    BufferInfo& buffer_info = buffers[alias_info.base_name];

    for (const auto& item : buffer_info.writers) {
      const auto& writer = item.first;
      if (writer != it && AliasInfo::Compare(alias_info, item.second) != AliasType::None) {
        IVLOG(4, boost::format("      other writer: %1%") % *writer);
        // For two zero blocks, the order does not matter
        if (!ZeroBlock(*it) || !ZeroBlock(*writer)) {
          dataflow_deps.insert(writer);
        }
      }
    }

    for (StatementIt reader : buffer_info.readers) {
      // Writing a buffer we're also reading blocks on all readers that
      // aren't us, but otherwise looks like a normal write of a buffer
      // that something else is reading.
      if (reader != it) {
        IVLOG(4, boost::format("      other reader: %1%") % *reader);
        dataflow_deps.insert(reader);
      }
    }

    if (ZeroBlock(*it)) {
      // Remove the previous zero blocks for the same buffer
      // Alias info sometimes not accurate. As zero block is
      // always whole block rewritten, so we just need the same base name.
      for (auto wit : buffer_info.writers) {
        if (ZeroBlock(*wit.first) && (alias_info.base_name == wit.second.base_name ||
                                     AliasInfo::Compare(alias_info, wit.second) == AliasType::Exact)) {
          buffer_info.writers.erase(wit.first);
          break;
        }
      }
    }

    buffer_info.writers.emplace(it, alias_info);
    buffer_info.readers.clear();
  }

  void ReadBuffer(StatementIt it, const std::string& name, const AliasMap& alias_map) {
    IVLOG(4, boost::format("    ReadBuffer> name: %1%, it: %2%") % name % *it);

    const AliasInfo& alias_info = alias_map.at(name);
    BufferInfo& buffer_info = buffers[alias_info.base_name];

    if (buffer_info.writers.count(it)) {
      // Reading a buffer we're also writing doesn't do anything; we just track the write.
      return;
    }

    for (const auto& item : buffer_info.writers) {
      dataflow_deps.insert(item.first);
    }

    buffer_info.readers.insert(it);
  }
};

}  // namespace

void ComputeDepsForBlock(Block* block, const AliasMap& alias_map) {
  IVLOG(3, "ComputeDeps> " << block->name);
  Tracker tracker;
  std::unordered_map<StatementIt, std::set<StatementIt>> transitive_deps;
  for (auto it = block->stmts.begin(); it != block->stmts.end(); it++) {
    // Adjust the current scalar and buffer tracking structures and update dataflow_deps.
    switch ((*it)->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(*it);
        IVLOG(3, "  load: " << load);
        tracker.ReadBuffer(it, load->from, alias_map);
        tracker.WriteScalar(*block, it, load->into);
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(*it);
        IVLOG(3, "  store: " << store);
        tracker.ReadScalar(*block, store->from);
        tracker.WriteBuffer(it, store->into, alias_map);
      } break;
      case StmtKind::LoadIndex: {
        auto load_index = LoadIndex::Downcast(*it);
        IVLOG(3, "  loadIndex: " << load_index);
        tracker.WriteScalar(*block, it, load_index->into);
      } break;
      case StmtKind::Special: {
        auto special = Special::Downcast(*it);
        IVLOG(3, "  special: " << special);
        for (const auto& in : special->inputs) {
          tracker.ReadBuffer(it, in, alias_map);
        }
        for (const auto& out : special->outputs) {
          tracker.WriteBuffer(it, out, alias_map);
        }
      } break;
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(*it);
        IVLOG(3, "  intrinsic: " << intrinsic);
        for (const auto& in : intrinsic->inputs) {
          tracker.ReadScalar(*block, in);
        }
        for (const auto& out : intrinsic->outputs) {
          tracker.WriteScalar(*block, it, out);
        }
      } break;
      case StmtKind::Constant: {
        auto constant = Constant::Downcast(*it);
        IVLOG(3, "  constant: " << constant);
        tracker.WriteScalar(*block, it, constant->name);
      } break;
      case StmtKind::Block: {
        auto inner = Block::Downcast(*it);
        IVLOG(3, "  block: " << inner->name);
        AliasMap inner_map(alias_map, inner.get());
        for (const auto& ref : inner->refs) {
          // N.B It doesn't matter whether we process IsReadDir or
          // IsWriteDir first, since a subsequent refinement might access
          // the same underlying physical buffer; we handle these cases in
          // ReadBuffer() and WriteBuffer().
          if (IsReadDir(ref.dir)) {
            tracker.ReadBuffer(it, ref.into(), inner_map);
          }
          if (IsWriteDir(ref.dir)) {
            tracker.WriteBuffer(it, ref.into(), inner_map);
          }
        }
      } break;
    }

    // At this point, dataflow_deps describes the dataflow dependencies of the current Statement.
    // Use it to compute the Statement's transitive dependencies.
    auto& tdeps = transitive_deps[it];
    for (auto dep : tracker.dataflow_deps) {
      const auto& dep_tdeps = transitive_deps.at(dep);
      tdeps.insert(dep_tdeps.begin(), dep_tdeps.end());
    }

    // (dataflow_deps - tdeps) is now the set of actual dependencies for this Statement.
    (*it)->deps.clear();
    std::set_difference(tracker.dataflow_deps.begin(),  //
                        tracker.dataflow_deps.end(),    //
                        tdeps.begin(),                  //
                        tdeps.end(),                    //
                        std::back_inserter((*it)->deps));

    // Add those actual dependencies back into the transitive deps
    // (note that dataflow_deps that are transitively covered by the
    // actual dependencies are already in the set of transitive deps).
    for (auto dep : (*it)->deps) {
      tdeps.insert(dep);
    }

    // Reset dataflow_deps for the next Statement.
    tracker.dataflow_deps.clear();
  }
}

// Recomputes Statement dependencies within all matching Blocks.
void ComputeDepsPass::Apply(stripe::Block* root) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(root, reqs, [](const AliasMap& map, stripe::Block* block) {  //
    ComputeDepsForBlock(block, map);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<ComputeDepsPass, proto::ComputeDepsPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
