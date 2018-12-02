// Copyright 2018, Intel Corporation

#include "tile/codegen/deps.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
    // TODO: Instead of tracking the latest writer, keep track of a
    // list of all currently active writers, removing writers from the
    // list when all written memory has been overwritten by subsequent writers.
    bool has_latest_writer = false;
    StatementIt latest_writer;
    std::unordered_set<StatementIt> current_readers;
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
          std::logic_error(printstring("Scalar %s written multiple times in %s", name.c_str(), block.name.c_str())));
    }
  }

  void ReadScalar(const Block& block, const std::string& name) {
    auto it = scalars.find(name);
    if (it == scalars.end()) {
      throw_with_trace(std::logic_error(
          printstring("Scalar %s read before it was written in %s", name.c_str(), block.name.c_str())));
    }
    dataflow_deps.insert(it->second);
  }

  void WriteBuffer(StatementIt it, const std::string& name, const AliasMap& alias_map) {
    const AliasInfo& alias_info = alias_map.at(name);
    BufferInfo& buffer_info = buffers[alias_info.base_name];

    // For now, we assume that each writing statement writes the entire
    // buffer.  TODO: Track child blocks as sub-buffer writes.
    if (buffer_info.has_latest_writer && buffer_info.latest_writer != it) {
      dataflow_deps.insert(buffer_info.latest_writer);
    }

    for (StatementIt reader : buffer_info.current_readers) {
      // Writing a buffer we're also reading blocks on all readers that
      // aren't us, but otherwise looks like a normal write of a buffer
      // that something else is reading.
      if (reader != it) {
        dataflow_deps.insert(reader);
      }
    }

    buffer_info.has_latest_writer = true;
    buffer_info.latest_writer = it;
    buffer_info.current_readers.clear();
  }

  void ReadBuffer(StatementIt it, const std::string& name, const AliasMap& alias_map) {
    const AliasInfo& alias_info = alias_map.at(name);
    BufferInfo& buffer_info = buffers[alias_info.base_name];

    if (buffer_info.has_latest_writer && buffer_info.latest_writer == it) {
      // Reading a buffer we're also writing doesn't do anything; we just track the write.
      return;
    }

    if (buffer_info.has_latest_writer) {
      dataflow_deps.insert(buffer_info.latest_writer);
    }

    buffer_info.current_readers.insert(it);
  }
};

}  // namespace

void ComputeDepsForBlock(Block* block, const AliasMap& alias_map) {  //
  Tracker tracker;
  std::unordered_map<StatementIt, std::set<StatementIt>> transitive_deps;
  for (auto it = block->stmts.begin(); it != block->stmts.end(); it++) {
    // Adjust the current scalar and buffer tracking structures and update dataflow_deps.
    switch ((*it)->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(*it);
        tracker.ReadBuffer(it, load->from, alias_map);
        tracker.WriteScalar(*block, it, load->into);
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(*it);
        tracker.ReadScalar(*block, store->from);
        tracker.WriteBuffer(it, store->into, alias_map);
      } break;
      case StmtKind::Special: {
        auto special = Special::Downcast(*it);
        for (const auto& in : special->inputs) {
          tracker.ReadBuffer(it, in, alias_map);
        }
        for (const auto& out : special->outputs) {
          tracker.WriteBuffer(it, out, alias_map);
        }
      } break;
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(*it);
        for (const auto& in : intrinsic->inputs) {
          tracker.ReadScalar(*block, in);
        }
        for (const auto& out : intrinsic->outputs) {
          tracker.WriteScalar(*block, it, out);
        }
      } break;
      case StmtKind::Constant: {
        auto constant = Constant::Downcast(*it);
        tracker.WriteScalar(*block, it, constant->name);
      } break;
      case StmtKind::Block: {
        auto inner = Block::Downcast(*it);
        for (const auto& ref : inner->refs) {
          // N.B It doesn't matter whether we process IsReadDir or
          // IsWriteDir first, since a subsequent refinement might access
          // the same underlying physical buffer; we handle these cases in
          // ReadBuffer() and WriteBuffer().
          if (IsReadDir(ref.dir)) {
            tracker.ReadBuffer(it, ref.from, alias_map);
          }
          if (IsWriteDir(ref.dir)) {
            tracker.WriteBuffer(it, ref.from, alias_map);
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

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
