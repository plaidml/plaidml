// Copyright 2018, Intel Corporation

#include "tile/codegen/schedule/deps.h"

#include <set>
#include <stdexcept>
#include <strstream>
#include <unordered_map>
#include <unordered_set>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace schedule {
namespace {

class BlockDepComputer : private stripe::MutableStmtVisitor {
 public:
  static void Run(stripe::Block* block, const AliasMap& alias_map);

 private:
  BlockDepComputer(stripe::Block* block, const AliasMap& alias_map);

  void WriteScalar(const std::string& name, stripe::Statement* stmt);
  void ReadScalar(const std::string& name, stripe::Statement* stmt);

  void WriteBuffer(const std::string& name, stripe::Statement* stmt);
  void ReadBuffer(const std::string& name, stripe::Statement* stmt);

  // Tracks the state of a buffer as it's operated on by the Block's Statements.
  struct BufferInfo {
    // TODO: Instead of tracking the latest writer, keep track of a
    // list of all currently active writers, removing writers from the
    // list when all written memory has been overwritten by subsequent
    // writers.
    stripe::Statement* latest_writer = nullptr;
    std::unordered_set<stripe::Statement*> current_readers;
  };

  void Visit(stripe::Load* load) final;
  void Visit(stripe::Store* store) final;
  void Visit(stripe::Constant* constant) final;
  void Visit(stripe::Special* special) final;
  void Visit(stripe::Intrinsic* intrinsic) final;
  void Visit(stripe::Block* block) final;

  // The block being processed.
  stripe::Block* block_;

  // The buffer's alias map.
  const AliasMap& alias_map_;

  // The dataflow dependencies of the current Statement being processed.
  std::set<stripe::Statement*> dataflow_deps_;

  // For each scalar: the Statement that wrote the scalar.
  std::unordered_map<std::string, stripe::Statement*> scalars_;

  // For each buffer (keyed by the buffer base name from the alias
  // map): information about the buffer's accessors (as of the current
  // Statement being processed).
  std::unordered_map<std::string, BufferInfo> buffers_;

  // The transitive dependencies of each Statement within the current
  // Block.
  std::unordered_map<stripe::Statement*, std::set<stripe::Statement*>> transitive_deps_;
};

void BlockDepComputer::Run(stripe::Block* block, const AliasMap& alias_map) {
  BlockDepComputer c{block, alias_map};

  for (const auto& stmt : block->stmts) {
    // Visit the statement, adjusting the current scalar and buffer
    // tracking structures and updating c.dataflow_deps_.
    stmt->Accept(&c);

    // At this point, dataflow_deps_ describes the dataflow
    // dependencies of the current Statement.  Use it to compute the
    // Statement's transitive dependencies.
    std::set<stripe::Statement*>* tdeps = &c.transitive_deps_[stmt.get()];
    for (stripe::Statement* dep : c.dataflow_deps_) {
      const std::set<stripe::Statement*>& dep_tdeps = c.transitive_deps_.at(dep);
      tdeps->insert(dep_tdeps.begin(), dep_tdeps.end());
    }

    // (dataflow_deps_ - tdeps) is now the set of actual dependencies
    // for this Statement.
    stmt->deps.clear();
    std::set_difference(c.dataflow_deps_.begin(), c.dataflow_deps_.end(), tdeps->begin(), tdeps->end(),
                        std::back_inserter(stmt->deps));

    // Add those actual dependencies back into the transitive deps
    // (note that dataflow_deps_ that are transitively covered by the
    // actual depeendencies are already in the set of transitive
    // deps).
    for (stripe::Statement* dep : stmt->deps) {
      tdeps->emplace(dep);
    }

    // Reset dataflow_deps_ for the next Statement.
    c.dataflow_deps_.clear();
  }
}

BlockDepComputer::BlockDepComputer(stripe::Block* block, const AliasMap& alias_map)
    : block_{block}, alias_map_{alias_map} {}

void BlockDepComputer::WriteScalar(const std::string& name, stripe::Statement* stmt) {
  // Note that scalars are SSA, so there will only ever be one writer per block.
  auto it_inserted = scalars_.emplace(name, stmt);
  if (!it_inserted.second) {
    std::stringstream ss;
    ss << "Scalar " << name << " written multiple times in " << *block_;
    throw std::logic_error{ss.str()};
  }
}

void BlockDepComputer::ReadScalar(const std::string& name, stripe::Statement* /* stmt */) {
  auto it = scalars_.find(name);
  if (it == scalars_.end()) {
    std::stringstream ss;
    ss << "Read scalar " << name << " before it was written in " << *block_;
    throw std::logic_error{ss.str()};
  }
  dataflow_deps_.emplace(it->second);
}

void BlockDepComputer::WriteBuffer(const std::string& name, stripe::Statement* stmt) {
  const AliasInfo& alias_info = alias_map_.at(name);
  BufferInfo& buffer_info = buffers_[alias_info.base_name];

  // For now, we assume that each writing statement writes the entire
  // buffer.  TODO: Track child blocks as sub-buffer writes.
  if (buffer_info.latest_writer && buffer_info.latest_writer != stmt) {
    dataflow_deps_.emplace(buffer_info.latest_writer);
  }

  for (stripe::Statement* reader : buffer_info.current_readers) {
    // Writing a buffer we're also reading blocks on all readers that
    // aren't us, but otherwise looks like a normal write of a buffer
    // that something else is reading.
    if (reader != stmt) {
      dataflow_deps_.emplace(reader);
    }
  }

  buffer_info.latest_writer = stmt;
  buffer_info.current_readers.clear();
}

void BlockDepComputer::ReadBuffer(const std::string& name, stripe::Statement* stmt) {
  const AliasInfo& alias_info = alias_map_.at(name);
  BufferInfo& buffer_info = buffers_[alias_info.base_name];

  if (buffer_info.latest_writer == stmt) {
    // Reading a buffer we're also writing doesn't do anything; we just track the write.
    return;
  }

  if (buffer_info.latest_writer) {
    dataflow_deps_.emplace(buffer_info.latest_writer);
  }

  buffer_info.current_readers.emplace(stmt);
}

void BlockDepComputer::Visit(stripe::Load* load) {
  ReadBuffer(load->from, load);
  WriteScalar(load->into, load);
}

void BlockDepComputer::Visit(stripe::Store* store) {
  ReadScalar(store->from, store);
  WriteBuffer(store->into, store);
}

void BlockDepComputer::Visit(stripe::Constant* constant) { WriteScalar(constant->name, constant); }

void BlockDepComputer::Visit(stripe::Special* special) {
  for (const auto& in : special->inputs) {
    ReadBuffer(in, special);
  }
  for (const auto& out : special->outputs) {
    WriteBuffer(out, special);
  }
}

void BlockDepComputer::Visit(stripe::Intrinsic* intrinsic) {
  for (const auto& in : intrinsic->inputs) {
    ReadScalar(in, intrinsic);
  }
  for (const auto& out : intrinsic->outputs) {
    WriteScalar(out, intrinsic);
  }
}

void BlockDepComputer::Visit(stripe::Block* block) {
  for (const auto& ref : block->refs) {
    // N.B It doesn't matter whether we process IsReadDir or
    // IsWriteDir first, since a subsequent refinement might access
    // the same underlying physical buffer; we handle these cases in
    // ReadBuffer() and WriteBuffer().
    if (IsReadDir(ref.dir)) {
      ReadBuffer(ref.from, block);
    }
    if (IsWriteDir(ref.dir)) {
      WriteBuffer(ref.from, block);
    }
  }
}

}  // namespace

void ComputeDepsForBlock(stripe::Block* block, const AliasMap& alias_map) { BlockDepComputer::Run(block, alias_map); }

void ComputeDepsForTree(stripe::Block* outermost_block) {
  // Tracks a block to be processed.
  struct PendingBlock {
    stripe::Block* block;
    AliasMap alias_map;
  };

  // The queue of blocks whose dependencies are to be computed.
  std::queue<PendingBlock> pending;
  AliasMap root;

  pending.push(PendingBlock{outermost_block, AliasMap{root, *outermost_block}});

  while (pending.size()) {
    PendingBlock& todo = pending.front();

    // Schedule dependencies within this block.
    ComputeDepsForBlock(todo.block, todo.alias_map);

    // Add child blocks to the queue.
    for (const auto& stmt : todo.block->stmts) {
      stripe::Block* child_block = dynamic_cast<stripe::Block*>(stmt.get());
      if (child_block) {
        pending.push(PendingBlock{child_block, AliasMap{todo.alias_map, *child_block}});
      }
    }

    pending.pop();
  }
}

}  // namespace schedule
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
