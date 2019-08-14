// Copyright 2018, Intel Corporation

#include "tile/codegen/placer.h"

#include <list>
#include <set>
#include <stack>

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/undirected_graph.hpp>

#include "tile/base/shape.h"
#include "tile/codegen/alias.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

// Some terminology: we refer to each Refinement that describes a
// newly-created block of memory (i.e. !IsReadDir(r) &&
// !IsWriteDir(r)) as a _Chunk_.
//
// Each chunk has a _Location_, assigned before placement begins; each
// location is exactly one memory arena.
//
// Each chunk also has an _Offset_ into its location's memory arena,
// assigned by this function -- _Placing_ the chunk.
//
// Placing assigns to each chunk a byte offset within the chunk's
// location, such that if two chunks overlap temporally, they do not
// overlap spatially.
//
// This code implements a simple arena placer.  The algorithm assumes:
//   * The order of statements is fixed
//   * All dependencies are accounted for in the statements.
//
// To place the chunks:
//
// * For each statement, we determine the transitive dependency set of
//   the statement -- the set of statements that must have completed
//   by the time the statement has completed.
//
// * For each chunk, we determine the set of statements that access
//   the chunk, and the intersection of the transitive dependency sets
//   of those statements (i.e. the set of statements known to have
//   completed when the chunk's accessors might run).  We also record
//   the index of the first statement to access the chunk.
//
// * We use these sets to construct a graph for each location, in
//   which the vertices are chunks at that location, and the edges
//   connect chunks that temporally overlap at the same location.
//   (Two chunks do not temporally overlap iff they are in different
//   locations, or if all accessors of the chunk with the lower
//   initial accessor index are in the transitive dependency set of
//   all accessors of the chunk with the higher initial accessor
//   index.)
//
// * For each chunk, we assign a placement, by looking at the
//   placements of existing temporally-overlapping chunks and finding
//   an offset in the location's arena where the chunk can safely
//   coexist with all already-placed chunks.
//
//   Note that optimal arena placement is NP-hard.  So instead of
//   finding an optimal placement, we use a simple approximation: we
//   place the chunks in largest-to-smallest order, using best-fit.
//
//   More concretely: for each chunk (in largest-to-smallest order),
//   we take the set of temporally-overlapping chunks, sort them by
//   existing offset, and then walk the list in order, looking for the
//   smallest free range that's big enough for the chunk.

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

constexpr std::size_t kDefaultAlignment = 4;

struct Chunk;

struct InterferenceVertex {
  Chunk* chunk;
};

typedef boost::adjacency_list<boost::listS, boost::listS, boost::undirectedS, InterferenceVertex> InterferenceGraph;

struct Chunk {
  Chunk(stripe::Refinement* ref_, std::size_t alignment, std::size_t stmt_limit)
      : ref{ref_},
        size{math::Align(stripe::Codec::Resolve(ref_->interior_shape)->byte_size(), alignment)},
        accessors{stmt_limit},
        transitive_accessor_deps{stmt_limit},
        subsequent_accessor_deps{stmt_limit} {}

  stripe::Refinement* ref;
  std::size_t size;
  bool placed = false;
  bool saw_first_accessor = false;
  std::size_t first_accessor_idx = 0;
  boost::dynamic_bitset<> accessors;
  boost::dynamic_bitset<> transitive_accessor_deps;
  boost::dynamic_bitset<> subsequent_accessor_deps;
  InterferenceGraph::vertex_descriptor interference_vertex;
};

struct StmtInfo {
  StmtInfo(std::size_t idx_, std::size_t stmt_limit) : idx{idx_}, transitive_deps{stmt_limit} {}

  std::size_t idx;
  boost::dynamic_bitset<> transitive_deps;
};

class ChunkUseRecorder : public stripe::MutableStmtVisitor {
 public:
  ChunkUseRecorder(const StmtInfo* stmt_info, const AliasMap* alias_map,
                   const std::unordered_map<std::string, Chunk*>* chunks)
      : stmt_info_{stmt_info}, alias_map_{alias_map}, chunks_{chunks} {}

  void Visit(stripe::Load* load) final { RecordUse(load->from); }

  void Visit(stripe::Store* store) final { RecordUse(store->into); }

  void Visit(stripe::LoadIndex* load_index) final {}

  void Visit(stripe::Constant*) final {}  // unused

  void Visit(stripe::Special* special) final {
    for (const std::string& input : special->inputs) {
      RecordUse(input);
    }
    for (const std::string& output : special->outputs) {
      RecordUse(output);
    }
  }

  void Visit(stripe::Intrinsic* intrinsic) final {}  // unused

  void Visit(stripe::Block* block) final {
    block_ = block;

    // Different instances of a block may execute in any order.  We
    // support this by having the block statement reference logically
    // use each refined chunk, forcing those chunks to interfere with
    // each other if they occupy the same arena.
    //
    // TODO: Two chunks could safely alias iff we can guarantee that
    // all reads from a particular offset within the chunk (given all
    // parallel instances of the block executing in any order) happen
    // before all writes to that same offset, which often occurs for
    // elementwise operations.
    for (auto& ref : block->refs) {
      if (IsReadDir(ref.dir) || IsWriteDir(ref.dir)) {
        RecordUse(ref.from);
      }
    }
  }

  stripe::Block* block() const { return block_; }

 private:
  void RecordUse(const std::string& name) {
    const AliasInfo& info = alias_map_->at(name);
    auto it = chunks_->find(info.base_name);
    if (it == chunks_->end()) {
      // This must be a chunk type we're not placing.
      return;
    }
    Chunk* chunk = it->second;
    chunk->accessors.set(stmt_info_->idx);
    if (!chunk->saw_first_accessor) {
      chunk->saw_first_accessor = true;
      chunk->first_accessor_idx = stmt_info_->idx;
      chunk->transitive_accessor_deps = stmt_info_->transitive_deps;
    } else {
      chunk->transitive_accessor_deps &= stmt_info_->transitive_deps;
    }
  }

  const StmtInfo* stmt_info_;
  const AliasMap* alias_map_;
  const std::unordered_map<std::string, Chunk*>* chunks_;
  stripe::Block* block_ = nullptr;
};

std::list<Chunk> BuildChunkList(stripe::Block* outermost_block, const std::set<stripe::Location>& locations,
                                std::size_t alignment, std::size_t stmt_limit, const stripe::Tags& skip_tags) {
  // This function:
  //
  // * Logically numbers each statement within the block
  //   (recursively),
  //
  // * Builds the transitive dependency set for each statement,
  //
  // * Determines the set of statements that access each chunk, and
  //   the intersection of the set of transitive dependencies for each
  //   of those statements (i.e. the set of statements known to have
  //   completed when the chunk's accessors might run).
  struct ToDo {
    stripe::Block* block;
    stripe::StatementIt next;
    AliasMap alias_map;
  };

  std::list<Chunk> result;
  std::unordered_map<std::string, Chunk*> chunks;

  auto add_block_chunks = [&](stripe::Block* block, const AliasMap& alias_map) {
    for (auto& ref : block->refs) {
      if (ref.dir == stripe::RefDir::None && locations.count(ref.location) && !ref.has_any_tags(skip_tags)) {
        auto chunk_it = result.emplace(result.end(), Chunk{&ref.mut(), alignment, stmt_limit});
        chunks[alias_map.at(ref.into()).base_name] = &*chunk_it;
      }
    }
  };

  AliasMap root_map;

  std::stack<ToDo> todo;
  todo.emplace(ToDo{outermost_block, outermost_block->stmts.begin(), AliasMap{root_map, outermost_block}});
  add_block_chunks(outermost_block, todo.top().alias_map);

  std::unordered_map<stripe::Statement*, StmtInfo> stmt_infos;

  std::size_t idx_limit = 0;

  while (todo.size()) {
    stripe::Block* block = todo.top().block;
    stripe::StatementIt& it = todo.top().next;
    AliasMap* alias_map = &todo.top().alias_map;

    for (;;) {
      if (it == block->stmts.end()) {
        todo.pop();
        break;
      }
      auto sit_placed = stmt_infos.emplace(it->get(), StmtInfo{idx_limit++, stmt_limit});
      StmtInfo& info = sit_placed.first->second;
      for (stripe::StatementIt& dep : (*it)->deps) {
        const auto& dep_info = stmt_infos.at(dep->get());
        info.transitive_deps.set(dep_info.idx);
        info.transitive_deps |= dep_info.transitive_deps;
      }
      ChunkUseRecorder recorder{&info, alias_map, &chunks};
      (*it)->Accept(&recorder);
      ++it;
      if (recorder.block()) {
        stripe::Block* sub_block = recorder.block();
        todo.emplace(ToDo{sub_block, sub_block->stmts.begin(), AliasMap{*alias_map, sub_block}});
        add_block_chunks(sub_block, todo.top().alias_map);
        break;
      }
    }
  }

  return result;
}

std::size_t CountStatements(stripe::Block* outermost_block) {
  std::size_t result = 0;

  std::queue<stripe::Block*> pending;
  pending.push(outermost_block);

  while (pending.size()) {
    stripe::Block* block = pending.front();
    pending.pop();
    result += block->stmts.size();
    for (const auto& stmt : block->stmts) {
      stripe::Block* child_block = dynamic_cast<stripe::Block*>(stmt.get());
      if (child_block) {
        pending.push(child_block);
      }
    }
  }

  return result;
}

}  // namespace

void PlaceRefinements(stripe::Block* outermost_block, const proto::MemoryPlacementPass& options) {
  std::set<stripe::Location> locations;
  for (const auto& loc : options.locs()) {
    locations.emplace(stripe::FromProto(loc));
  }

  std::size_t stmt_limit = CountStatements(outermost_block);
  auto alignment = options.alignment() ? options.alignment() : kDefaultAlignment;
  auto skip_tags = stripe::FromProto(options.skip_tags());

  std::list<Chunk> chunks = BuildChunkList(outermost_block, locations, alignment, stmt_limit, skip_tags);

  // Edge case: no chunks means nothing to do.  And then after this,
  // we can assume there's at least one chunk.
  if (!chunks.size()) {
    return;
  }

  // Ensure chunks are sorted by earliest accessor.
  chunks.sort([](const Chunk& lhs, const Chunk& rhs) { return lhs.first_accessor_idx < rhs.first_accessor_idx; });

  // Initialize subsequent accessor dependencies.
  {
    boost::dynamic_bitset<> deps = chunks.back().transitive_accessor_deps;
    for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
      deps &= it->transitive_accessor_deps;
      it->subsequent_accessor_deps = deps;
    }
  }

  // Initialize the graph.
  InterferenceGraph interference_graph;
  for (auto& chunk : chunks) {
    chunk.interference_vertex = boost::add_vertex(InterferenceVertex{&chunk}, interference_graph);
  }

  // Compute the interference edges.  Note that this is O(N^2) in the
  // worst case.  :-/
  for (auto earlier = chunks.begin(); earlier != chunks.end(); ++earlier) {
    auto later = earlier;
    ++later;
    for (; later != chunks.end(); ++later) {
      if (earlier->ref->location != later->ref->location) {
        // These cannot spatially overlap; they never interfere with
        // each other.
        continue;
      }
      if (earlier->accessors.is_subset_of(later->subsequent_accessor_deps)) {
        // All accessors of the earlier chunk are transitive
        // dependencies of this and every subsequent chunk; there can
        // be no interference, so we are done checking chunks later
        // than the current earliest.
        break;
      }
      if (earlier->accessors.is_subset_of(later->transitive_accessor_deps)) {
        // All accessors of the earlier chunk are transitive
        // dependencies of all of the accessors of the later chunk --
        // these chunks do not temporally overlap, and cannot
        // interfere with each other.
        continue;
      }
      // Otherwise, these chunks may be alive at the same time in the
      // same location; they may interfere with each other.
      add_edge(earlier->interference_vertex, later->interference_vertex, interference_graph);
    }
  }

  // Re-sort chunks by size, since we want to place them in
  // largest-to-smallest order.
  chunks.sort([](const Chunk& lhs, const Chunk& rhs) { return lhs.size > rhs.size; });

  // Place the chunks.
  std::vector<Chunk*> already_placed;
  for (auto& chunk : chunks) {
    // Build a vector of already-placed chunks that we need to
    // consider when placing this chunk.
    already_placed.clear();
    auto edges = out_edges(chunk.interference_vertex, interference_graph);
    for (auto it = edges.first; it != edges.second; ++it) {
      InterferenceGraph::vertex_descriptor tgt = target(*it, interference_graph);
      Chunk* candidate = interference_graph[tgt].chunk;
      if (candidate->placed) {
        already_placed.push_back(candidate);
      }
    }

    // Sort the vector by placement offset, lowest-offset first.
    std::sort(already_placed.begin(), already_placed.end(),
              [](const Chunk* lhs, const Chunk* rhs) { return lhs->ref->offset < rhs->ref->offset; });

    // Scan for usable gaps in the already_placed vector.
    // N.B. The already_placed vector may contain overlapping chunks.
    //
    // We keep track of the overall limit offset we've seen so far, as
    // well as whether we've seen a usable gap at all, the location of
    // that gap, and the known size of that gap.
    //
    // Note that the gap we're considering is always
    // [offset_limit...current->ref->offset).  This is because either:
    //
    // * We don't have a gap yet; some earlier-processed
    //   already-placed chunk must have started at an offset that's
    //   too low for the current chunk we're placing to fit before it,
    //   and subsequent chunks have either been overlapping or had too
    //   small a gap for us to consider.  offset_limit is the first
    //   offset that's known to be past all these earlier-processed
    //   chunks.
    //
    // * We do have a candidate gap.  It's the best gap we know about
    //   given all the chunks we've seen so far, which cover the
    //   entire space up to offset_limit.  So any better gap has to
    //   start at offset_limit.
    std::size_t offset_limit = 0;
    bool have_gap = false;
    std::size_t gap_offset = 0;
    std::size_t gap_size = 0;

    for (Chunk* placed_chunk : already_placed) {
      // See whether we have a usable gap here.
      if (offset_limit < placed_chunk->ref->offset) {
        std::size_t candidate_size = placed_chunk->ref->offset - offset_limit;
        if (chunk.size <= candidate_size && (!have_gap || (candidate_size < gap_size))) {
          // This is a usable gap, and either we don't have a gap or
          // the candidate gap is smaller than the best gap we've
          // found so far.
          have_gap = true;
          gap_offset = offset_limit;
          gap_size = candidate_size;
        }
      }

      // Update offset_limit.
      std::size_t placed_chunk_limit = placed_chunk->ref->offset + placed_chunk->size;
      if (offset_limit < placed_chunk_limit) {
        offset_limit = placed_chunk_limit;
      }
    }

    // If we don't have a gap yet, offset_limit is where we have to put the current chunk.
    if (!have_gap) {
      gap_offset = offset_limit;
    }

    // We have an offset for this chunk.
    chunk.ref->offset = gap_offset;
    chunk.ref->set_tag("placed");
    chunk.placed = true;
  }
}

void MemoryPlacementPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    PlaceRefinements(block, options_);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<MemoryPlacementPass, proto::MemoryPlacementPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
