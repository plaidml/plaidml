// Copyright 2018, Intel Corp.

#include "tile/codegen/schedule.h"

#include <forward_list>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/optional.hpp>

#include "base/util/error.h"
#include "base/util/lookup.h"
#include "base/util/throw.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"

// This code implements a simple single-linear-pass caching memory
// scheduler for a single Stripe Block.  It builds up information
// about the cache state on the fly as it performs a scan through the
// Statements, and uses some simple/straightforward heuristics to
// guide its decisions.
//
// N.B. We perform the linear scan of the Statements in *reverse*
// order -- Statements in the future of the scheduling pass are in the
// past of the runtime execution.
//
// At the top of our scheduling loop, the current state is "What the
// future would like us to arrange for it."  That is, the
// runtime-future is going to assume that there are various values at
// particular offsets in the local cache, and it's the scheduler's job
// to extend the current state to incorporate the Statement its
// considering, inserting swapping blocks as needed, such that the
// runtime future's invariants (i.e. Statements that have previously
// been scheduled) hold.
//
// (The reason we do the scheduling in reverse is that we want to
// initiate data movement as early (in runtime terms) as possible, and
// the code seems to wind up being simpler if its internal
// state-of-the-system datastructures are tracking the desired
// runtime-future state of the system than those datastructures are
// tracking the runtime-past of the system and then fixing up that
// past.)

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

constexpr std::size_t kDefaultAlignment = 4;

using ColorGraph = boost::undirected_graph<>;
using ColorVertex = boost::graph_traits<ColorGraph>::vertex_descriptor;

struct CacheEntry;

// RefInfo contains information around the usage of one particular
// backing ref during the scan.
struct RefInfo {
  RefInfo(stripe::Refinement* ref_, AliasInfo alias_info_)
      : ref(*ref_),  //
        alias_info{std::move(alias_info_)},
        exterior_cache_shape{ref.interior_shape},
        name{ref.into()} {
    IVLOG(3, "Creating RefInfo " << ref << " extents=" << alias_info.extents);

    // Convert the cached shape to use natural striding.
    std::uint64_t stride = 1;
    for (std::size_t idx = 0; idx < exterior_cache_shape.dims.size(); ++idx) {
      auto& dim = exterior_cache_shape.dims.at(exterior_cache_shape.dims.size() - idx - 1);
      dim.stride = stride;
      stride *= dim.size;
    }

    auto sizes = exterior_cache_shape.sizes();
    size = stripe::Codec::Resolve(exterior_cache_shape)->byte_size();

    for (size_t i = 0; i < sizes.size(); i++) {
      std::string iname = "i" + std::to_string(i);
      swap_idxs.emplace_back(stripe::Index{iname, sizes[i]});
      ref_swap_access.emplace_back(stripe::Affine(iname));
      cache_swap_access.emplace_back(stripe::Affine(iname));
    }

    ref_swap_shape = ref.interior_shape;
    cache_swap_shape = exterior_cache_shape;
    for (size_t i = 0; i < sizes.size(); i++) {
      ref_swap_shape.dims[i].size = 1;
      cache_swap_shape.dims[i].size = 1;
    }
  }

  // The actual backing ref -- e.g. DRAM.  We keep a copy because when
  // we add a refinement to our block's refinement vector, we
  // invalidate all iterators / pointers.
  stripe::Refinement ref;

  // The alias info for this ref.  N.B. This may be either an
  // AliasInfo for the block being scheduled, or an alias info for a
  // subblock, depending on the statement being scheduled that
  // resulted in this alias.
  AliasInfo alias_info;

  // The shape of the ref's data when it's in the local cache, when
  // the data is exterior to the sub-statements (i.e. eligible to be
  // reused across sub-blocks).  Note that this may differ from the
  // ref's shape.
  TensorShape exterior_cache_shape;

  // The shapes to use for swap block refinements.
  TensorShape ref_swap_shape;
  TensorShape cache_swap_shape;

  // The affines to use for swapping.
  std::vector<stripe::Affine> ref_swap_access;
  std::vector<stripe::Affine> cache_swap_access;

  // The indices to use for swapping.
  std::vector<stripe::Index> swap_idxs;

  // The size of the ref (when cached).
  std::size_t size;

  // True iff this refinement's been used by the schedule.
  // Unused refinements are pruned.
  bool used = false;

  // True iff the final write for this RefInfo has been seen
  // (i.e. false initially, and set to true by the first swap-out in
  // scheduling order to write to this ref).  This is used to cover
  // the case where multiple writers are updating an out-ref: we must
  // swap-out the final write (in runtime order), but should elide
  // other swap-outs if possible.
  bool saw_final_write = false;

  // The current CacheEntry to use to access a local instantiation of
  // the backing ref -- i.e. the CacheEntry where some
  // previously-scheduled reader is expecting the value it needs.
  //
  // Note that there will only be one at scheduling time, even though
  // at runtime, there might be multiple copies of the CacheEntry in
  // memory at once.
  CacheEntry* cache_entry = nullptr;

  // The Statements that're going to be (runtime-future) swapping in
  // the contents of the backing memory -- i.e. the Statements that
  // will need to pick up a dependency on the swap-out statement that
  // writes the backing memory.
  std::unordered_set<stripe::Statement*> swap_in_readers;

  // The index to use for the next CacheEntry for this refinement.
  std::size_t next_cache_entry = 0;

  // The vector of RefInfos that refine the same base refinement.
  std::vector<RefInfo*>* aliases = nullptr;

  // The earliest (runtime-past) sub-statement of the main block that
  // writes to this refinement.
  stripe::Statement* earliest_writer = nullptr;

  // The local name of this ref.
  std::string name;

  // The color interference vertex of this ref.
  ColorVertex color_vertex;
};

using RefInfoKey = std::string;

// A range of memory.
struct MemRange {
  MemRange() {}
  MemRange(std::size_t begin_, std::size_t end_) : begin{begin_}, end{end_} {}

  std::size_t begin = 0;
  std::size_t end = 0;

  std::size_t size() const { return end - begin; }
};

std::ostream& operator<<(std::ostream& o, MemRange mr) { return o << "[" << mr.begin << " - " << mr.end << ")"; }

// Returns true iff the supplied ranges overlap.
bool RangesOverlap(MemRange a, MemRange b) { return (a.begin < b.end) && (b.begin < a.end); }

// Returns true iff the supplied range overlaps any of the ranges in the range list.
bool RangesOverlap(MemRange range, const std::list<MemRange>& range_list) {
  for (const auto& check_range : range_list) {
    if (RangesOverlap(range, check_range)) {
      return true;
    }
  }
  return false;
}

// Subtracts a range from a particular range (identified by 'it')
// within a list of ranges.  'it' must be a dereferencable iterator.
void SubtractRange(MemRange sub, std::list<MemRange>* range_list, std::list<MemRange>::iterator it) {
  auto& range = *it;

  // So there are four cases here:
  if (sub.begin <= range.begin) {
    // The range we're subtracting begins at or before the begin of the current range.
    if (sub.end < range.end) {
      // The range we're subtracting is taking a chunk off the low side of the current range.
      range.begin = sub.end;
    } else {
      // The range we're subtracting completely covers the current range.
      range_list->erase(it);
    }
  } else if (range.end < sub.end) {
    // The range we're subtracting ends after the end of the current range.
    // Since the range we're subtracting begins after the begin of the current range,
    // we're subtracting a chunk off the high side of the current range.
    range.end = sub.begin;
  } else {
    // The range we're subtracting splits the current range.
    // We emplace a new entry for the low part of the current range, and adjust
    // the current range to be the high part of the current range.
    range_list->emplace_front(MemRange{range.begin, sub.begin});
    range.begin = sub.end;
  }
}

// Subtracts a range from a list of ranges.
void SubtractRange(MemRange sub, std::list<MemRange>* range_list) {
  IVLOG(3, "        Subtracting range " << sub << " from: " << *range_list);
  for (auto it = range_list->begin(); it != range_list->end();) {
    auto cit = it;
    ++it;
    if (!RangesOverlap(sub, *cit)) {
      continue;
    }

    SubtractRange(sub, range_list, cit);
  }
  IVLOG(3, "        Ranges are now " << *range_list);
}

// Represents a single proposed placement of a statement input or output.
struct Placement {
  Placement() {}
  Placement(stripe::RefDir dir_, std::size_t size_, bool is_internal_, const std::string& interior_name_)
      : dir{dir_}, size{size_}, is_internal{is_internal_}, interior_name{interior_name_} {}
  Placement(stripe::RefDir dir_, MemRange range_, CacheEntry* entry_)
      : dir{dir_}, size{range_.end - range_.begin}, range{range_}, entry{entry_} {}

  // What the Statement is doing with this placement.
  stripe::RefDir dir = stripe::RefDir::None;

  // The size of the placement (equal to range.end-range.begin, once
  // range has been established).
  std::size_t size;

  // Where the entry should go.
  stripe::Affine unit;
  MemRange range;

  // The entry for this Placement.  N.B. This may be nullptr, in which
  // case it will be filled in when the plan is accepted.
  CacheEntry* entry = nullptr;

  // Indicates whether this is an internal placement (caching a
  // partial refinement swapped within the sub-statement block being
  // scheduled) or an external placement (which can be reused between
  // sub-statements).
  bool is_internal = false;

  // For internal placements, the interior name used to refer to
  // the entry within the block.
  std::string interior_name;

  // The internal access affines for this placement (used for adding
  // subblock-local swaps; only used if is_internal==true).
  std::vector<stripe::Affine> access;
};

std::ostream& operator<<(std::ostream& o, const Placement& p) { return o << p.range; }

struct PlacementKey {
  RefInfo* ri;
  TensorShape cache_shape;
  std::vector<stripe::Affine> access;
};

bool operator<(const PlacementKey& lhs, const PlacementKey& rhs) {
  return std::tie(lhs.ri->ref.into(), lhs.cache_shape, lhs.access) <
         std::tie(rhs.ri->ref.into(), rhs.cache_shape, rhs.access);
}

// Represents a placement plan for a particular Statement.
using PlacementPlan = std::map<PlacementKey, Placement>;

// Translates a location within a sub-block from
// parent-block-indices to sub-block-indices (connecting indices to
// the parent indices as needed).
void TranslateLocation(stripe::Block* child_block, stripe::Location* loc) {
  // Map of affine of parent idxs -> child idx.
  std::map<stripe::Affine, std::string> existing_idxs;
  for (const auto& idx : child_block->idxs) {
    existing_idxs[idx.affine] = idx.name;
  }

  // Translate each unit of the location.
  for (auto& dev : loc->devs) {
    for (auto& unit : dev.units) {
      if (unit.isConstant()) {
        continue;
      }
      auto it_inserted = existing_idxs.emplace(unit, std::string{});
      if (it_inserted.second) {
        std::string name = dev.name;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        it_inserted.first->second = child_block->unique_idx_name(name);
        child_block->idxs.push_back(stripe::Index{it_inserted.first->second, 1, unit});
      }
      unit = stripe::Affine{it_inserted.first->second};
    }
  }
}

// CacheEntry represents one particular local instantiation of a
// value.  (i.e. swapping out a value and swapping it back in results
// in a new CacheEntry).
struct CacheEntry {
  explicit CacheEntry(std::pair<PlacementKey, Placement> pkey_placement)
      : source{pkey_placement.first.ri},
        name{source->name + "^" + std::to_string(source->next_cache_entry++)},
        unit{pkey_placement.second.unit},
        range{pkey_placement.second.range},
        shape{pkey_placement.first.cache_shape},
        is_internal{pkey_placement.second.is_internal},
        interior_name{pkey_placement.second.interior_name} {
    uncovered_ranges.push_back(range);
  }

  // The CacheEntry's backing refinement.
  RefInfo* source;

  // The CacheEntry's refinement's name (its "into" when it becomes a
  // Refinement).
  std::string name;

  // The CacheEntry's memory range when it's being used.
  stripe::Affine unit;
  MemRange range;

  // The data shape for this CacheEntry.  For internal cache entries,
  // this is the shape of the entry interior to the sub-statement
  // that's accessing it; otherwise, it is the shape of the cache
  // entry exterior to any particular sub-statement.
  TensorShape shape;

  // Whether this CacheEntry is internal-only.
  bool is_internal;

  // When this CacheEntry is internal to a sub-statement (a Block),
  // the interior name used for it within the block.
  std::string interior_name;

  // CacheEntry usage tracking.  These track the runtime-future use of
  // the CacheEntry's memory range.
  //
  // At each point in scheduling where the CacheEntry's backing memory
  // is read, the reader is added to the readers set of the CacheEntry
  // it's actually reading, and all existing writers of any CacheEntry
  // covering the memory (which are in the runtime-future of the
  // reader) pick up a dependency on the reader (since those writers
  // can't reuse the memory until all readers of that memory have
  // completed).
  //
  // At each point in scheduling where the CacheEntry's backing memory
  // is written:
  //
  //   * Existing overlapping readers pick up a dependency on the
  //     current statement,
  //
  //   * The writer is added to the writers map,
  //
  //   * If CacheEntry is also an input to the writing Statement
  //     (i.e. the writer is also a reader), the writing Statement is
  //     also added to readers.
  //
  // N.B. At write-time, the CacheEntry will always already exist
  // (edge case: it might not, if what we're writing is a program
  // output, but in that case, we schedule a swap out to main memory,
  // and that swap-out becomes the reader that causes the CacheEntry
  // to exist).
  stripe::StatementIt first_accessor;  // In runtime order
  std::unordered_map<stripe::Statement*, AliasInfo> writers;
  std::unordered_map<stripe::Statement*, AliasInfo> readers;

  // True iff we've seen this entry written by the first statement (in
  // runtime order) that writes to it.  This is used to determine
  // swap-in necessity: if we've seen that first statement write to
  // this CacheEntry, then there's no need for a swap-in; otherwise,
  // the first statement to write to this CacheEntry is in the
  // runtime-past, and so if we're covering it up with another entry,
  // we need to schedule a swap-in.
  bool saw_earliest_writer = false;

  // The CacheEntry's position in its active cache entry list.
  std::list<CacheEntry*>::iterator active_iterator;

  // The CacheEntry's uncovered ranges.  When this list is empty, the
  // CacheEntry is removed from the active cache entry list.
  std::list<MemRange> uncovered_ranges;
};

// Represents a unit of IO performed by a sub-statement.
struct IO {
  IO(RefInfo* ri_, stripe::RefDir dir_) : ri{ri_}, dir{dir_}, interior_shape{ri_->exterior_cache_shape} {}
  explicit IO(std::pair<RefInfo*, stripe::RefDir> p) : IO{p.first, p.second} {}

  IO(RefInfo* ri_, const stripe::Refinement& interior_ref)
      : ri{ri_},
        dir{interior_ref.dir},
        interior_shape{interior_ref.interior_shape},
        interior_name{interior_ref.into()},
        access{interior_ref.access} {
    // Restride the interior shape - if it's used, it needs to be in
    // compact form.
    std::size_t stride = 1;
    for (std::size_t i = interior_shape.dims.size(); i; --i) {
      interior_shape.dims[i - 1].stride = stride;
      stride *= interior_shape.dims[i - 1].size;
    }
  }

  RefInfo* ri;
  stripe::RefDir dir;
  TensorShape interior_shape;
  std::string interior_name;
  std::vector<stripe::Affine> access;  // N.B. Only valid for block statements
};

// Encapsulates the notion of a post-scheduling update to a Statement,
// rewriting its refinement references (recursively, in the case of
// Blocks) to point to the cache entries determined by memory
// placement.
//
// The update uses the current contents of the evolving RefInfo
// datastructure; it must be used while the contents of the RefInfo
// structure are appropriate for the current Statement -- i.e. before
// scheduling the next Statement.
class StatementBinder final {
 public:
  // Construct an uninitialized StatementBinder.
  StatementBinder() {}

  // Construct a StatementBinder for a non-Block.
  explicit StatementBinder(std::vector<std::pair<std::string*, RefInfo*>> updates)
      : non_block_updates_{std::move(updates)} {}

  // Construct a StatementBinder for a Block.
  StatementBinder(std::vector<std::pair<stripe::Refinement*, RefInfo*>> updates, stripe::Block* block,
                  const stripe::Location* mem_loc)
      : block_updates_{std::move(updates)}, block_{block}, mem_loc_{mem_loc} {}

  // Apply the StatementBinder to the Statement from which it was generated.
  void ApplyBindings() {
    if (!block_) {
      // Non-Blocks are easy.
      for (auto& update : non_block_updates_) {
        *update.first = update.second->cache_entry->name;
      }
      return;
    }

    // For Blocks, we need to do a little more work to recursively update the refinements.
    for (auto& update : block_updates_) {
      stripe::Refinement* ref = update.first;
      RefInfo* ri = update.second;
      ref->from = ri->cache_entry->name;
      ref->location = PartialEval(*mem_loc_, {{"unit", ri->cache_entry->unit.constant()}});
      TranslateLocation(block_, &ref->location);
      if (ri->cache_entry->is_internal) {
        ref->interior_shape = ri->cache_entry->shape;
        for (auto& access : ref->access) {
          access = 0;
        }
      } else {
        for (size_t i = 0; i < ref->interior_shape.dims.size(); i++) {
          ref->interior_shape.dims[i].stride = ri->exterior_cache_shape.dims[i].stride;
        }
      }
      FixupRefs(block_, ref->into());
    }
  }

 private:
  std::vector<std::pair<std::string*, RefInfo*>> non_block_updates_;
  std::vector<std::pair<stripe::Refinement*, RefInfo*>> block_updates_;
  stripe::Block* block_ = nullptr;
  const stripe::Location* mem_loc_ = nullptr;
};

// Gathers a Statement's IO information.
class IOGatherer final : private stripe::MutableStmtVisitor {
 public:
  static std::pair<std::vector<IO>, StatementBinder> Gather(stripe::Statement* stmt, const stripe::Location& loc,
                                                            std::map<RefInfoKey, RefInfo>* ri_map) {
    IOGatherer visitor{loc, ri_map};
    stmt->Accept(&visitor);
    return std::make_pair(std::move(visitor.ios_), std::move(visitor.binder_));
  }

 private:
  IOGatherer(const stripe::Location& loc, std::map<RefInfoKey, RefInfo>* ri_map) : loc_{&loc}, ri_map_{ri_map} {}

  RefInfo* FindDirectRefInfo(const std::string& name) { return &ri_map_->at(name); }

  void Visit(stripe::Load* load) final {
    auto* ri = FindDirectRefInfo(load->from);
    ios_.emplace_back(ri, stripe::RefDir::In);
    binder_ = StatementBinder{std::vector<std::pair<std::string*, RefInfo*>>{{&load->from, ri}}};
  }

  void Visit(stripe::Store* store) final {
    auto* ri = FindDirectRefInfo(store->into);
    ios_.emplace_back(ri, stripe::RefDir::Out);
    binder_ = StatementBinder{std::vector<std::pair<std::string*, RefInfo*>>{{&store->into, ri}}};
  }

  void Visit(stripe::LoadIndex* load_index) final {}

  void Visit(stripe::Constant*) final {}

  void Visit(stripe::Special* special) final {
    // TODO: Handle the case where a special accesses a single tensor multiple times.
    std::vector<std::pair<std::string*, RefInfo*>> updates;
    std::unordered_map<RefInfo*, stripe::RefDir> accesses;
    for (auto nit = special->inputs.begin(); nit != special->inputs.end(); ++nit) {
      auto* ri = FindDirectRefInfo(*nit);
      accesses[ri] = stripe::RefDir::In;
      updates.emplace_back(&*nit, ri);
    }
    for (auto nit = special->outputs.begin(); nit != special->outputs.end(); ++nit) {
      auto* ri = FindDirectRefInfo(*nit);
      updates.emplace_back(&*nit, ri);
      auto it_inserted = accesses.emplace(ri, stripe::RefDir::Out);
      if (!it_inserted.second) {
        it_inserted.first->second = UnionDir(it_inserted.first->second, stripe::RefDir::Out);
      }
    }
    ios_ = std::vector<IO>{accesses.cbegin(), accesses.cend()};
    binder_ = StatementBinder{std::move(updates)};
  }

  void Visit(stripe::Intrinsic*) final {}

  void Visit(stripe::Block* block) final {
    std::vector<std::pair<stripe::Refinement*, RefInfo*>> updates;
    for (auto& ref : block->refs) {
      if (ref.dir == stripe::RefDir::None) {
        continue;  // This isn't an IO ref.
      }
      auto* ri = &ri_map_->at(ref.from);
      updates.emplace_back(std::make_pair(&ref.mut(), ri));
      ios_.emplace_back(ri, ref);
    }
    binder_ = StatementBinder{std::move(updates), block, loc_};
  }

  const stripe::Location* loc_;
  std::map<RefInfoKey, RefInfo>* ri_map_;
  std::vector<IO> ios_;
  StatementBinder binder_;
};

// The scheduler class itself.
class Scheduler {
 public:
  static void Schedule(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options);

 private:
  // Builds a map for looking up RefInfos for a given block access.
  static std::map<RefInfoKey, RefInfo> BuildRefInfoMap(stripe::Block* block, const AliasMap* alias_map);

  Scheduler(const AliasMap* alias_map, stripe::Block* block, const proto::SchedulePass& options);

  // Runs the scheduler over its block.
  void Run();

  // Pre-initializes useful datastructures for placing:
  // * A prototype plan containing placements for every cache entry
  //   that's already been established by a runtime-future Statement,
  // * A vector of IOs that need to be placed for the current Statement.
  std::tuple<PlacementPlan, std::vector<IO>> GatherPlacementState(const std::vector<IO>& ios);

  // Makes a placement plan, trying several strategies.
  boost::optional<PlacementPlan> TryMakePlan(stripe::Block* current_block, const std::vector<IO>& ios);

  // Determines the memory units on which a tensor is allowed to exist.
  std::vector<stripe::Affine> GetValidUnits(PlacementPlan* plan, PlacementPlan::iterator it);

  // Attempts to augment a placement plan using the supplied ranges.
  bool TryPlaceInRanges(PlacementPlan* plan, const std::vector<std::pair<PlacementKey, Placement>>& placements,
                        std::map<stripe::Affine, std::list<MemRange>> ranges);

  // Attempts to make a placement plan that preserves the current
  // Statement's existing inputs and outputs, and does not collide
  // with any previously-scheduled CacheEntry unless that CacheEntry
  // has a writer (i.e. does not require swap-in).
  boost::optional<PlacementPlan> TryMakePlanWithNoSwaps(const PlacementPlan& existing_entry_plan,
                                                        const std::vector<std::pair<PlacementKey, Placement>>& todos);

  // Attempts to make a placement plan that preserves the current
  // Statement's existing inputs and outputs, but allows collisions
  // with previously-scheduled CacheEntries (producing swap-ins).
  boost::optional<PlacementPlan> TryMakePlanWithSwaps(const PlacementPlan& existing_entry_plan,
                                                      const std::vector<std::pair<PlacementKey, Placement>>& todos);

  // Makes an worst-possible-case placement plan, by scheduling
  // without regard to existing entries.  This is guaranteed to work
  // iff the shape of every refinement can simultaneously fit into
  // memory, but isn't guaranteed to be optimal at all.
  boost::optional<PlacementPlan> TryMakeFallbackPlan(const std::vector<std::pair<PlacementKey, Placement>>& placements);

  // Schedules a swap-in operation:
  // * Adds a swap-in block just before the supplied iterator,
  // * Sets the swap-in block to be the writer of the target,
  // * Adds the swap-in block to its source refinement's set of
  //   swap-in-readers,
  // * Gives all readers of the target a dependency on the swap-in block, and
  // * Returns the iterator to the swap-in block.
  //
  // If the swap-in block should have a dependency on something, it's
  // up to the caller to add it.
  //
  // Note that there's no need to give the swap-in a dependency on the
  // supplied Statement -- that will happen automatically, since the
  // swap-in will need to have a dependency on *all* accessors of the
  // new CacheEntry that's overlapping the target.
  //
  // Note that there's no need to clear the readers of the target,
  // although it wouldn't hurt if we did so.  The reason is that when
  // we add the SwapIn, there will be no subsequently-added writers of
  // the target; other accessors of the same underlying value will
  // access it via a different (newly created) CacheEntry.
  stripe::StatementIt ScheduleSwapIn(stripe::StatementIt si, CacheEntry* ent);

  // Schedules a swap-out operation:
  // * Adds a swap-out block just before the supplied iterator,
  // * Gives the swap-in readers a dependency on the swap-out block,
  // * Sets the saw_final_write flag in the source ref, and
  // * Returns the iterator to the new block, so that it can be added as a
  //   dependency to all previously-scheduled writers of overlapping memory.
  //
  // If the swap-out block should have a dependency on something, it's
  // up to the caller to add it.
  stripe::StatementIt ScheduleSwapOut(stripe::StatementIt si, CacheEntry* ent,
                                      const std::unordered_set<stripe::Statement*>* swap_in_readers);

  // Schedules a swap-in operation at the beginning of a sub-block.
  void AddSubblockSwapIn(stripe::Block* block, CacheEntry* ent, const std::string& backing_ref_name,
                         const std::vector<stripe::Affine>& access);

  // Schedules a swap-out operation at the end of a sub-block.
  void AddSubblockSwapOut(stripe::Block* block, CacheEntry* ent, const std::string& backing_ref_name,
                          const std::vector<stripe::Affine>& access);

  // Rebuilds the scheduler's block's transitive dependencies -- the
  // deps computed directly by scheduling are conservative.
  void RebuildTransitiveDeps();

  stripe::Block* block_;
  proto::SchedulePass options_;
  stripe::Location mem_loc_;
  std::size_t mem_bytes_;
  std::size_t alignment_;
  stripe::Location xfer_loc_;
  std::map<RefInfoKey, RefInfo> ri_map_;
  std::unordered_map<stripe::Refinement*, std::vector<RefInfo*>> base_ref_aliases_;
  const AliasMap* alias_map_;

  // A list of all of the CacheEntries we create during Run().  These
  // will be converted into Refinements at the end of scheduling.
  std::list<CacheEntry> cache_entries_;

  // The currently-active CacheEntries, grouped by affine, and ordered
  // by starting offset -- i.e. for each affine, the list of
  // CacheEntries that the runtime-future is expecting to have
  // available to it.  This is used for finding holes for new
  // CacheEntries.  Note that there may be overlaps, and there may be
  // duplicates (multiple CacheEntry objects for the same backing
  // refinement), and that these CacheEntries may not be valid for the
  // current statement to use -- valid CacheEntries must be found via
  // ri_map_.
  //
  // Entries are removed from this list when their memory is
  // completely covered by subsequently-created CacheEntries -- i.e. a
  // runtime-future CacheEntry does not need to have dependencies on
  // the accessors of a currently-being-scheduled CacheEntry if some
  // set of CacheEntries scheduled between them completely cover the
  // runtime-future CacheEntry; the CacheEntries in that covering set
  // will have already added dependencies to the accessors of the
  // runtime-future CacheEntry.
  std::map<stripe::Affine, std::list<CacheEntry*>> active_entries_;

  // The set of valid units.
  std::set<stripe::Affine> units_;

  // The color interference graph (used when a coloring algorithm is
  // enabled).
  ColorGraph color_graph_;
};

void Scheduler::Schedule(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options) {
  Scheduler{&alias_map, block, options}.Run();
}

std::map<RefInfoKey, RefInfo> Scheduler::BuildRefInfoMap(stripe::Block* block, const AliasMap* alias_map) {
  std::map<RefInfoKey, RefInfo> ri_map;
  // Add the current block's refs.
  for (auto& ref : block->refs) {
    const AliasInfo& ai = alias_map->at(ref.into());
    ri_map.emplace(ref.into(), RefInfo{&ref.mut(), ai});
  }

  // Update earliest-writer entries.
  for (auto& stmt : block->stmts) {
    for (const auto& written_ref_name : stmt->buffer_writes()) {
      auto& ri = safe_at(&ri_map, written_ref_name);
      if (!ri.earliest_writer) {
        ri.earliest_writer = stmt.get();
      }
    }
  }
  return ri_map;
}

Scheduler::Scheduler(const AliasMap* alias_map, stripe::Block* block, const proto::SchedulePass& options)
    : block_{block},
      options_{options},
      mem_loc_(stripe::FromProto(options.mem_loc())),
      mem_bytes_{options.mem_kib() * 1024},
      alignment_{options.alignment() ? options.alignment() : kDefaultAlignment},
      xfer_loc_(stripe::FromProto(options.xfer_loc())),
      ri_map_{BuildRefInfoMap(block, alias_map)},
      alias_map_(alias_map) {
  for (auto& rikey_ri : ri_map_) {
    RefInfo* ri = &rikey_ri.second;
    std::vector<RefInfo*>* aliases = &base_ref_aliases_[ri->alias_info.base_ref];
    aliases->emplace_back(ri);
    ri->aliases = aliases;
  }

  switch (options.mem_assignment_algorithm_case()) {
    case proto::SchedulePass::kColorIoUnique:
      for (auto& key_ri : ri_map_) {
        key_ri.second.color_vertex = color_graph_.add_vertex();
      }
      for (auto& ps : block_->stmts) {
        auto inputs = ps->buffer_reads();
        for (auto iia = inputs.begin(); iia != inputs.end(); ++iia) {
          for (auto iib = iia + 1; iib != inputs.end(); ++iib) {
            color_graph_.add_edge(safe_at(ri_map_, *iia).color_vertex, safe_at(ri_map_, *iib).color_vertex);
          }
        }
        auto outputs = ps->buffer_writes();
        for (auto oia = outputs.begin(); oia != outputs.end(); ++oia) {
          for (auto oib = oia + 1; oib != outputs.end(); ++oib) {
            color_graph_.add_edge(safe_at(ri_map_, *oia).color_vertex, safe_at(ri_map_, *oib).color_vertex);
          }
        }
      }
      for (auto unit : options.color_io_unique().units()) {
        units_.emplace(unit);
      }
      break;
    case proto::SchedulePass::MEM_ASSIGNMENT_ALGORITHM_NOT_SET:
      for (auto& key_ri : ri_map_) {
        stripe::Affine unit = 0;
        if (key_ri.second.ref.cache_unit) {
          unit = *key_ri.second.ref.cache_unit;
        }
        units_.emplace(unit);
      }
      break;
  }
}

void Scheduler::Run() {
  // The main scheduling loop.
  //
  // N.B. At the start of the loop, si points to one-past the
  //      statement that we're about to schedule, so we decrement it
  //      at the top of the loop (after the condition check) rather
  //      than in the normal loop continuation statement (which
  //      happens before the condition check).
  for (stripe::StatementIt si = block_->stmts.end(); si != block_->stmts.begin();) {
    auto si_next = si;
    --si;

    stripe::Block* current_block = dynamic_cast<stripe::Block*>(si->get());

    if (VLOG_IS_ON(2)) {
      if (current_block) {
        VLOG(2) << "Scheduling " << current_block->name;
      } else {
        VLOG(2) << "Scheduling " << si->get();
      }
    }

    // Build the vector of IOs performed by this statement.
    std::vector<IO> ios;
    StatementBinder binder;
    std::tie(ios, binder) = IOGatherer::Gather(si->get(), mem_loc_, &ri_map_);

    // Add swap-ins for any existing CacheEntries that are invalidated
    // by scheduling this statement.
    std::unordered_map<RefInfo*, std::unordered_set<stripe::Statement*>> ri_writer_swap_in_readers;
    for (const auto& io : ios) {
      if (!IsWriteDir(io.dir)) {
        continue;
      }
      RefInfo* ri = io.ri;
      auto* ri_writer_swap_in_readers_set = &ri_writer_swap_in_readers[ri];
      for (RefInfo* alias_ri : *ri->aliases) {
        if ((alias_ri == ri) || AliasInfo::Compare(ri->alias_info, alias_ri->alias_info) != AliasType::None) {
          // All accesses to alias_ri will depend on this write.
          if ((alias_ri != ri) && alias_ri->cache_entry) {
            si_next = ScheduleSwapIn(si_next, alias_ri->cache_entry);
            alias_ri->cache_entry = nullptr;
          }

          // Copy all current swap-in readers -- note that this
          // includes the current RefInfo's swap-in-readers.
          for (stripe::Statement* swap_in_reader : alias_ri->swap_in_readers) {
            ri_writer_swap_in_readers_set->emplace(swap_in_reader);
          }
        }
      }
    }

    // Figure out where we're going to put any newly-created CacheEntries.
    boost::optional<PlacementPlan> plan_option = TryMakePlan(current_block, ios);
    if (!plan_option) {
      LOG(WARNING) << "Failed to create placement plan fitting within " << (mem_bytes_ / 1024)
                   << " KiB memory boundary";
      if (current_block) {
        LOG(WARNING) << "Block " << current_block->name << " simultaneously requires:";
      } else {
        LOG(WARNING) << "The program simultaneously requires:";
      }
      for (const auto& io : ios) {
        LOG(WARNING) << "  " << stripe::PrintRefinement{io.ri->ref, current_block};
      }
      throw_with_trace(error::ResourceExhausted{"Program requires more memory than is available"});
    }

    PlacementPlan plan = std::move(plan_option).value();

    // For each input in the plan:
    //
    //   Either there's an existing CacheEntry where we can expect to
    //   find it (i.e. something in the runtime-future also needs the
    //   value), or we need to create one.
    //
    //   Either way, we need to add the current Statement to the
    //   dependency set of all runtime-future writers of memory
    //   covered by the CacheEntry, since those writers must not run
    //   until the current Statement completes (even if at scheduling
    //   time we already created the CacheEntry because some
    //   previously-scheduled Statement read from it).
    //
    //   If we're creating a CacheEntry, we may be using memory that
    //   will be overwritten by runtime-future CacheEntries (which we
    //   can observe via the per-affine active entries list).  So
    //   there's a little more processing to do.
    //
    //   For each runtime-future CacheEntry that is going to overwrite
    //   our newly-created CacheEntry:
    //
    //     * We subtract our current CacheEntry from the future
    //       CacheEntry's range (possibly removing it from its
    //       affine's active entries)
    //
    //     * If the future CacheEntry doesn't have a writer, we give
    //       it one, by adding a swap-in.
    //
    // For each output in the plan:
    //
    //   Either there's an existing CacheEntry (accessed via ri_map_)
    //   where runtime-future statements will be expecting to find the
    //   output, or there isn't, and we need to create one.
    //
    //   If we need to create one: in addition to the creation rules
    //   for input CacheEntries, there may also be runtime-future
    //   Statements that depend on the value; they will have already
    //   allocated CacheEntries for the value, which will have been
    //   overwritten by this point (otherwise we would've found the
    //   most recent via ri_map_), and the overwriters will have
    //   created swap-in Statements as needed to fill in the
    //   appropriate CacheEntries.  So we need to schedule a swap-out
    //   to initialize the backing memory, and add a dependency from
    //   those swap-ins to the swap-out.
    //
    //   We also need to schedule a swap-out if the backing memory is
    //   an out/inout Refinement in the current Block -- presumably
    //   the parent Block is wanting the value, even if the value is
    //   not going to be used by a runtime-future Statement within the
    //   current Block.

    std::map<stripe::Affine, std::list<CacheEntry*>> added_entries;

    std::vector<stripe::Refinement> added_refs;
    std::unordered_map<RefInfo*, std::string> internal_swap_backing_ref_names;

    // TODO: There's a straightforward way of walking the plan's
    // placements and the existing active_entries_ at the same
    // time, saving a lot of comparisons.  We don't bother for now,
    // but only because it's a little complicated to get right and
    // premature optimization is the root of all evil, but if we
    // observe a lot of comparisons being done via RangesOverlap(), we
    // have a way to fix it.

    for (auto& pkey_placement : plan) {
      RefInfo* ri = pkey_placement.first.ri;
      IVLOG(3, "Applying placement for " << ri->name);
      auto& placement = pkey_placement.second;

      CacheEntry* ent = placement.entry;
      bool is_new_entry = (ent == nullptr);

      if (is_new_entry) {
        // This Placement requires a new entry.
        ent = &*cache_entries_.emplace(cache_entries_.end(), CacheEntry{pkey_placement});
        IVLOG(3, "Created cache entry " << ent->name << " at " << ent->range << " with affine=" << ent->unit
                                        << " shape=" << ent->shape << " is_internal=" << ent->is_internal);
        placement.entry = ent;
        ri->cache_entry = ent;
      }

      stripe::StatementIt reuse_dep = si;

      if (placement.is_internal) {
        // This CacheEntry reserves temporary cache space within a
        // serialized sub-statement (which must be a Block).  So we
        // need to insert swap-in and swap-out instructions into the
        // block.
        //
        // TODO: Once we have Block serialization flag support, make
        // sure to set current_block->is_serialized = true

        // We need to make sure the inner block can access the backing refinement.
        std::string internal_swap_backing_ref_name;
        auto it = internal_swap_backing_ref_names.find(ri);
        if (it != internal_swap_backing_ref_names.end()) {
          internal_swap_backing_ref_name = it->second;
        } else {
          internal_swap_backing_ref_name = current_block->unique_ref_name(ri->name + "_storage");
          internal_swap_backing_ref_names[ri] = internal_swap_backing_ref_name;
          added_refs.push_back(stripe::Refinement{
              placement.dir,                   // dir
              ent->source->ref.into(),         // from
              internal_swap_backing_ref_name,  // into
              ent->source->alias_info.access,  // access
              ent->source->alias_info.shape,   // interior_shape
              "",                              // agg_op
              ent->source->ref.location,       // location
              0,                               // offset
              ent->source->ref.bank_dim,       // bank_dim
          });
        }
        if (stripe::IsReadDir(placement.dir)) {
          AddSubblockSwapIn(current_block, ent, internal_swap_backing_ref_name, pkey_placement.first.access);
        }
        if (stripe::IsWriteDir(placement.dir)) {
          AddSubblockSwapOut(current_block, ent, internal_swap_backing_ref_name, pkey_placement.first.access);
        }
      } else {
        // This CacheEntry may be reused between multiple sub-statements.
        // Add dependency tracking information and swaps as needed.
        if (IsWriteDir(placement.dir)) {
          for (auto& reader_aliasinfo : ent->readers) {
            if (AliasInfo::Compare(ri->alias_info, reader_aliasinfo.second) != AliasType::None) {
              reader_aliasinfo.first->deps.emplace_back(si);
            }
          }

          ent->writers.emplace(si->get(), ri->alias_info);
          if (si->get() == ent->source->earliest_writer) {
            ent->saw_earliest_writer = true;
          }
        }

        if (IsReadDir(placement.dir)) {
          ent->readers.emplace(si->get(), ri->alias_info);
        }

        ent->first_accessor = si;

        // Determine whether this CacheEntry will need to be swapped
        // out, setting up reuse_dep to be the dependency that
        // overlapping CacheEntry objects will use.

        if (IsWriteDir(placement.dir) &&
            ((IsWriteDir(ri->ref.dir) && !ri->saw_final_write) || !ri_writer_swap_in_readers[ri].empty())) {
          IVLOG(3, "  Adding swap-out for " << ent->name << " at " << ent->range);
          IVLOG(3, "    IsWriteDir(): " << IsWriteDir(ri->ref.dir));
          IVLOG(3, "    SawFinalWrite(): " << ri->saw_final_write);
          IVLOG(3, "    Swap-in-readers.empty(): " << ri->swap_in_readers.empty());
          auto next_si = si;
          ++next_si;
          reuse_dep = ScheduleSwapOut(next_si, ent, &ri_writer_swap_in_readers[ri]);
          (*reuse_dep)->deps.emplace_back(si);
        }
      }

      // Add dependency tracking information for all
      // previously-created CacheEntries whose ranges overlap the
      // current CacheEntry.
      //
      // N.B. After the SubtractRange() call, we may remove future_ent
      // from its active_entries_ list.  To ensure that our iteration
      // is safe, we explicitly manage it, and make sure to advance
      // the iterator prior to the post-SubtractRange() removal.
      auto& active_entlist = active_entries_[ent->unit];
      for (auto fit = active_entlist.begin(); fit != active_entlist.end();) {
        CacheEntry* future_ent = *fit;
        ++fit;
        if (future_ent == ent || !RangesOverlap(ent->range, future_ent->uncovered_ranges)) {
          continue;
        }

        if (is_new_entry) {
          IVLOG(3, "New entry " << ent->name << " at " << ent->range << " collides with existing entry "
                                << future_ent->name << " at " << future_ent->range);
          if (!future_ent->saw_earliest_writer) {
            auto next_it = reuse_dep;
            ++next_it;
            IVLOG(3, "  Adding swap-in for " << future_ent->name << " at " << future_ent->range);
            ScheduleSwapIn(next_it, future_ent);
          }
          for (auto& writer_aliasinfo : future_ent->writers) {
            writer_aliasinfo.first->deps.emplace_back(reuse_dep);
          }
          SubtractRange(ent->range, &future_ent->uncovered_ranges);
          if (future_ent->uncovered_ranges.empty()) {
            IVLOG(3, "  Existing entry " << future_ent->name
                                         << " is now completely covered; removing from active entries");
            IVLOG(3, "    Active iterator is " << &*future_ent->active_iterator << " active_entlist is at "
                                               << &active_entlist << ", contains:");
            if (VLOG_IS_ON(3)) {
              for (auto entp = active_entlist.begin(); entp != active_entlist.end(); ++entp) {
                IVLOG(3, "    " << &*entp << ": " << (*entp)->name << " at " << (*entp)->range);
              }
            }
            active_entlist.erase(future_ent->active_iterator);
          }

          // Make sure we don't use this entry for accessing this ref
          // after this point.
          if (future_ent->source->cache_entry == future_ent) {
            future_ent->source->cache_entry = nullptr;
          }
        }

        for (auto& writer_aliasinfo : future_ent->writers) {
          writer_aliasinfo.first->deps.emplace_back(reuse_dep);
        }
      }

      if (is_new_entry && !placement.is_internal) {
        IVLOG(3, "Adding " << ent->name << " at " << ent->range << " to added_entries");
        auto& active_entlist = added_entries[ent->unit];
        ent->active_iterator = active_entlist.emplace(active_entlist.end(), ent);
        IVLOG(3, "  Active iterator was " << &*ent->active_iterator << "; list at " << &active_entlist
                                          << ", size=" << active_entlist.size());
      }
    }  // Plan-application loop

    IVLOG(3, "Splicing into active_entries_");
    for (auto& added_unit_entlist : added_entries) {
      auto& active_entlist = active_entries_[added_unit_entlist.first];
      active_entlist.splice(active_entlist.begin(), added_unit_entlist.second);
      active_entlist.sort([](CacheEntry* lhs, CacheEntry* rhs) { return lhs->range.begin < rhs->range.begin; });
    }

    if (VLOG_IS_ON(3)) {
      IVLOG(3, "active_entries_ now contains:");
      for (auto& affine_entlist : active_entries_) {
        IVLOG(3, "  Affine: " << affine_entlist.first);
        for (auto* ent : affine_entlist.second) {
          IVLOG(3, "    " << ent->name << " at " << ent->range);
        }
      }
    }

    binder.ApplyBindings();
    if (current_block && added_refs.size()) {
      current_block->refs.insert(added_refs.begin(), added_refs.end());
    }

    // Remove all RefInfo pointers to internal-only CacheEntries used
    // by the plan, so that they're not eligible for reuse by
    // subsequent statements.
    for (auto& pkey_placement : plan) {
      RefInfo* ri = pkey_placement.first.ri;
      if (ri->cache_entry && ri->cache_entry->is_internal) {
        ri->cache_entry = nullptr;
      }
    }
  }

  // Add swap-in writers for every active CacheEntry without a writer.
  //
  // All of the writerless CacheEntries can co-exist at the beginning
  // of the program, and we guarantee that outputs will not clobber
  // those entries before they're used.  So we can insert swap-in
  // blocks for these CacheEntries in any order, at any point in the
  // schedule before they're first used
  //
  // So: we add the swap-in for each CacheEntry just before the kernel
  // that actually uses it.  On synchronous systems, it doesn't matter
  // what order we use; on asynchronous systems, the swap-in blocks
  // have no dependencies, allowing them to execute in any order
  // anyway, but this will tend to queue them for memory transfer in
  // an order that enables the compute units to get busy ASAP.
  for (auto& affine_entlist : active_entries_) {
    for (auto* ent : affine_entlist.second) {
      if (!ent->source->earliest_writer) {
        IVLOG(3, "  Adding final swap-in for " << ent->name);
        ScheduleSwapIn(ent->first_accessor, ent);
      }
    }
  }

  // Add a Refinement for each CacheEntry.
  for (auto& ent : cache_entries_) {
    auto ref = stripe::Refinement::FromInto(ent.name);
    auto ri = block_->ref_by_into(ent.name, false);
    if (ri != block_->refs.end()) {
      ref = ri->WithInto(ent.name);
    } else {
      ref = ent.source->ref.WithInto(ent.name);
    }
    ref.dir = stripe::RefDir::None;
    ref.from.clear();
    ref.interior_shape = ent.shape;
    ref.location = PartialEval(mem_loc_, {{"unit", ent.unit.constant()}});
    ref.offset = ent.range.begin;
    block_->refs.emplace(std::move(ref));
  }

  // Move used Refinements back into the block.
  for (auto& rikey_ri : ri_map_) {
    if (rikey_ri.second.used) {
      auto ri = block_->refs.find(rikey_ri.second.ref.into());
      if (ri != block_->refs.end()) {
        block_->refs.erase(ri);
      }
      block_->refs.emplace(rikey_ri.second.ref);
    }
  }

  RebuildTransitiveDeps();
}

std::tuple<PlacementPlan, std::vector<IO>> Scheduler::GatherPlacementState(const std::vector<IO>& ios) {
  PlacementPlan plan;
  std::unordered_map<RefInfo*, stripe::RefDir> todo_map;

  for (const auto& io : ios) {
    VLOG(3) << "  Planning IO for RefInfo " << io.ri << " " << io.ri->name;
    // See whether we've already created a Placement for this ref.
    PlacementKey pkey{io.ri, io.ri->exterior_cache_shape, {}};
    auto it = plan.find(pkey);
    if (it != plan.end()) {
      // We've already made a Placement; add in our direction, and we're done.
      it->second.dir = UnionDir(it->second.dir, io.dir);
      continue;
    }

    // See whether we already have an active CacheEntry for this IO.
    if (io.ri->cache_entry && !io.ri->cache_entry->saw_earliest_writer) {
      // We do -- create a Placement describing it.
      plan.emplace(pkey, Placement{io.dir, io.ri->cache_entry->range, io.ri->cache_entry});
      continue;
    }

    // Otherwise, we're going to need to allocate a Placement.  We'll
    // do it after processing all inputs, so that we can do placement
    // in size order with correct directions.
    auto it_inserted = todo_map.emplace(io.ri, io.dir);
    if (!it_inserted.second) {
      it_inserted.first->second = UnionDir(it_inserted.first->second, io.dir);
    }
  }

  if (options_.mem_assignment_algorithm_case() == proto::SchedulePass::kColorIoUnique) {
    // Now that we have our directions initialized, perform color
    // checks on the entries within the plan.  When we detect
    // conflicting entries, turn them into todos.
    std::set<stripe::Affine> used_input_units;
    std::set<stripe::Affine> used_output_units;
    auto it = plan.begin();
    while (it != plan.end()) {
      bool color_valid = true;
      if (stripe::IsReadDir(it->second.dir)) {
        color_valid &= used_input_units.emplace(it->first.ri->cache_entry->unit).second;
      }
      if (stripe::IsWriteDir(it->second.dir)) {
        color_valid &= used_output_units.emplace(it->first.ri->cache_entry->unit).second;
      }
      if (!color_valid) {
        // Turn the current placement into a todo.
        todo_map.emplace(it->first.ri, it->second.dir);
        it = plan.erase(it);
        continue;
      }
      ++it;
    }
  }

  // Return the placements to be made.
  std::vector<IO> todos;
  for (auto& refinfo_refdir : todo_map) {
    todos.emplace_back(refinfo_refdir.first, refinfo_refdir.second);
  }

  return std::make_tuple(std::move(plan), std::move(todos));
}

std::vector<std::pair<PlacementKey, Placement>> MakeFullPlacements(const std::vector<IO>& ios) {
  std::vector<std::pair<PlacementKey, Placement>> result;
  for (const auto& io : ios) {
    result.emplace_back(PlacementKey{io.ri, io.ri->exterior_cache_shape, {}},
                        Placement{io.dir, io.ri->size, false, ""});
  }

  // Sort the placements, largest-first, using the underlying
  // refinement name as the tiebreaker.
  std::sort(result.begin(), result.end(),
            [](const std::pair<PlacementKey, Placement>& lhs, const std::pair<PlacementKey, Placement>& rhs) {
              return std::tie(rhs.second.size, rhs.first.ri->name) < std::tie(lhs.second.size, lhs.first.ri->name);
            });

  return result;
}

std::vector<std::pair<PlacementKey, Placement>> MakePartialPlacements(const std::vector<IO>& ios) {
  IVLOG(3, "  MakePartialPlacements>");
  std::vector<std::pair<PlacementKey, Placement>> result;
  for (const auto& io : ios) {
    std::size_t interior_size = stripe::Codec::Resolve(io.interior_shape)->byte_size();
    bool is_internal = interior_size != io.ri->size;
    IVLOG(3, "      " << io.ri->name << " shape=" << io.interior_shape << " interior_size=" << interior_size
                      << " external_size=" << io.ri->size << " is_internal=" << is_internal);
    std::vector<stripe::Affine> access;
    if (is_internal) {
      access = io.access;
    }
    result.emplace_back(PlacementKey{io.ri, io.interior_shape, access},
                        Placement{io.dir, interior_size, is_internal, io.interior_name});
  }

  // Sort the placements, largest-first, using the underlying
  // refinement name as the tiebreaker.
  std::sort(result.begin(), result.end(),
            [](const std::pair<PlacementKey, Placement>& lhs, const std::pair<PlacementKey, Placement>& rhs) {
              return std::tie(rhs.second.size, rhs.first.ri->name) < std::tie(lhs.second.size, lhs.first.ri->name);
            });

  return result;
}

boost::optional<PlacementPlan> Scheduler::TryMakePlan(stripe::Block* current_block, const std::vector<IO>& ios) {
  // Initialize useful planning inputs.
  PlacementPlan existing_entry_plan;
  std::vector<IO> todos;
  std::tie(existing_entry_plan, todos) = GatherPlacementState(ios);

  if (VLOG_IS_ON(3)) {
    IVLOG(3, "  Existing entries in plan:");
    for (auto& pkey_placement : existing_entry_plan) {
      IVLOG(3, "    " << pkey_placement.first.ri->name << " -> " << pkey_placement.second);
    }
    IVLOG(3, "  ToDos:");
    for (auto& io : todos) {
      auto isize = stripe::Codec::Resolve(io.interior_shape)->byte_size();
      IVLOG(3, "      Ref=" << io.ri->name << " size=" << io.ri->size << " isize=" << isize);
    }
  }

  auto todo_fulls = MakeFullPlacements(todos);
  auto todo_partials = MakePartialPlacements(todos);

  boost::optional<PlacementPlan> plan;

  plan = TryMakePlanWithNoSwaps(existing_entry_plan, todo_fulls);
  if (plan) {
    IVLOG(2, "  Made plan with full IO and no swaps");
    return *plan;
  }

  plan = TryMakePlanWithNoSwaps(existing_entry_plan, todo_partials);
  if (plan) {
    IVLOG(2, "  Made plan with loop IO and no swaps");
    return *plan;
  }

  plan = TryMakePlanWithSwaps(existing_entry_plan, todo_fulls);
  if (plan) {
    IVLOG(2, "  Made plan with full IO and swaps");
    return plan;
  }

  plan = TryMakePlanWithSwaps(existing_entry_plan, todo_partials);
  if (plan) {
    IVLOG(2, "  Made plan with loop IO and swaps");
    return plan;
  }

  plan = TryMakeFallbackPlan(MakeFullPlacements(ios));
  if (plan) {
    IVLOG(2, "  Made no-loop plan ignoring existing entries");
    return plan;
  }

  if (current_block) {
    plan = TryMakeFallbackPlan(MakePartialPlacements(ios));
    if (plan) {
      IVLOG(2, "  Made looping plan ignoring existing entries");
      return plan;
    }
  }

  IVLOG(2, "  Failed to make plan");
  return boost::none;
}

std::vector<stripe::Affine> Scheduler::GetValidUnits(PlacementPlan* plan, PlacementPlan::iterator it) {
  switch (options_.mem_assignment_algorithm_case()) {
    case proto::SchedulePass::kColorIoUnique: {
      std::set<stripe::Affine> used_units;
      for (auto pit = plan->begin(); pit != plan->end(); ++pit) {
        if (pit == it) {
          continue;
        }
        used_units.emplace(pit->second.unit);
      }
      for (const auto& unit_entlist : active_entries_) {
        for (CacheEntry* ent : unit_entlist.second) {
          if (edge(it->first.ri->color_vertex, ent->source->color_vertex, color_graph_).second) {
            used_units.emplace(ent->unit);
          }
        }
      }

      std::vector<stripe::Affine> result;
      std::set_difference(units_.begin(), units_.end(), used_units.begin(), used_units.end(),
                          std::back_inserter(result));
      return result;
    }
    case proto::SchedulePass::MEM_ASSIGNMENT_ALGORITHM_NOT_SET:
      // For the default algorithm, we use the unit from the original
      // refinement, which may be explicitly specified.
      if (it->first.ri->ref.cache_unit) {
        return std::vector<stripe::Affine>{*it->first.ri->ref.cache_unit};
      }
      return std::vector<stripe::Affine>{0};
  }

  // Appease awful compilers.
  return std::vector<stripe::Affine>{};
}

bool Scheduler::TryPlaceInRanges(PlacementPlan* plan, const std::vector<std::pair<PlacementKey, Placement>>& placements,
                                 std::map<stripe::Affine, std::list<MemRange>> unit_ranges) {
  // For each IO in largest->smallest size, determine a placement.
  // For each one, we want to pick the smallest free range that is
  // still big enough to hold the IO.
  IVLOG(3, "      Looking for placements");
  for (const auto& pkey_placement : placements) {
    auto it_inserted = plan->emplace(pkey_placement.first, pkey_placement.second);
    if (it_inserted.second) {
      // A new Placement.
      std::size_t size = pkey_placement.second.size;
      IVLOG(3, "        Finding placement for " << pkey_placement.first.ri->name << ", size=" << size);
      boost::optional<std::pair<stripe::Affine, std::list<MemRange>::iterator>> best_so_far;
      std::size_t best_waste_so_far = mem_bytes_;
      for (const auto& unit : GetValidUnits(plan, it_inserted.first)) {
        IVLOG(3, "          Checking free space on unit " << unit);
        auto& ranges = unit_ranges[unit];
        for (auto rit = ranges.begin(); rit != ranges.end(); ++rit) {
          IVLOG(3, "            Considering range " << *rit);
          if (rit->size() < size) {
            continue;
          }
          std::size_t waste = rit->size() - size;
          if (best_waste_so_far <= waste) {
            continue;
          }
          IVLOG(3, "          Range " << *rit << " is the best so far");
          best_so_far = std::make_pair(unit, rit);
          best_waste_so_far = waste;
        }
      }
      if (!best_so_far) {
        IVLOG(3, "          No placement available for " << pkey_placement.first.ri->name << ", size=" << size);
        return false;
      }
      auto assigned_range = MemRange{best_so_far->second->begin, best_so_far->second->begin + size};
      SubtractRange(assigned_range, &unit_ranges[best_so_far->first], best_so_far->second);
      it_inserted.first->second.unit = best_so_far->first;
      it_inserted.first->second.range = assigned_range;
    } else {
      // An existing Placement.
      it_inserted.first->second.dir = UnionDir(it_inserted.first->second.dir, pkey_placement.second.dir);
    }
  }

  return true;
}

boost::optional<PlacementPlan> Scheduler::TryMakePlanWithNoSwaps(
    const PlacementPlan& existing_entry_plan, const std::vector<std::pair<PlacementKey, Placement>>& todos) {
  PlacementPlan plan{existing_entry_plan};
  std::map<stripe::Affine, std::list<MemRange>> unit_ranges;

  // Build a list of the available ranges.  For our purposes, a range
  // is available if it already has an initial writer (=> it is not
  // going to require a swap-in), and if its RefInfo is not already in
  // the plan (because RefInfos that are in the plan are required by
  // the current statement).
  for (const auto& unit : units_) {
    auto& ranges = unit_ranges[unit];
    ranges = std::list<MemRange>{MemRange{0, mem_bytes_}};
    for (auto* ent : active_entries_[unit]) {
      PlacementKey pkey{ent->source, ent->source->exterior_cache_shape, {}};
      IVLOG(3, "      Saw range " << ent->range << " used by " << ent->name << " saw_earliest_writer="
                                  << ent->saw_earliest_writer << " plan.count=" << plan.count(pkey));
      if (!(ent->saw_earliest_writer && !plan.count(pkey))) {
        IVLOG(3, "      Subtracting range " << ent->range << " used by " << ent->name);
        SubtractRange(ent->range, &ranges);
      }
    }
  }

  if (!TryPlaceInRanges(&plan, todos, std::move(unit_ranges))) {
    return boost::none;
  }

  return plan;
}

boost::optional<PlacementPlan> Scheduler::TryMakePlanWithSwaps(
    const PlacementPlan& existing_entry_plan, const std::vector<std::pair<PlacementKey, Placement>>& todos) {
  PlacementPlan plan{existing_entry_plan};
  std::map<stripe::Affine, std::list<MemRange>> unit_ranges;

  // Build a list of the available ranges.  For our purposes, a range
  // is available as long as its RefInfo is not already in the plan
  // (because RefInfos that are in the plan can be used by the current
  // statement).
  for (const auto& unit : units_) {
    auto& ranges = unit_ranges[unit];
    ranges = std::list<MemRange>{MemRange{0, mem_bytes_}};
    for (auto* ent : active_entries_[unit]) {
      PlacementKey pkey{ent->source, ent->source->exterior_cache_shape, {}};
      IVLOG(3, "      Saw range " << ent->range << " used by " << ent->name << " saw_earliest_writer="
                                  << ent->saw_earliest_writer << " plan.count=" << plan.count(pkey));
      if (plan.count(pkey)) {
        IVLOG(3, "      Subtracting range " << ent->range << " used by " << ent->name);
        SubtractRange(ent->range, &ranges);
      }
    }
  }

  if (!TryPlaceInRanges(&plan, todos, std::move(unit_ranges))) {
    return boost::none;
  }

  return plan;
}

boost::optional<PlacementPlan> Scheduler::TryMakeFallbackPlan(
    const std::vector<std::pair<PlacementKey, Placement>>& placements) {
  // TODO: Consider pipelining and small-group parallel processing.
  //       There's an interesting tradeoff here: increased parallelism
  //       means we have less memory to hold cross-substatement data,
  //       which may then require additional swapping.  So we may want
  //       to schedule parallelism as part of the overall scheduling
  //       pass, rather than as a separate pass.

  PlacementPlan plan;
  std::map<stripe::Affine, std::size_t> offsets;

  for (const auto& unit : units_) {
    offsets[unit] = 0;
  }

  auto assign_placement = [&](const std::pair<PlacementKey, Placement>& pkey_placement, const stripe::Affine& unit) {
    auto it_inserted = plan.emplace(pkey_placement.first, pkey_placement.second);
    if (it_inserted.second) {
      std::size_t& offset = safe_at(&offsets, unit);
      // A new Placement.
      std::size_t size = pkey_placement.second.size;
      it_inserted.first->second.range.begin = offset;
      it_inserted.first->second.range.end = offset + size;
      offset += math::Align(size, alignment_);
      IVLOG(3, "      Placed " << pkey_placement.first.ri->name << " at " << it_inserted.first->second.range
                               << ", next=" << offset);
    } else {
      // An existing Placement.
      it_inserted.first->second.dir = UnionDir(it_inserted.first->second.dir, pkey_placement.second.dir);
    }
  };

  switch (options_.mem_assignment_algorithm_case()) {
    case proto::SchedulePass::kColorIoUnique: {
      // Assign all inputs, rotating through the memory units, so that
      // we guarantee they have unique colors.
      if (placements.size() > units_.size()) {
        // This can't be scheduled with IO coloring -- there are more
        // placements than units.
        return boost::none;
      }
      std::set<stripe::Affine> used_output_units;
      auto unit_it = units_.begin();
      for (const auto& pkey_placement : placements) {
        if (stripe::IsReadDir(pkey_placement.second.dir)) {
          assign_placement(pkey_placement, *unit_it);
          if (stripe::IsWriteDir(pkey_placement.second.dir)) {
            used_output_units.emplace(*unit_it);
          }
          ++unit_it;
        }
      }
      for (const auto& pkey_placement : placements) {
        if (!stripe::IsReadDir(pkey_placement.second.dir)) {
          // Assign all non-inputs, rotating through the memory units,
          // using the first unit unused by earlier outputs that fits.
          // Note that since the placements are ordered by size
          // (largest-first), this is guaranteed to succeed if success
          // is possible.
          auto required = math::Align(pkey_placement.second.size, alignment_);
          bool found_unit = false;
          for (const auto& unit_offset : offsets) {
            if (required <= (mem_bytes_ - unit_offset.second) && used_output_units.emplace(unit_offset.first).second) {
              assign_placement(pkey_placement, unit_offset.first);
              found_unit = true;
              break;
            }
          }
          if (!found_unit) {
            return boost::none;  // Unable to build a placement plan.
          }
        }
      }
      break;
    }

    case proto::SchedulePass::MEM_ASSIGNMENT_ALGORITHM_NOT_SET:
      // Assign all placements in order.
      for (const auto& pkey_placement : placements) {
        stripe::Affine unit = 0;
        if (pkey_placement.first.ri->ref.cache_unit) {
          unit = *pkey_placement.first.ri->ref.cache_unit;
        }
        assign_placement(pkey_placement, unit);
      }
      break;
  }

  // Verify the plan.
  for (const auto& unit_offset : offsets) {
    if (mem_bytes_ < unit_offset.second) {
      return boost::none;
    }
  }

  return plan;
}

stripe::StatementIt Scheduler::ScheduleSwapIn(stripe::StatementIt si, CacheEntry* ent) {
  stripe::Block swap_block;
  ent->source->used = true;
  swap_block.name = "swap_in_" + ent->name;
  swap_block.location = xfer_loc_;
  swap_block.idxs = ent->source->swap_idxs;
  TranslateLocation(&swap_block, &swap_block.location);
  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::In,            // dir
      ent->source->ref.into(),       // from
      "src",                         // into
      ent->source->ref_swap_access,  // access
      ent->source->ref_swap_shape,   // interior_shape
      "",                            // agg_op
      ent->source->ref.location,     // location
      0,                             // offset
      ent->source->ref.bank_dim,     // bank_dim
  });

  auto banked_mem_loc = PartialEval(mem_loc_, {{"unit", ent->unit.constant()}});
  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::Out,             // dir
      ent->name,                       // from
      "dst",                           // into
      ent->source->cache_swap_access,  // access
      ent->source->cache_swap_shape,   // interior_shape
      "",                              // agg_op
      banked_mem_loc,                  // location
      0,                               // offset
      ent->source->ref.bank_dim,       // bank_dim
  });

  for (auto& ref : swap_block.refs) {
    TranslateLocation(&swap_block, &ref.mut().location);
  }

  if (options_.add_constraints()) {
    for (size_t i = 0; i < ent->source->swap_idxs.size(); i++) {
      alias_map_->AddConstraintForIndex(&swap_block, ent->source->alias_info, i, ent->source->swap_idxs[i].name);
    }
  }

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  stripe::StatementIt swap_in_it = block_->stmts.emplace(si, std::make_shared<stripe::Block>(std::move(swap_block)));
  stripe::Statement* swap_in = swap_in_it->get();
  ent->writers.emplace(swap_in, ent->source->alias_info);
  ent->source->swap_in_readers.emplace(swap_in);
  for (auto& reader_aliasinfo : ent->readers) {
    reader_aliasinfo.first->deps.emplace_back(swap_in_it);
  }
  ent->saw_earliest_writer = true;
  return swap_in_it;
}

stripe::StatementIt Scheduler::ScheduleSwapOut(stripe::StatementIt si, CacheEntry* ent,
                                               const std::unordered_set<stripe::Statement*>* swap_in_readers) {
  stripe::Block swap_block;
  ent->source->used = true;
  swap_block.name = "swap_out_" + ent->name;
  swap_block.location = xfer_loc_;
  swap_block.idxs = ent->source->swap_idxs;
  TranslateLocation(&swap_block, &swap_block.location);
  auto banked_mem_loc = PartialEval(mem_loc_, {{"unit", ent->unit.constant()}});
  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::In,              // dir
      ent->name,                       // from
      "src",                           // into
      ent->source->cache_swap_access,  // access
      ent->source->cache_swap_shape,   // interior_shape
      "",                              // agg_op
      banked_mem_loc,                  // location
      0,                               // offset
      ent->source->ref.bank_dim,       // bank_dim
  });

  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::Out,           // dir
      ent->source->ref.into(),       // from
      "dst",                         // into
      ent->source->ref_swap_access,  // access
      ent->source->ref_swap_shape,   // interior_shape
      "",                            // agg_op
      ent->source->ref.location,     // location
      0,                             // offset
      ent->source->ref.bank_dim,     // bank_dim
  });

  for (auto& ref : swap_block.refs) {
    TranslateLocation(&swap_block, &ref.mut().location);
  }

  if (options_.add_constraints()) {
    for (size_t i = 0; i < ent->source->swap_idxs.size(); i++) {
      alias_map_->AddConstraintForIndex(&swap_block, ent->source->alias_info, i, ent->source->swap_idxs[i].name);
    }
  }

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  stripe::StatementIt swap_out_it = block_->stmts.emplace(si, std::make_shared<stripe::Block>(std::move(swap_block)));
  if (swap_in_readers) {
    for (stripe::Statement* reader : *swap_in_readers) {
      reader->deps.emplace_back(swap_out_it);
    }
  }
  ent->source->saw_final_write = true;
  return swap_out_it;
}

void Scheduler::AddSubblockSwapIn(stripe::Block* block, CacheEntry* ent, const std::string& backing_ref_name,
                                  const std::vector<stripe::Affine>& access) {
  stripe::Block swap_block;
  swap_block.name = "read_slice_of_" + ent->source->name;
  swap_block.location = xfer_loc_;

  // Add indicies used by the backing storage access offset affines to
  // the swap statement.
  std::unordered_set<std::string> idxs;
  for (const auto& acc : access) {
    for (const auto& idx_val : acc.getMap()) {
      auto it_inserted = idxs.emplace(idx_val.first);
      if (it_inserted.second) {
        swap_block.idxs.emplace_back(stripe::Index{idx_val.first, 1, stripe::Affine(idx_val.first)});
      }
    }
  }

  // Build indices to describe ranging over the block.
  std::vector<stripe::Affine> local_src_access;
  std::vector<stripe::Affine> local_dst_access;
  for (std::size_t i = 0; i < access.size(); ++i) {
    std::string iname = swap_block.unique_idx_name("i" + std::to_string(i));
    swap_block.idxs.emplace_back(stripe::Index{iname, ent->shape.dims[i].size});
    local_src_access.emplace_back(stripe::Affine(iname) + access[i]);
    local_dst_access.emplace_back(stripe::Affine(iname));
    if (options_.add_constraints()) {
      alias_map_->AddConstraintForIndex(&swap_block, ent->source->alias_info, i, iname);
    }
  }

  TranslateLocation(&swap_block, &swap_block.location);

  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::In,           // dir
      backing_ref_name,             // from
      "src",                        // into
      local_src_access,             // access
      ent->source->ref_swap_shape,  // interior_shape
      "",                           // agg_op
      ent->source->ref.location,    // location
      0,                            // offset
      ent->source->ref.bank_dim,    // bank_dim
  });

  auto banked_mem_loc = PartialEval(mem_loc_, {{"unit", ent->unit.constant()}});
  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::Out,            // dir
      ent->interior_name,             // from
      "dst",                          // into
      local_dst_access,               // access
      ent->source->cache_swap_shape,  // interior_shape
      "",                             // agg_op
      banked_mem_loc,                 // location
      0,                              // offset
      ent->source->ref.bank_dim,      // bank_dim
  });

  for (auto& ref : swap_block.refs) {
    TranslateLocation(&swap_block, &ref.mut().location);
  }

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  block->stmts.emplace(block->stmts.begin(), std::make_shared<stripe::Block>(std::move(swap_block)));
}

void Scheduler::AddSubblockSwapOut(stripe::Block* block, CacheEntry* ent, const std::string& backing_ref_name,
                                   const std::vector<stripe::Affine>& access) {
  stripe::Block swap_block;
  swap_block.name = "write_slice_of_" + ent->source->name;
  swap_block.location = xfer_loc_;

  // Add indicies used by the backing storage access offset affines to
  // the swap statement.
  std::unordered_set<std::string> idxs;
  for (const auto& acc : access) {
    for (const auto& idx_val : acc.getMap()) {
      auto it_inserted = idxs.emplace(idx_val.first);
      if (it_inserted.second) {
        swap_block.idxs.emplace_back(stripe::Index{idx_val.first, 1, stripe::Affine(idx_val.first)});
      }
    }
  }

  // Build indices to describe ranging over the block.
  std::vector<stripe::Affine> local_src_access;
  std::vector<stripe::Affine> local_dst_access;
  for (std::size_t i = 0; i < access.size(); ++i) {
    std::string iname = swap_block.unique_idx_name("i" + std::to_string(i));
    swap_block.idxs.emplace_back(stripe::Index{iname, ent->shape.dims[i].size});
    local_src_access.emplace_back(stripe::Affine(iname));
    local_dst_access.emplace_back(stripe::Affine(iname) + access[i]);
    if (options_.add_constraints()) {
      alias_map_->AddConstraintForIndex(&swap_block, ent->source->alias_info, i, iname);
    }
  }

  TranslateLocation(&swap_block, &swap_block.location);

  auto banked_mem_loc = PartialEval(mem_loc_, {{"unit", ent->unit.constant()}});
  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::In,             // dir
      ent->interior_name,             // from
      "src",                          // into
      local_src_access,               // access
      ent->source->cache_swap_shape,  // interior_shape
      "",                             // agg_op
      banked_mem_loc,                 // location
      0,                              // offset
      ent->source->ref.bank_dim,      // bank_dim
  });

  swap_block.refs.emplace(stripe::Refinement{
      stripe::RefDir::Out,          // dir
      backing_ref_name,             // from
      "dst",                        // into
      local_dst_access,             // access
      ent->source->ref_swap_shape,  // interior_shape
      "",                           // agg_op
      ent->source->ref.location,    // location
      0,                            // offset
      ent->source->ref.bank_dim,    // bank_dim
  });

  for (auto& ref : swap_block.refs) {
    TranslateLocation(&swap_block, &ref.mut().location);
  }

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  block->stmts.emplace(block->stmts.end(), std::make_shared<stripe::Block>(std::move(swap_block)));
}

void Scheduler::RebuildTransitiveDeps() {
  std::unordered_map<stripe::StatementIt, std::unordered_set<stripe::StatementIt>> tdeps;
  tdeps.reserve(block_->stmts.size());

  for (auto sit = block_->stmts.begin(); sit != block_->stmts.end(); ++sit) {
    std::unordered_set<stripe::StatementIt> stmt_deps;
    std::unordered_set<stripe::StatementIt> stmt_tdeps;
    for (auto dep : (*sit)->deps) {
      stmt_deps.emplace(dep);
      stmt_tdeps.insert(tdeps[dep].begin(), tdeps[dep].end());
    }
    (*sit)->deps.clear();
    std::set_difference(stmt_deps.begin(), stmt_deps.end(), stmt_tdeps.begin(), stmt_tdeps.end(),
                        std::back_inserter((*sit)->deps));
    stmt_tdeps.insert(stmt_deps.begin(), stmt_deps.end());
    tdeps.emplace(sit, std::move(stmt_tdeps));
  }
}

}  // namespace

void ScheduleBlock(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options) {
  Scheduler::Schedule(alias_map, block, options);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
