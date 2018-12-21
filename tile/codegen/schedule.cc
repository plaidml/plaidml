// Copyright 2018, Intel Corp.

#include "tile/codegen/schedule.h"

#include <forward_list>

#include <boost/optional.hpp>

#include "base/util/error.h"
#include "base/util/throw.h"
#include "tile/codegen/localize.h"

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

inline constexpr std::size_t RoundUp(std::size_t count, std::size_t alignment) {
  return ((count + alignment - 1) / alignment) * alignment;
}

struct CacheEntry;

// RefInfo contains information around the usage of one particular
// backing ref during the scan.
struct RefInfo {
  explicit RefInfo(stripe::Refinement* ref_) : ref(*ref_) {
    TensorShape raw_ts = ref.shape;
    auto sizes = raw_ts.sizes();
    // cache_shape = SimpleShape(ref.shape.type, sizes);
    // TODO: consider a better way to retain index order
    cache_shape = raw_ts;
    size = cache_shape.byte_size();

    for (size_t i = 0; i < sizes.size(); i++) {
      std::string iname = std::string("i") + std::to_string(i);
      swap_idxs.emplace_back(stripe::Index{iname, sizes[i]});
      swap_access.emplace_back(stripe::Affine(iname));
    }

    ref_swap_shape = ref.shape;
    cache_swap_shape = cache_shape;
    for (size_t i = 0; i < sizes.size(); i++) {
      ref_swap_shape.dims[i].size = 1;
      cache_swap_shape.dims[i].size = 1;
    }
  }

  // The actual backing ref -- e.g. DRAM.  We keep a copy because when
  // we add a refinement to our block's refinement vector, we
  // invalidate all iterators / pointers.
  stripe::Refinement ref;

  // The shape of the ref's data when it's in the local cache.
  // Note that this may differ from the ref's shape.
  TensorShape cache_shape;

  // The shapes to use for swap block refinements.
  TensorShape ref_swap_shape;
  TensorShape cache_swap_shape;

  // The affine to use for swapping.
  std::vector<stripe::Affine> swap_access;

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

  // The AliasInfo for this refinement.
  const AliasInfo* alias_info = nullptr;

  // The earliest (runtime-past) sub-statement of the main block that
  // writes to this refinement.
  stripe::Statement* earliest_writer = nullptr;
};

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

// CacheEntry represents one particular local instantiation of a
// value.  (i.e. swapping out a value and swapping it back in results
// in a new CacheEntry).
struct CacheEntry {
  CacheEntry(RefInfo* source_, MemRange range_)
      : source{source_}, name{source->ref.into + "_" + std::to_string(source->next_cache_entry++)}, range{range_} {
    uncovered_ranges.push_back(range);
  }

  // The CacheEntry's backing refinement.
  RefInfo* source;

  // The CacheEntry's refinement's name (its "into" when it becomes a
  // Refinement).
  std::string name;

  // The CacheEntry's memory range when it's being used.
  MemRange range;

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
  std::unordered_map<stripe::Statement*, boost::optional<AliasInfo>> writers;
  std::unordered_map<stripe::Statement*, boost::optional<AliasInfo>> readers;

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

// Represents a single proposed placement of a statement input or output.
struct Placement {
  Placement() {}
  Placement(RefInfo* source_, stripe::RefDir dir_) : source{source_}, dir{dir_} {}
  Placement(RefInfo* source_, stripe::RefDir dir_, MemRange range_, CacheEntry* entry_)
      : source{source_}, dir{dir_}, range{range_}, entry{entry_} {}

  // The backing RefInfo for the CacheEntry.
  RefInfo* source = nullptr;

  // What the Statement is doing with this placement.
  stripe::RefDir dir = stripe::RefDir::None;

  // Where the entry should go.
  MemRange range;

  // The entry for this Placement.  N.B. This may be nullptr, in which
  // case it will be filled in when the plan is accepted.
  CacheEntry* entry = nullptr;
};

std::ostream& operator<<(std::ostream& o, const Placement& p) {
  return o << p.range << " affine=" << p.source->ref.location.unit;
}

// Represents a placement plan for a particular Statement.
using PlacementPlan = std::unordered_map<std::string, Placement>;

// Rewrites a Statement to apply the supplied Plan.
class RefRewriter final : private stripe::MutableStmtVisitor {
 public:
  static void Rewrite(stripe::Statement* stmt, const PlacementPlan& plan, const stripe::Location& loc) {
    RefRewriter rewriter{plan, loc};
    stmt->Accept(&rewriter);
  }

 private:
  RefRewriter(const PlacementPlan& plan, const stripe::Location& loc) : plan_{&plan}, loc_{&loc} {}

  void Visit(stripe::Load*) final;
  void Visit(stripe::Store*) final;
  void Visit(stripe::Constant*) final {}
  void Visit(stripe::Special*) final;
  void Visit(stripe::Intrinsic*) final {}
  void Visit(stripe::Block*) final;

  const PlacementPlan* plan_;
  const stripe::Location* loc_;
};

void RefRewriter::Visit(stripe::Load* load) {
  auto it = plan_->find(load->from);
  if (it == plan_->end()) {
    return;
  }
  load->from = it->second.entry->name;
}

void RefRewriter::Visit(stripe::Store* store) {
  auto it = plan_->find(store->into);
  if (it == plan_->end()) {
    return;
  }
  store->into = it->second.entry->name;
}

void RefRewriter::Visit(stripe::Special* special) {
  for (auto nit = special->inputs.begin(); nit != special->inputs.end(); ++nit) {
    auto pit = plan_->find(*nit);
    if (pit != plan_->end()) {
      *nit = pit->second.entry->name;
    }
  }
  for (auto nit = special->outputs.begin(); nit != special->outputs.end(); ++nit) {
    auto pit = plan_->find(*nit);
    if (pit != plan_->end()) {
      *nit = pit->second.entry->name;
    }
  }
}

void RefRewriter::Visit(stripe::Block* block) {
  for (auto& ref : block->refs) {
    auto it = plan_->find(ref.from);
    if (it == plan_->end()) {
      continue;
    }
    ref.from = it->second.entry->name;
    ref.location = *loc_;
    ref.location.unit = it->second.entry->source->ref.location.unit;
    for (size_t i = 0; i < ref.shape.dims.size(); i++) {
      ref.shape.dims[i].stride = it->second.entry->source->cache_shape.dims[i].stride;
    }
    FixupRefs(block, ref.into);
  }
}

// The scheduler class itself.
class Scheduler {
 public:
  static void Schedule(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options);

 private:
  // Builds a map for looking up RefInfos for a given block access.
  static std::unordered_map<std::string, RefInfo> BuildRefInfoMap(stripe::Block* block);

  Scheduler(const AliasMap* alias_map, stripe::Block* block, const proto::SchedulePass& options);

  // Runs the scheduler over its block.
  void Run();

  // Pre-initializes useful datastructures for placing:
  // * A prototype plan containing placements for every cache entry
  //   that's already been established by a runtime-future Statement,
  // * A vector of RefInfos that need to be placed for the current
  //   Statement.
  std::tuple<PlacementPlan, std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>>
  GatherPlacementState(stripe::Statement* stmt);

  // Makes a placement plan, trying several strategies.
  PlacementPlan MakePlacementPlan(stripe::Statement* stmt);

  // Attempts to augment a placement plan using the supplied ranges.
  bool TryPlaceInRanges(PlacementPlan* plan, const std::vector<std::pair<RefInfo*, stripe::RefDir>>& todos,
                        std::list<MemRange> ranges);

  // Attempts to make a placement plan that preserves the current
  // Statement's existing inputs and outputs, and does not collide
  // with any previously-scheduled CacheEntry unless that CacheEntry
  // has a writer (i.e. does not require swap-in).
  bool TryMakePlanWithNoSwaps(PlacementPlan* plan,
                              const std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>& todos);

  // Attempts to make a placement plan that preserves the current
  // Statement's existing inputs and outputs, but allows collisions
  // with previously-scheduled CacheEntries (producing swap-ins).
  bool TryMakePlanWithSwaps(PlacementPlan* plan,
                            const std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>& todos);

  // Makes a worst-possible-case placement plan; guaranteed to always
  // work (as long as every Statement can fit into memory), but not
  // guaranteed to be optimal at all.
  PlacementPlan MakeFallbackPlan(stripe::Statement* stmt);

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
  stripe::StatementIt AddSwapIn(stripe::StatementIt si, CacheEntry* ent);

  // Schedules a swap-out operation:
  // * Adds a swap-out block just before the supplied iterator,
  // * Gives the swap-in readers a dependency on the swap-out block,
  // * Sets the saw_final_write flag in the source ref, and
  // * Returns the iterator to the new block, so that it can be added as a
  //   dependency to all previously-scheduled writers of overlapping memory.
  //
  // If the swap-out block should have a dependency on something, it's
  // up to the caller to add it.
  stripe::StatementIt AddSwapOut(stripe::StatementIt si, CacheEntry* ent,
                                 const std::unordered_set<stripe::Statement*>& swap_in_readers);

  // Rebuilds the scheduler's block's transitive dependencies -- the
  // deps computed directly by scheduling are conservative.
  void RebuildTransitiveDeps();

  const AliasMap* alias_map_;
  stripe::Block* block_;
  stripe::Location mem_loc_;
  std::size_t mem_bytes_;
  std::size_t alignment_;
  stripe::Location xfer_loc_;
  bool allow_out_of_range_accesses_;
  std::unordered_map<std::string, RefInfo> ri_map_;
  std::unordered_map<stripe::Refinement*, std::vector<RefInfo*>> base_ref_aliases_;

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
  std::map<stripe::Affine, std::list<CacheEntry*>> active_affine_entries_;
};

void Scheduler::Schedule(const AliasMap& alias_map, stripe::Block* block, const proto::SchedulePass& options) {
  Scheduler{&alias_map, block, options}.Run();
}

std::unordered_map<std::string, RefInfo> Scheduler::BuildRefInfoMap(stripe::Block* block) {
  std::unordered_map<std::string, RefInfo> ri_map;
  for (auto& ref : block->refs) {
    ri_map.emplace(ref.into, RefInfo{&ref});
  }
  for (auto& stmt : block->stmts) {
    for (const auto& written_ref_name : stmt->buffer_writes()) {
      auto& ri = ri_map.at(written_ref_name);
      if (!ri.earliest_writer) {
        ri.earliest_writer = stmt.get();
      }
    }
  }
  return ri_map;
}

Scheduler::Scheduler(const AliasMap* alias_map, stripe::Block* block, const proto::SchedulePass& options)
    : alias_map_{alias_map},
      block_{block},
      mem_loc_(stripe::FromProto(options.mem_loc())),
      mem_bytes_{options.mem_kib() * 1024},
      alignment_{options.alignment() ? options.alignment() : kDefaultAlignment},
      xfer_loc_(stripe::FromProto(options.xfer_loc())),
      allow_out_of_range_accesses_{options.allow_out_of_range_accesses()},
      ri_map_{BuildRefInfoMap(block)} {
  for (auto& name_ref : ri_map_) {
    RefInfo* ri = &name_ref.second;
    ri->alias_info = &alias_map_->at(ri->ref.into);
    std::vector<RefInfo*>* aliases = &base_ref_aliases_[ri->alias_info->base_ref];
    aliases->emplace_back(ri);
    ri->aliases = aliases;
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

    IVLOG(3, "Scheduling " << si->get());

    // Add swap-ins for any existing CacheEntries that are invalidated
    // by scheduling this statement.
    std::unordered_map<RefInfo*, std::unordered_set<stripe::Statement*>> ri_writer_swap_in_readers;
    {
      std::list<RefInfo*> ris_whose_swap_in_readers_should_be_cleared;
      for (const auto& output : (*si)->buffer_writes()) {
        RefInfo* ri = &ri_map_.at(output);
        auto* ri_writer_swap_in_readers_set = &ri_writer_swap_in_readers[ri];
        for (RefInfo* alias_ri : *ri->aliases) {
          if ((alias_ri == ri) || AliasInfo::Compare(*ri->alias_info, *alias_ri->alias_info) != AliasType::None) {
            // All accesses to alias_ri will depend on this write.
            if ((alias_ri != ri) && alias_ri->cache_entry) {
              si_next = AddSwapIn(si_next, alias_ri->cache_entry);
              alias_ri->cache_entry = nullptr;
            }

            // Copy all current swap-in readers -- note that this
            // includes the current RefInfo's swap-in-readers.
            for (stripe::Statement* swap_in_reader : alias_ri->swap_in_readers) {
              ri_writer_swap_in_readers_set->emplace(swap_in_reader);
            }
            // Later, clear the ri's swap-in readers.  Note that we
            // actually do the clearing after processing all of the
            // statement's outputs, so that we correctly handle
            // multiple outputs that happen to address different parts
            // of an underlying base refinement.
            ris_whose_swap_in_readers_should_be_cleared.emplace_back(alias_ri);
          }
        }
      }
      for (auto* ri : ris_whose_swap_in_readers_should_be_cleared) {
        ri->swap_in_readers.clear();
      }
    }

    // Figure out where we're going to put any newly-created CacheEntries.
    PlacementPlan plan = MakePlacementPlan(si->get());

    // Generate the AliasMap for this statement, which we'll use for
    // keeping track of accessors.
    boost::optional<AliasMap> current_alias_map;
    stripe::Block* current_block = dynamic_cast<stripe::Block*>(si->get());
    if (current_block) {
      current_alias_map = AliasMap{*alias_map_, current_block};
    }

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

    std::map<stripe::Affine, std::list<CacheEntry*>> added_affine_entries;

    // TODO: There's a straightforward way of walking the plan's
    // placements and the existing active_affine_entries_ at the same
    // time, saving a lot of comparisons.  We don't bother for now,
    // but only because it's a little complicated to get right and
    // premature optimization is the root of all evil, but if we
    // observe a lot of comparisons being done via RangesOverlap(), we
    // have a way to fix it.

    for (auto& name_placement : plan) {
      IVLOG(3, "Applying placement for " << name_placement.first);
      auto& placement = name_placement.second;
      if (!allow_out_of_range_accesses_ && mem_bytes_ < placement.range.end) {
        throw_with_trace(error::ResourceExhausted{"Program requires more memory than is available"});
      }
      CacheEntry* ent = placement.entry;
      bool is_new_entry = (ent == nullptr);
      IVLOG(3, "  IsNewEntry: " << is_new_entry);

      if (is_new_entry) {
        // This Placement requires a new entry.
        ent = &*cache_entries_.emplace(cache_entries_.end(), CacheEntry{placement.source, placement.range});
        IVLOG(3, "Created cache entry " << ent->name << " at " << ent->range
                                        << " with affine=" << ent->source->ref.location.unit);
        placement.entry = ent;
        placement.source->cache_entry = ent;
      }

      // Get the substatement's alias info for this refinement, if any.
      boost::optional<AliasInfo> alias_info = boost::none;
      if (current_alias_map) {
        for (auto& ref : current_block->refs) {
          if (ref.from == ent->source->ref.into) {
            alias_info = boost::make_optional(current_alias_map->at(ref.into));
            break;
          }
        }
      }

      // Add dependency tracking information for this CacheEntry.
      if (IsWriteDir(placement.dir)) {
        for (auto& reader_aliasinfo : ent->readers) {
          if (reader_aliasinfo.second == boost::none || alias_info == boost::none ||
              AliasInfo::Compare(*alias_info, *reader_aliasinfo.second) != AliasType::None) {
            reader_aliasinfo.first->deps.emplace_back(si);
          }
        }

        ent->writers.emplace(si->get(), alias_info);
        if (si->get() == ent->source->earliest_writer) {
          ent->saw_earliest_writer = true;
        }
      }

      if (IsReadDir(placement.dir)) {
        ent->readers.emplace(si->get(), alias_info);
      }

      ent->first_accessor = si;

      // Determine whether this CacheEntry will need to be swapped
      // out, setting up reuse_dep to be the dependency that
      // overlapping CacheEntry objects will use.
      stripe::StatementIt reuse_dep = si;

      if (IsWriteDir(placement.dir) && ((IsWriteDir(placement.source->ref.dir) && !placement.source->saw_final_write) ||
                                        !ri_writer_swap_in_readers[placement.source].empty())) {
        IVLOG(3, "  Adding swap-out for " << ent->name << " at " << ent->range);
        IVLOG(3, "    IsWriteDir(): " << IsWriteDir(placement.source->ref.dir));
        IVLOG(3, "    SawFinalWrite(): " << placement.source->saw_final_write);
        IVLOG(3, "    Swap-in-readers.empty(): " << placement.source->swap_in_readers.empty());
        auto next_si = si;
        ++next_si;
        reuse_dep = AddSwapOut(next_si, ent, ri_writer_swap_in_readers[placement.source]);
        (*reuse_dep)->deps.emplace_back(si);
      }

      // Add dependency tracking information for all
      // previously-created CacheEntries whose ranges overlap the
      // current CacheEntry.
      //
      // N.B. After the SubtractRange() call, we may remove future_ent
      // from its active_affine_entries_ list.  To ensure that our iteration
      // is safe, we explicitly manage it, and make sure to advance
      // the iterator prior to the post-SubtractRange() removal.
      auto& active_entlist = active_affine_entries_[ent->source->ref.location.unit];
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
            auto swap_in_it = AddSwapIn(next_it, future_ent);
            (*swap_in_it)->deps.emplace_back(reuse_dep);
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

      if (is_new_entry) {
        IVLOG(3, "Adding " << ent->name << " at " << ent->range << " to added_affine_entries");
        auto& active_entlist = added_affine_entries[ent->source->ref.location.unit];
        ent->active_iterator = active_entlist.emplace(active_entlist.end(), ent);
        IVLOG(3, "  Active iterator was " << &*ent->active_iterator << "; list at " << &active_entlist
                                          << ", size=" << active_entlist.size());
      }
    }

    IVLOG(3, "Splicing into active_affine_entries_");
    for (auto& added_affine_entlist : added_affine_entries) {
      auto& active_entlist = active_affine_entries_[added_affine_entlist.first];
      active_entlist.splice(active_entlist.begin(), added_affine_entlist.second);
      active_entlist.sort([](CacheEntry* lhs, CacheEntry* rhs) { return lhs->range.begin < rhs->range.begin; });
    }

    if (VLOG_IS_ON(3)) {
      IVLOG(3, "active_affine_entries_ now contains:");
      for (auto& affine_entlist : active_affine_entries_) {
        IVLOG(3, "  Affine: " << affine_entlist.first);
        for (auto* ent : affine_entlist.second) {
          IVLOG(3, "    " << ent->name << " at " << ent->range);
        }
      }
    }

    RefRewriter::Rewrite(si->get(), plan, mem_loc_);
  }

  // Add swap-in writers for every CacheEntry without a writer.
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
  for (auto& affine_entlist : active_affine_entries_) {
    for (auto* ent : affine_entlist.second) {
      if (!ent->source->earliest_writer) {
        IVLOG(3, "  Adding final swap-in for " << ent->name);
        AddSwapIn(ent->first_accessor, ent);
      }
    }
  }

  // Add a Refinement for each CacheEntry.
  block_->refs.reserve(block_->refs.size() + ri_map_.size() + cache_entries_.size());
  for (auto& ent : cache_entries_) {
    auto ref = block_->ref_by_into(ent.name, false);
    if (ref == block_->refs.end()) {
      ref = block_->refs.emplace(block_->refs.end(), ent.source->ref);
    }
    ref->dir = stripe::RefDir::None;
    ref->from.clear();
    ref->into = ent.name;
    ref->shape = ent.source->cache_shape;
    ref->location = mem_loc_;
    ref->location.unit = ent.source->ref.location.unit;
    ref->is_const = false;
    ref->offset = ent.range.begin;
  }

  // Move used Refinements back into the block.
  for (auto& name_ri : ri_map_) {
    if (name_ri.second.used) {
      auto ref = block_->ref_by_into(name_ri.second.ref.into, false);
      if (ref == block_->refs.end()) {
        block_->refs.emplace_back(std::move(name_ri.second.ref));
      } else {
        *ref = name_ri.second.ref;
      }
    }
  }

  RebuildTransitiveDeps();

  // Refinement order doesn't matter -- so sort the refinements by
  // their "into" field (which all refinements have), to simplify
  // testing.
  std::sort(block_->refs.begin(), block_->refs.end(), [](const stripe::Refinement& lhs, const stripe::Refinement& rhs) {
    return std::less<std::string>{}(lhs.into, rhs.into);
  });
}

std::tuple<PlacementPlan, std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>>
Scheduler::GatherPlacementState(stripe::Statement* stmt) {
  PlacementPlan plan;
  std::unordered_map<RefInfo*, stripe::RefDir> todo_map;

  auto add = [&](const std::string& name, stripe::RefDir dir) {
    // See whether we've already created a Placement for this IO.
    auto it = plan.find(name);
    if (it != plan.end()) {
      // We've already made a Placement; add in our direction, and we're done.
      it->second.dir = UnionDir(it->second.dir, dir);
      return;
    }

    // See whether we already have an active CacheEntry for this IO.
    auto* ri = &ri_map_.at(name);
    if (ri->cache_entry) {
      // We do -- create a Placement describing it.
      plan.emplace(name, Placement{ri, dir, ri->cache_entry->range, ri->cache_entry});
      return;
    }

    // Otherwise, we're going to need to allocate a Placement.  We'll
    // do it after processing all inputs, so that we can do placement
    // in size order with correct directions.
    auto it_inserted = todo_map.emplace(ri, dir);
    if (!it_inserted.second) {
      it_inserted.first->second = UnionDir(it_inserted.first->second, dir);
    }
  };

  for (const auto& output : stmt->buffer_writes()) {
    add(output, stripe::RefDir::Out);
  }

  for (const auto& input : stmt->buffer_reads()) {
    add(input, stripe::RefDir::In);
  }

  // Organize the placements to be made per-affine, largest-first, using the
  // underlying refinement 'into' name as the tiebreaker.
  std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>> todos;
  for (auto& refinfo_refdir : todo_map) {
    todos[refinfo_refdir.first->ref.location.unit].emplace_back(refinfo_refdir);
  }
  for (auto& affine_refvec : todos) {
    std::sort(affine_refvec.second.begin(), affine_refvec.second.end(),
              [](std::pair<RefInfo*, stripe::RefDir> lhs, std::pair<RefInfo*, stripe::RefDir> rhs) {
                return std::tie(rhs.first->size, rhs.first->ref.into) < std::tie(lhs.first->size, lhs.first->ref.into);
              });
  }

  return std::make_tuple(std::move(plan), std::move(todos));
}

PlacementPlan Scheduler::MakePlacementPlan(stripe::Statement* stmt) {
  // Initialize useful planning inputs.
  PlacementPlan existing_entry_plan;
  std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>> todos;

  std::tie(existing_entry_plan, todos) = GatherPlacementState(stmt);

  if (VLOG_IS_ON(3)) {
    IVLOG(3, "  Existing entries in plan:");
    for (auto& name_placement : existing_entry_plan) {
      IVLOG(3, "    " << name_placement.first << " -> " << name_placement.second);
    }
    IVLOG(3, "  ToDos:");
    for (auto& affine_refvec : todos) {
      IVLOG(3, "    Affine=" << affine_refvec.first);
      for (auto& ri_dir : affine_refvec.second) {
        IVLOG(3, "      Ref=" << ri_dir.first->ref.into << " size=" << ri_dir.first->size);
      }
    }
  }

  PlacementPlan plan{existing_entry_plan};
  if (TryMakePlanWithNoSwaps(&plan, todos)) {
    IVLOG(3, "  Made plan with no swaps");
    return plan;
  }

  plan = existing_entry_plan;
  if (TryMakePlanWithSwaps(&plan, todos)) {
    IVLOG(3, "  Made plan with swaps");
    return plan;
  }

  // Plan from scratch.  This does not pay attention to existing
  // entries, so more swaps will be required, but it is guaranteed to
  // work.
  IVLOG(3, "  Using fallback plan");
  return MakeFallbackPlan(stmt);
}

bool Scheduler::TryPlaceInRanges(PlacementPlan* plan, const std::vector<std::pair<RefInfo*, stripe::RefDir>>& todos,
                                 std::list<MemRange> ranges) {
  // For each todo in largest->smallest size, determine a placement.
  // For each one, we want to pick the smallest free range that is
  // still big enough to hold the todo.
  IVLOG(3, "      Looking for placements");
  for (auto todo : todos) {
    IVLOG(3, "        Finding placement for " << todo.first->ref.into << ", size=" << todo.first->size);
    std::list<MemRange>::iterator best_so_far = ranges.end();
    std::size_t best_waste_so_far = mem_bytes_;
    for (auto rit = ranges.begin(); rit != ranges.end(); ++rit) {
      if (rit->size() < todo.first->size) {
        continue;
      }
      std::size_t waste = rit->size() - todo.first->size;
      if (best_waste_so_far <= waste) {
        continue;
      }
      IVLOG(3, "          Range " << *rit << " is the best so far");
      best_so_far = rit;
      best_waste_so_far = waste;
    }
    if (best_so_far == ranges.end()) {
      return false;
    }
    auto assigned_range = MemRange{best_so_far->begin, best_so_far->begin + todo.first->size};
    SubtractRange(assigned_range, &ranges, best_so_far);
    plan->emplace(todo.first->ref.into, Placement{todo.first, todo.second, assigned_range, nullptr});
  }

  return true;
}

bool Scheduler::TryMakePlanWithNoSwaps(
    PlacementPlan* plan, const std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>& todos) {
  IVLOG(3, "    Attempting plan with no swaps");

  for (auto& affine_refvec : todos) {
    // Build a list of the available ranges.  For our purposes, a range
    // is available if it already has an initial writer (=> it is not
    // going to require a swap-in), and if its RefInfo is not already in
    // the plan (because RefInfos that are in the plan are required by
    // the current statement).
    IVLOG(3, "      Planning memory affine=" << affine_refvec.first);
    std::list<MemRange> ranges{MemRange{0, mem_bytes_}};
    for (auto* ent : active_affine_entries_[affine_refvec.first]) {
      IVLOG(3, "      Saw range " << ent->range << " used by " << ent->name << " saw_earliest_writer="
                                  << ent->saw_earliest_writer << " plan.count=" << plan->count(ent->source->ref.into));
      if (!(ent->saw_earliest_writer && !plan->count(ent->source->ref.into))) {
        IVLOG(3, "      Subtracting range " << ent->range << " used by " << ent->name);
        SubtractRange(ent->range, &ranges);
      }
    }

    if (!TryPlaceInRanges(plan, affine_refvec.second, std::move(ranges))) {
      return false;
    }
  }

  return true;
}

bool Scheduler::TryMakePlanWithSwaps(
    PlacementPlan* plan, const std::map<stripe::Affine, std::vector<std::pair<RefInfo*, stripe::RefDir>>>& todos) {
  IVLOG(3, "    Attempting plan with swaps");

  for (auto& affine_refvec : todos) {
    // Build a list of the available ranges.  For our purposes, a range
    // is available as long as its RefInfo is not already in the plan
    // (because RefInfos that are in the plan are required by the
    // current statement).
    std::list<MemRange> ranges{MemRange{0, mem_bytes_}};
    for (auto* ent : active_affine_entries_[affine_refvec.first]) {
      IVLOG(3, "      Saw range " << ent->range << " used by " << ent->name << " saw_earliest_writer="
                                  << ent->saw_earliest_writer << " plan.count=" << plan->count(ent->source->ref.into));
      if (plan->count(ent->source->ref.into)) {
        IVLOG(3, "      Subtracting range " << ent->range << " used by " << ent->name);
        SubtractRange(ent->range, &ranges);
      }
    }

    if (!TryPlaceInRanges(plan, affine_refvec.second, std::move(ranges))) {
      return false;
    }
  }

  return true;
}

PlacementPlan Scheduler::MakeFallbackPlan(stripe::Statement* stmt) {
  PlacementPlan plan;
  std::size_t offset = 0;

  auto add = [&](const std::string& name, stripe::RefDir dir) {
    auto* ri = &ri_map_.at(name);
    auto it_inserted = plan.emplace(name, Placement{ri, dir});
    if (it_inserted.second) {
      // A new Placement.
      it_inserted.first->second.range.begin = offset;
      it_inserted.first->second.range.end = offset + ri->size;
      offset += RoundUp(ri->size, alignment_);
    } else {
      // An existing Placement.
      it_inserted.first->second.dir = UnionDir(it_inserted.first->second.dir, dir);
    }
  };

  for (const auto& output : stmt->buffer_writes()) {
    add(output, stripe::RefDir::Out);
  }

  for (const auto& input : stmt->buffer_reads()) {
    add(input, stripe::RefDir::In);
  }

  return plan;
}

stripe::StatementIt Scheduler::AddSwapIn(stripe::StatementIt si, CacheEntry* ent) {
  stripe::Block swap_block;
  ent->source->used = true;
  swap_block.name = "swap_in_" + ent->name;
  swap_block.location = xfer_loc_;
  swap_block.idxs = ent->source->swap_idxs;
  swap_block.refs.push_back(stripe::Refinement{
      stripe::RefDir::In,           // dir
      ent->source->ref.into,        // from
      "src",                        // into
      ent->source->swap_access,     // access
      ent->source->ref_swap_shape,  // shape
      "",                           // agg_op
      ent->source->ref.location,    // location
      true,                         // is_const
      0,                            // offset
      ent->source->ref.bank_dim,    // bank_dim
  });

  auto banked_mem_loc = mem_loc_;
  banked_mem_loc.unit = ent->source->ref.location.unit;
  swap_block.refs.push_back(stripe::Refinement{
      stripe::RefDir::Out,            // dir
      ent->name,                      // from
      "dst",                          // into
      ent->source->swap_access,       // access
      ent->source->cache_swap_shape,  // shape
      "",                             // agg_op
      banked_mem_loc,                 // location
      false,                          // is_const
      0,                              // offset
      ent->source->ref.bank_dim,      // bank_dim
  });

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  stripe::StatementIt swap_in_it = block_->stmts.emplace(si, std::make_shared<stripe::Block>(std::move(swap_block)));
  stripe::Statement* swap_in = swap_in_it->get();
  ent->writers.emplace(swap_in, boost::none);
  ent->source->swap_in_readers.emplace(swap_in);
  for (auto& reader_aliasinfo : ent->readers) {
    reader_aliasinfo.first->deps.emplace_back(swap_in_it);
  }
  return swap_in_it;
}

stripe::StatementIt Scheduler::AddSwapOut(stripe::StatementIt si, CacheEntry* ent,
                                          const std::unordered_set<stripe::Statement*>& swap_in_readers) {
  stripe::Block swap_block;
  ent->source->used = true;
  swap_block.name = "swap_out_" + ent->name;
  swap_block.location = xfer_loc_;
  swap_block.idxs = ent->source->swap_idxs;
  swap_block.refs.push_back(stripe::Refinement{
      stripe::RefDir::In,             // dir
      ent->name,                      // from
      "src",                          // into
      ent->source->swap_access,       // access
      ent->source->cache_swap_shape,  // shape
      "",                             // agg_op
      mem_loc_,                       // location
      true,                           // is_const
      0,                              // offset
      ent->source->ref.bank_dim,      // bank_dim
  });

  swap_block.refs.push_back(stripe::Refinement{
      stripe::RefDir::Out,          // dir
      ent->source->ref.into,        // from
      "dst",                        // into
      ent->source->swap_access,     // access
      ent->source->ref_swap_shape,  // shape
      "",                           // agg_op
      ent->source->ref.location,    // location
      false,                        // is_const
      0,                            // offset
      ent->source->ref.bank_dim,    // bank_dim
  });

  swap_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  swap_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));

  stripe::StatementIt swap_out_it = block_->stmts.emplace(si, std::make_shared<stripe::Block>(std::move(swap_block)));
  for (stripe::Statement* reader : swap_in_readers) {
    reader->deps.emplace_back(swap_out_it);
  }
  ent->source->saw_final_write = true;
  return swap_out_it;
}

void Scheduler::RebuildTransitiveDeps() {
  std::unordered_map<stripe::StatementIt, std::set<stripe::StatementIt>> tdeps;
  tdeps.reserve(block_->stmts.size());

  for (auto sit = block_->stmts.begin(); sit != block_->stmts.end(); ++sit) {
    std::set<stripe::StatementIt> stmt_deps;
    std::set<stripe::StatementIt> stmt_tdeps;
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
