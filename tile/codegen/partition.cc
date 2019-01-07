// Copyright 2018, Intel Corporation

#include "tile/codegen/partition.h"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/math.h"
#include "tile/codegen/tags.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct UseInfo {
  std::string idx_name;
};

using UseMap = std::map<Block*, UseInfo>;

struct BankInfo {
  Block* base_block;
  std::string base_ref;
  size_t dim_pos;
  UseMap uses;
  TensorShape banked_shape;
  size_t num_banks;
};

const Index* BiggestIndexForDim(const Block& block, const Refinement& ref, int dim_pos) {
  const Index* biggest_idx = nullptr;
  size_t biggest_range = 0;
  for (const auto& kvp : ref.access[dim_pos].getMap()) {
    auto idx = block.idx_by_name(kvp.first);
    if (idx && idx->range > biggest_range) {
      biggest_range = idx->range;
      biggest_idx = idx;
    }
  }
  return biggest_idx;
}

void SplitRefinement(Refinement* ref, const BankInfo& bank_info, const std::string& bank_name) {
  ref->bank_dim = BankDimension{bank_info.dim_pos, ref->interior_shape, ref->into};
  for (size_t i = 0; i < ref->interior_shape.dims.size(); i++) {
    ref->interior_shape.dims[i].stride = bank_info.banked_shape.dims[i].stride;
  }
  ref->location.unit = Affine{bank_name};
}

void SplayRefinement(Block* block, const Refinement& ref, const BankInfo& bank_info) {
  for (size_t i = 0; i < bank_info.num_banks; i++) {
    auto name = str(boost::format("%1%%%%2%") % ref.into % i);
    Location loc{ref.location.name, Affine(i)};
    Refinement new_ref{
        ref.dir,                 // dir
        ref.from,                // from
        name,                    // into
        ref.access,              // access
        bank_info.banked_shape,  // shape
        ref.agg_op,              // agg_op
        loc,                     // location
        ref.is_const,            // is_const
        ref.offset,              // offset
        ref.bank_dim,            // bank_dim
    };
    new_ref.set_tag("banked");
    block->refs.emplace_back(new_ref);
  }
}

// bool IsRefRelatedToIdx(const Refinement& ref, const std::string& idx_name) {
//   for (const auto& aff : ref.access) {
//     if (aff.get(idx_name)) {
//       return true;
//     }
//   }
//   return false;
// }

// struct SplitRefInfo {
//   std::string name;
//   TensorShape scalar_shape;
//   TensorShape banked_shape;
//   std::vector<Affine> scalar_access;
//   std::vector<Affine> banked_access;
// };

// void FixupLoad() {
//   std::map<std::string, SplitRefInfo> related_map;
//   for (auto& ref : block->refs) {
//     if (IsRefRelatedToIdx(ref, idx_name)) {
//       std::string orig_name(ref.into);
//       auto raw_name = block->unique_ref_name(orig_name + "_raw");
//       std::vector<size_t> sizes(ref.shape.dims.size(), 1);
//       TensorShape scalar_shape = SimpleShape(ref.shape.type, sizes);
//       std::vector<Affine> scalar_access(ref.shape.dims.size());
//       TensorShape banked_shape = ref.shape;
//       for (auto& dim : banked_shape.dims) {
//         dim.size = 1;
//       }
//       SplitRefInfo info{
//           raw_name,       // name
//           banked_shape,   // banked_shape
//           scalar_shape,   // scalar_shape
//           scalar_access,  // scalar_access
//           ref.access,     // banked_access
//       };
//       related_map.emplace(orig_name, info);
//       ref.into = raw_name;
//     }
//   }

//   // Replace related loads and stores with switch blocks to handle halo conditions
//   for (auto it = block->stmts.begin(); it != block->stmts.end(); it++) {
//     auto load = Load::Downcast(*it);
//     if (load) {
//       auto it_related = related_map.find(load->from);
//       if (it_related != related_map.end()) {
//         const auto& info = it_related->second;
//         IVLOG(2, "load: ref " << load->from << " is related to " << idx_name << " via " << info.name);

//         auto interior = std::make_shared<Block>();
//         interior->name = load->from + "_load_interior";
//         interior->stmts.emplace_back(std::make_shared<Load>("src", "$X"));
//         interior->stmts.emplace_back(std::make_shared<Store>("$X", "dst"));
//         interior->refs.emplace_back(Refinement{
//             RefDir::In,          // dir
//             info.name,           // from
//             "src",               // into
//             info.banked_access,  // access
//             info.banked_shape,   // shape
//         });
//         interior->refs.emplace_back(Refinement{
//             RefDir::Out,         // dir
//             load->from,          // from
//             "dst",               // into
//             info.scalar_access,  // access
//             info.banked_shape,   // shape
//         });
//         // interior->constraints.emplace_back(info.banked_access - Affine(bank_name, part_size) -
//         Affine(offset_name)); block->stmts.emplace(it, interior);

//         auto low_edge = std::make_shared<Block>();
//         low_edge->name = load->from + "_load_low_edge";
//         block->stmts.emplace(it, low_edge);

//         auto high_edge = std::make_shared<Block>();
//         high_edge->name = load->from + "_load_high_edge";
//         block->stmts.emplace(it, high_edge);
//       }
//     }
//     auto store = Store::Downcast(*it);
//     if (store) {
//       auto ref = block->ref_by_into(store->into);
//       if (IsRefRelatedToIdx(*ref, idx_name)) {
//         LOG(INFO) << "store: ref " << ref->into << " is related to " << idx_name;
//       }
//     }
//   }
// Finally add new refs
// for (const auto& info : related_map) {
//   block->refs.emplace_back(Refinement{
//       RefDir::None,              // dir
//       "",                        // from
//       info.first,                // into
//       {},                        // access
//       info.second.scalar_shape,  // shape
//   });
// }

void FixupJumps(Block* block, Refinement* ref, size_t num_banks, const std::string& bank_name, bool is_primary) {
  for (auto it = block->stmts.begin(); it != block->stmts.end(); it++) {
    switch ((*it)->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(*it);
        if (load->from == ref->into) {
          IVLOG(2, "Load needs fixup: " << *load);
        }
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(*it);
        if (store->into == ref->into) {
          IVLOG(2, "Store needs fixup: " << *store);
        }
      } break;
      case StmtKind::Block: {
        auto inner = Block::Downcast(*it);
        auto inner_ref = inner->ref_by_from(ref->into, false);
        if (inner_ref != inner->refs.end()) {
          // add passthru for bank idx
          auto new_bank_name = inner->unique_idx_name(bank_name);
          Index bank_idx{new_bank_name, 1, Affine{bank_name}};
          if (is_primary) {
            bank_idx.set_tag("bank");
          }
          inner->idxs.emplace_back(bank_idx);
          // extend dim of refinement
          for (size_t i = 0; i < ref->interior_shape.dims.size() - 1; i++) {
            inner_ref->interior_shape.dims[i].stride = ref->interior_shape.dims[i].stride;
          }
          inner_ref->location.unit = Affine{new_bank_name};
          inner_ref->bank_dim = ref->bank_dim;
          FixupJumps(inner.get(), &*inner_ref, num_banks, new_bank_name, is_primary);
        }
      } break;
      default:
        break;
    }
  }
}

void SplitIndex(Block* block,                 //
                Refinement* ref,              //
                const std::string& idx_name,  //
                const BankInfo& bank_info,    //
                bool is_primary) {
  auto bank_name = block->unique_idx_name(idx_name + "_bank");
  auto offset_name = block->unique_idx_name(idx_name + "_off");
  auto bank_idx = block->idx_by_name(idx_name);
  auto orig_range = bank_idx->range;
  auto offset_range = IntDivCeil(orig_range, bank_info.num_banks);

  SplitRefinement(ref, bank_info, bank_name);

  if (is_primary) {
    bank_idx->set_tag("bank");
  }
  bank_idx->name = bank_name;
  bank_idx->range = bank_info.num_banks;

  block->idxs.emplace_back(Index{offset_name, offset_range, Affine{}});

  ref->access[bank_info.dim_pos].substitute(idx_name, Affine{offset_name});

  Affine combo = Affine(bank_name, offset_range) + Affine(offset_name);

  // Substitute refs
  for (auto& other_ref : block->refs) {
    if (&other_ref != ref) {
      for (auto& aff : other_ref.access) {
        aff.substitute(idx_name, combo);
      }
    }
  }

  // Substitute child block index affines
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& idx : inner->idxs) {
        idx.affine.substitute(idx_name, combo);
      }
    }
  }

  // Substitute constraints
  for (auto& constraint : block->constraints) {
    constraint.substitute(idx_name, combo);
  }

  // Deal with uneven splits
  if (orig_range % bank_info.num_banks) {
    block->constraints.emplace_back(int64_t(orig_range - 1) - Affine(bank_name, offset_range) - Affine(offset_name));
  }

  FixupJumps(block, ref, bank_info.num_banks, bank_name, is_primary);
}

void PartitionBuffer(Block* block, const std::string& ref_name, const BankInfo& bank_info, const Tags& set_tags) {
  auto ref = &*block->ref_by_into(ref_name);
  ref->bank_dim = BankDimension{bank_info.dim_pos, ref->interior_shape, ref->into};
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    auto inner_ref = inner->ref_by_from(ref->into, false);
    if (inner_ref != inner->refs.end()) {
      // is this a passthru or a legitimate top-level use?
      const auto& bank_access = inner_ref->access[bank_info.dim_pos];
      if (bank_access.isConstant()) {
        IVLOG(2, boost::format("Passthru of %1% on %2%") % inner_ref->into % inner->name);
        PartitionBuffer(inner.get(), inner_ref->into, bank_info, set_tags);
      } else {
        std::string idx_name;
        bool is_primary;
        auto it = bank_info.uses.find(inner.get());
        if (it == bank_info.uses.end()) {
          IVLOG(2, "Use not found on block " << inner->name << " for ref " << inner_ref->into);
          IVLOG(3, "  ref: " << *ref);
          auto idx = BiggestIndexForDim(*inner, *inner_ref, bank_info.dim_pos);
          if (!idx) {
            throw_with_trace(std::runtime_error(
                str(boost::format("Could not find valid index to bank on ref '%1%' in block '%2%'") %  //
                    inner_ref->into % inner->name)));
          }
          is_primary = false;
          idx_name = idx->name;
          IVLOG(3, "  idx: " << *idx);
        } else {
          is_primary = true;
          idx_name = it->second.idx_name;
        }
        IVLOG(2, boost::format("Split dim of %1% on %2% via %3%") % inner_ref->into % inner->name % idx_name);
        SplitIndex(inner.get(), &*inner_ref, idx_name, bank_info, is_primary);
        inner->add_tags(set_tags);
      }
    }
  }
  SplayRefinement(block, *ref, bank_info);
}

void DebankBlocks(Block* block, const Location& loc) {
  std::vector<std::string> to_erase;
  for (auto& ref : block->refs) {
    if (ref.location.name == loc.name && ref.has_tag("banked")) {
      to_erase.push_back(ref.into);
      continue;
    }
    if (ref.location.name == loc.name && ref.bank_dim) {
      const auto& banked_dim = ref.interior_shape.dims[ref.bank_dim->dim_pos];
      ref.access[ref.bank_dim->dim_pos] += ref.location.unit * banked_dim.stride;
      ref.location.unit = 0;  // TODO: make unit optional?
      if (ref.dir != RefDir::None) {
        ref.from = ref.bank_dim->orig_name;
      }
      for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
        ref.interior_shape.dims[i].stride = ref.bank_dim->orig_shape.dims[i].stride;
      }
      ref.bank_dim = boost::none;
    }
  }
  for (const auto& name : to_erase) {
    auto ref = block->ref_by_into(name);
    block->refs.erase(ref);
  }
  for (auto& stmt : block->stmts) {
    auto inner = stripe::Block::Downcast(stmt);
    if (inner) {
      DebankBlocks(inner.get(), loc);
    }
  }
}

}  // namespace

void PartitionPass(Block* root, const proto::PartitionPass& options) {
  std::map<std::string, BankInfo> buf_banks;
  auto reqs = FromProto(options.reqs());
  auto set_tags = FromProto(options.set_tags());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, Block* block) {
    IVLOG(2, "Partition> block: " << block->name);
    if (block->ref_outs().size() != 1) {
      IVLOG(1, boost::format("Partition> skipped '%1%' due to multiple outputs") % block->name);
      return;  // Only work on blocks with a single output
    }
    // Get the (only) output
    const Refinement* out_ref = block->ref_outs()[0];
    // Find the largest input
    size_t biggest = 0;
    AliasInfo big_alias;
    const Refinement* big_ref = nullptr;
    for (const auto& ref : block->ref_ins()) {
      // Get the source buffer size
      const auto& alias_info = map.at(ref->into);
      size_t src_size = alias_info.base_ref->interior_shape.elem_size();
      if (src_size > biggest) {
        biggest = src_size;
        big_ref = ref;
        big_alias = alias_info;
      }
    }
    if (big_ref == nullptr) {
      IVLOG(1, "Partition> skipped due to no inputs");
      return;  // No inputs?  Skip this block
    }
    // Find the evenest index that has a non-zero stride both on the large
    // input and on the output refinement
    std::string idx_name;
    size_t best_tile = 0;
    double best_ratio = 0;
    for (size_t i = 0; i < block->idxs.size(); i++) {
      const auto& idx = block->idxs[i];
      if (big_ref->FlatAccess().get(idx.name) == 0) {
        continue;
      }
      if (out_ref->FlatAccess().get(idx.name) == 0) {
        continue;
      }
      size_t tile_size = IntDivCeil(idx.range, options.num_parts());
      size_t rounded_size = tile_size * options.num_parts();
      double ratio = static_cast<double>(idx.range) / static_cast<double>(rounded_size);
      IVLOG(3, "           "
                   << "idx: " << idx                          //
                   << ", num_parts: " << options.num_parts()  //
                   << ", tile_size: " << tile_size            //
                   << ", rounded_size: " << rounded_size      //
                   << ", ratio: " << ratio);
      if (ratio > best_ratio) {
        best_ratio = idx.range;
        best_tile = tile_size;
        idx_name = idx.name;
      }
    }
    if (best_tile == 0) {
      IVLOG(1, "Partition> skipped due to no valid indexes");
      return;  // No valid indexes?  Skip this block
    }
    // Determine the bank_dim for the split memory
    boost::optional<size_t> dim_pos;
    for (size_t i = 0; i < big_ref->access.size(); i++) {
      if (big_ref->access[i].get(idx_name)) {
        if (dim_pos) {
          IVLOG(1, "Partition> skipped due to complex banking");
          return;
        }
        dim_pos = i;
      }
    }
    IVLOG(2, "           dim_pos: " << dim_pos << ", base_ref: " << *big_alias.base_ref);
    if (!dim_pos) {
      IVLOG(1, "Could not find dimension to bank on for block: " << block->name << ", ref: " << *big_alias.base_ref);
      return;
    }

    auto& bank_info = buf_banks[big_alias.base_name];
    if (bank_info.dim_pos && bank_info.dim_pos != dim_pos) {
      throw_with_trace(
          std::runtime_error(str(boost::format("Multiple uses differ about dim_pos on %1%") % big_alias.base_name)));
    }
    bank_info.base_block = big_alias.base_block;
    bank_info.base_ref = big_alias.base_ref->into;
    bank_info.dim_pos = *dim_pos;
    bank_info.uses[block].idx_name = idx_name;
    bank_info.banked_shape = big_alias.base_ref->interior_shape;
    auto part_size = IntDivCeil(bank_info.banked_shape.dims[*dim_pos].size, options.num_parts());
    bank_info.banked_shape.resize_dim(*dim_pos, part_size);
    bank_info.num_banks = options.num_parts();
  });

  for (const auto& kvp : buf_banks) {
    const auto& bank_info = kvp.second;
    IVLOG(2, "Partition> base_name: " << kvp.first << ", dim_pos: " << bank_info.dim_pos);
    IVLOG(2, "    banked_shape: " << bank_info.banked_shape);
    for (const auto& use : bank_info.uses) {
      IVLOG(2, "           block: " << use.first->name << ", idx_name: " << use.second.idx_name);
    }
    PartitionBuffer(bank_info.base_block, bank_info.base_ref, bank_info, set_tags);
  }
}

void DebankPass(stripe::Block* root, const proto::DebankPass& options) {
  auto loc = stripe::FromProto(options.loc());
  DebankBlocks(root, loc);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
