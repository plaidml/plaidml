// Copyright 2018, Intel Corporation

#include "tile/codegen/partition.h"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/tidy.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
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
  std::string base_ref;
  size_t dim_pos;
  UseMap uses;
  TensorShape banked_shape;
  size_t num_banks;
};

Index* BiggestIndexForDim(Block* block, const Refinement& ref, int dim_pos) {
  Index* biggest_idx = nullptr;
  size_t biggest_range = 0;
  for (const auto& kvp : ref.access[dim_pos].getMap()) {
    auto idx = block->idx_by_name(kvp.first);
    if (idx && idx->range > biggest_range) {
      biggest_range = idx->range;
      biggest_idx = idx;
    }
  }
  return biggest_idx;
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
//       std::string orig_name(ref.into());
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
//         LOG(INFO) << "store: ref " << ref->into() << " is related to " << idx_name;
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

// void FixupJumps(Block* block, Refinement* ref, size_t num_banks, const std::string& bank_name, bool is_primary) {
//   for (auto it = block->stmts.begin(); it != block->stmts.end(); it++) {
//     switch ((*it)->kind()) {
//       case StmtKind::Load: {
//         auto load = Load::Downcast(*it);
//         if (load->from == ref->into()) {
//           IVLOG(2, "Load needs fixup: " << *load);
//         }
//       } break;
//       case StmtKind::Store: {
//         auto store = Store::Downcast(*it);
//         if (store->into == ref->into()) {
//           IVLOG(2, "Store needs fixup: " << *store);
//         }
//       } break;
//       case StmtKind::Block: {
//         auto inner = Block::Downcast(*it);
//         auto inner_ref = inner->ref_by_from(ref->into(), false);
//         if (inner_ref != inner->refs.end()) {
//           // add passthru for bank idx
//           auto new_bank_name = inner->unique_idx_name(bank_name);
//           Index bank_idx{new_bank_name, 1, Affine{bank_name}};
//           if (is_primary) {
//             bank_idx.set_tag("bank");
//           }
//           inner->idxs.emplace_back(bank_idx);
//           // extend dim of refinement
//           for (size_t i = 0; i < ref->interior_shape.dims.size() - 1; i++) {
//             inner_ref->interior_shape.dims[i].stride = ref->interior_shape.dims[i].stride;
//           }
//           inner_ref->location.unit = Affine{new_bank_name};
//           inner_ref->bank_dim = ref->bank_dim;
//           FixupJumps(inner.get(), &*inner_ref, num_banks, new_bank_name, is_primary);
//         }
//       } break;
//       default:
//         break;
//     }
//   }
// }

struct BankedRef {
  BankDimension bank_dim;
  Affine cache_unit;
};

void PartitionBuffer(const AliasMap& alias_map,                          //
                     Block* block,                                       //
                     const std::map<std::string, BankInfo>& bank_infos,  //
                     const Tags& set_tags,                               //
                     const std::string& idx_tag) {
  IVLOG(2, "PartitionBuffer> " << block->name);
  std::map<std::string, size_t> tile_by_name;
  std::map<std::string, BankedRef> banked_refs;
  std::set<std::string> primary_idxs;
  // Initialize tile_by_name from the block's index ranges
  for (const auto& idx : block->idxs) {
    tile_by_name[idx.name] = idx.range;
  }
  // Look for refinements that need to be banked
  for (auto& ref : block->refs) {
    const auto& alias_info = alias_map.at(ref.into());
    auto it_bank_info = bank_infos.find(alias_info.base_name);
    if (it_bank_info == bank_infos.end()) {
      continue;
    }
    Index* idx = nullptr;
    const auto& bank_info = it_bank_info->second;
    auto it_usage = bank_info.uses.find(block);
    if (it_usage == bank_info.uses.end()) {
      idx = BiggestIndexForDim(block, ref, bank_info.dim_pos);
      if (!idx) {
        throw_with_trace(
            std::runtime_error(str(boost::format("Could not find valid index to bank on ref %1% in block %2%") %  //
                                   ref.into() % block->name)));
      }
      IVLOG(3, "  secondary> ref: " << ref.into() << ", idx: " << *idx);
    } else {
      idx = block->idx_by_name(it_usage->second.idx_name);
      primary_idxs.insert(idx->name);
      IVLOG(3, "  primary>   ref: " << ref.into() << ", idx: " << *idx);
    }
    // This tag is to prevent idxs from being pruned.
    // A later unroll pass will need these pinned idxs for expansion so that location affine expressions
    // can get fully resolved.
    idx->set_tag("$part");
    // Update the tile size for this index
    tile_by_name[idx->name] = math::RoundUp(idx->range, bank_info.num_banks);
    // Record information used for updating the outer block post-tiling.
    banked_refs[ref.into()] = BankedRef{BankDimension{bank_info.dim_pos}, Affine{idx->name}};
  }
  // Convert the tile based on index name to a tile based on index position
  TileShape tile(block->idxs.size());
  for (size_t i = 0; i < block->idxs.size(); i++) {
    tile[i] = tile_by_name.at(block->idxs[i].name);
  }
  IVLOG(2, "  tile: " << tile_by_name << " " << tile);
  ApplyTile(block, tile, false, true);
  // Update refinements, do this after tiling so that only the outer block gets these updates.
  for (const auto& item : banked_refs) {
    // Use ref_by_into because ApplyTile might cause Refinement pointers to become invalid
    block->ref_by_into(item.first)->mut().cache_unit = item.second.cache_unit;
  }
  block->set_tags(set_tags);
  for (const auto& idx_name : primary_idxs) {
    auto idx = block->idx_by_name(idx_name);
    if (!idx) {
      throw_with_trace(std::runtime_error(str(boost::format("Could not find primary index %1% on block %2%") %  //
                                              idx_name % block->name)));
    }
    idx->set_tag(idx_tag);
  }
  PruneIndexes(block, {"$part"});
}

void CollectBankInfo(std::map<std::string, BankInfo>* bank_infos,  //
                     const AliasMap& alias_map,                    //
                     Block* block,                                 //
                     const proto::PartitionPass& options) {
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
    const auto& alias_info = alias_map.at(ref->into());
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
    if (big_ref->FlatAccess().get(idx.name) == 0 || out_ref->FlatAccess().get(idx.name) == 0 || idx.range == 1) {
      continue;
    }
    size_t tile_size = math::RoundUp(idx.range, options.num_parts());
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

  auto& bank_info = (*bank_infos)[big_alias.base_name];
  if (bank_info.dim_pos && bank_info.dim_pos != dim_pos) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Multiple uses differ about dim_pos on %1%") % big_alias.base_name)));
  }
  auto base_ref = big_alias.base_ref;
  bank_info.base_ref = base_ref->into();
  bank_info.dim_pos = *dim_pos;
  bank_info.uses[block].idx_name = idx_name;
  bank_info.banked_shape = base_ref->interior_shape;
  auto part_size = math::RoundUp(bank_info.banked_shape.dims[*dim_pos].size, options.num_parts());
  bank_info.banked_shape.resize_dim(bank_info.dim_pos, part_size);
  bank_info.num_banks = options.num_parts();
  base_ref->bank_dim = BankDimension{bank_info.dim_pos};
}

}  // namespace

void PartitionMemoryPass(Block* root, const proto::PartitionPass& options) {
  std::map<std::string, BankInfo> bank_infos;
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&](const AliasMap& alias_map, Block* block) {  //
    CollectBankInfo(&bank_infos, alias_map, block, options);
  });

  auto set_tags = FromProto(options.set_tags());
  RunOnBlocks(root, reqs, [&](const AliasMap& alias_map, Block* block) {  //
    PartitionBuffer(alias_map, block, bank_infos, set_tags, options.idx_tag());
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
