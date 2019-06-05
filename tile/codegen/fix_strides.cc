// Copyright 2019, Intel Corp.

#include "tile/codegen/fix_strides.h"

#include "base/util/any_factory_map.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/localize.h"
#include "tile/math/util.h"

namespace vertexai {
namespace tile {
namespace codegen {

static void DoTranspose(char* out, const char* in, const TensorShape& os, const TensorShape& is, size_t dim = 0) {
  size_t elem_width = byte_width(os.type);
  if (dim == os.dims.size()) {
    memcpy(out, in, elem_width);
  } else {
    for (size_t i = 0; i < os.dims[dim].size; i++) {
      DoTranspose(out + i * os.dims[dim].stride * elem_width, in + i * is.dims[dim].stride * elem_width, os, is,
                  dim + 1);
    }
  }
}

static void FixStridesBlock(stripe::Block* block, CompilerState* state) {
  context::Context ctx;
  std::set<std::string> used_in_special;
  for (auto& stmt : block->stmts) {
    auto spec = stripe::Special::Downcast(stmt);
    if (!spec || spec->name == "zero") {
      continue;
    }
    for (auto& s : spec->inputs) {
      used_in_special.emplace(s);
    }
    for (auto& s : spec->outputs) {
      used_in_special.emplace(s);
    }
  }
  for (auto& ref : block->refs) {
    // Skip things used in specials
    if (used_in_special.count(ref.into())) {
      IVLOG(2, "Skipping " << ref.into() << " due to use in special");
      continue;
    }
    // Skip non allocations
    if (ref.dir != stripe::RefDir::None) {
      IVLOG(2, "Skipping " << ref.into() << " due to direction");
      continue;
    }
    // Get the shape
    auto shape = ref.interior_shape;
    // Skip non-const user buffers
    if (ref.has_tag("user") && !shape.is_const) {
      IVLOG(2, "Skipping " << ref.into() << " due to user");
      continue;
    }
    // Skip PRNG buffers
    if (shape.type == DataType::PRNG || shape.type == DataType::BOOLEAN) {
      continue;
    }
    // Make an 'index order' vector to sort
    std::vector<size_t> order;
    for (size_t i = 0; i < shape.dims.size(); i++) {
      order.push_back(i);
    }
    // Sort it by number of factors (more factors = first)
    std::sort(order.begin(), order.end(), [&](size_t i, size_t j) -> bool {
      size_t factors_i = math::NumFactors(shape.dims[i].size);
      size_t factors_j = math::NumFactors(shape.dims[j].size);
      // Pick the one with more factors first
      if (factors_i != factors_j) {
        return factors_i > factors_j;
      }
      // Otherwise, largest original index (so order is stable)
      return i > j;
    });
    // Make new striding pattern
    size_t inner = 1;
    auto new_shape = shape;
    for (size_t idx : order) {
      new_shape.dims[idx].stride = inner;
      inner *= new_shape.dims[idx].size;
    }
    // If we have nothing to adjust, continue
    if (shape == new_shape) {
      continue;
    }
    // FOr now skip the tricky ones
    if (ref.has_tag("user") && shape.is_const) {
      auto new_buffer = state->const_bufs->allocator->allocate(new_shape.byte_size());
      auto old_view = state->const_bufs->buffers.at(ref.into())->MapCurrent(ctx).get();
      auto new_view = new_buffer->MapDiscard(ctx);
      const char* old_ptr = old_view->begin();
      char* new_ptr = new_view->begin();
      DoTranspose(new_ptr, old_ptr, new_shape, shape);
      state->const_bufs->buffers[ref.into()] = new_buffer;
    }
    ref.mut().interior_shape = new_shape;
    FixupRefs(block, ref.into());
    IVLOG(2, "Changed shape: " << shape << " -> " << new_shape);
  }
}

void FixStridesPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    FixStridesBlock(block, state);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<FixStridesPass, proto::FixStridesPass>::Register();
  return 0;
}();
}  // namespace

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
