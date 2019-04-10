// Copyright 2018, Intel Corporation

#include "tile/codegen/codec.h"

#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

void AssignCodec(Block* block, const Tags& datatypes, const std::string& codec) {
  IVLOG(2, "  block: " << block->name);
  for (auto& ref : block->refs) {
    auto ref_type = to_string(ref.interior_shape.type);
    if (datatypes.count(ref_type)) {
      IVLOG(2, "    ref: " << ref.into());
      ref.mut().interior_shape.codec = codec;
    }
  }
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AssignCodec(inner.get(), datatypes, codec);
    }
  }
}

}  // namespace

void AssignCodecPass(stripe::Block* root, const proto::AssignCodecPass& options) {
  auto datatypes = FromProto(options.datatypes());
  IVLOG(2, "AssignCodecPass> codec: " << options.codec());
  AssignCodec(root, datatypes, options.codec());
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
