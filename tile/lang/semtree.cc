// Copyright 2017, Vertex.AI.

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

void Type::log(el::base::type::ostream_t& os) const { os << to_string(*this); }

std::string to_string(const Type& ty) {
  std::ostringstream os;
  if (ty.region == Type::LOCAL) {
    os << "local ";
  } else if (ty.region == Type::GLOBAL) {
    os << "global ";
  }
  if (ty.base == Type::POINTER_CONST) {
    os << "const ";
  }
  if (ty.base == Type::TVOID) {
    os << "void ";
  }
  if (ty.base == Type::INDEX) {
    os << "index ";
  }
  if (ty.base != Type::TVOID && ty.base != Type::INDEX) {
    os << to_string(ty.dtype);
  }
  if (ty.vec_width > 1) {
    os << 'x' << std::to_string(ty.vec_width);
  }
  if (ty.base == Type::POINTER_MUT || ty.base == Type::POINTER_CONST) {
    os << '*';
  }
  if (ty.array) {
    os << '[' << std::to_string(ty.array) << ']';
  }
  return os.str();
}

Block::Block() : statements{std::make_shared<std::vector<StmtPtr>>()} {}

Block::Block(std::vector<StmtPtr> s) : Block() {
  statements->insert(statements->end(), std::make_move_iterator(s.begin()), std::make_move_iterator(s.end()));
}

Block::Block(StmtPtr s) : Block() { append(std::move(s)); }

void Block::merge(BlockPtr other) {
  statements->insert(statements->end(), other->statements->begin(), other->statements->end());
}

void Block::append(StmtPtr p) { statements->emplace_back(std::move(p)); }

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
