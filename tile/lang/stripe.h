#pragma once

#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace lang {

std::ostream& operator<<(std::ostream& os, const stripe::proto::Declaration& decl);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Load& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Store& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Special& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Intrinsic& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Constant& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Constraint& constraint);

void Print(std::ostream& os, const stripe::proto::BufferAccess& access, const stripe::proto::Block& block);
void Print(std::ostream& os, const stripe::proto::Constraint& constraint, const stripe::proto::Block& block);
void Print(std::ostream& os, const stripe::proto::Statement& stmt, size_t depth);
void Print(std::ostream& os, const stripe::proto::Block& block, size_t depth = 0);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
