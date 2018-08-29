#pragma once

#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace lang {

std::ostream& operator<<(std::ostream& os, const stripe::proto::Declaration& decl);
std::ostream& operator<<(std::ostream& os, const stripe::proto::RefineIn& ref);
std::ostream& operator<<(std::ostream& os, const stripe::proto::RefineOut& ref);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Load& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Store& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Primitive& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Constant& op);
std::ostream& operator<<(std::ostream& os, const stripe::proto::BufferAccess& access);
std::ostream& operator<<(std::ostream& os, const stripe::proto::Constraint& constraint);

void Print(std::ostream& os, const stripe::proto::Statement& stmt, size_t depth);
void Print(std::ostream& os, const stripe::proto::Block& block, size_t depth);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
