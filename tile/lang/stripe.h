#pragma once

#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace stripe {
namespace proto {

std::ostream& operator<<(std::ostream& os, const Declaration& decl);
std::ostream& operator<<(std::ostream& os, const Load& op);
std::ostream& operator<<(std::ostream& os, const Store& op);
std::ostream& operator<<(std::ostream& os, const Special& op);
std::ostream& operator<<(std::ostream& os, const Intrinsic& op);
std::ostream& operator<<(std::ostream& os, const Constant& op);
std::ostream& operator<<(std::ostream& os, const Block& block);

void PrintAccess(std::ostream& os, const BufferAccess& access, const Block& block);
void PrintConstraint(std::ostream& os, const Constraint& constraint, const Block& block);
void PrintStatement(std::ostream& os, const Statement& stmt, size_t depth);
void PrintBlock(std::ostream& os, const Block& block, size_t depth = 0);

}  // namespace proto
}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
