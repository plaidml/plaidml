// Copyright 2019, Intel Corporation

#pragma once

#include <memory>

namespace pmlc {
namespace dialect {
namespace tile {

struct TileProgram;
struct StripeProgram;

std::shared_ptr<StripeProgram> LowerIntoStripe(TileProgram* program);

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
