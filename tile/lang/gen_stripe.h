#pragma once

#include <string>

#include "tile/lang/compose.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace lang {

std::shared_ptr<stripe::Block> GenerateStripe(const RunInfo& runinfo, bool i8_mode = false);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
