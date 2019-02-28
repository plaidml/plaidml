#pragma once

#include <string>

#include "tile/lang/compose.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace lang {

struct Stripe {
  std::shared_ptr<stripe::Block> program;
  uint64_t total_macs = 0;
};

Stripe GenerateStripe(const RunInfo& runinfo, bool i8_mode = false);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
