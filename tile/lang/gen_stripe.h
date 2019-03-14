#pragma once

#include <memory>
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
std::shared_ptr<stripe::Block> GenerateStripe(const Program& prog,      //
                                              const ShapeMap& inputs,   //
                                              const ShapeMap& outputs,  //
                                              ShapeMap* all);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
