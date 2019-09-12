#pragma once

#include <memory>
#include <string>

#include "tile/lang/runinfo.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace lang {

std::shared_ptr<stripe::Program> GenerateStripe(const RunInfo& runinfo, bool i8_mode = false);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
