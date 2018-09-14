#pragma once

#include <string>

#include "tile/lang/compose.h"
#include "tile/proto/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace lang {

stripe::proto::Block GenerateStripe(const RunInfo& runinfo);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
