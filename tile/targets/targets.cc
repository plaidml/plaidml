#include "tile/targets/targets.h"

#include "base/config/config.h"
#include "tile/targets/configs.h"

namespace vertexai {
namespace tile {
namespace targets {

codegen::proto::Configs GetConfigs() { return ParseConfig<codegen::proto::Configs>(kConfigs); }

}  // namespace targets
}  // namespace tile
}  // namespace vertexai
