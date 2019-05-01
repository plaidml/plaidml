#include "tile/targets/gpu/configs.h"

#include "tile/codegen/driver.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace gpu {

[[gnu::unused]] auto init = []() {
  codegen::Configs::Register("gpu/amd", kGpuAmd);
  codegen::Configs::Register("gpu/intel_gen9", kGpuIntelGen9);
  codegen::Configs::Register("gpu/nvidia", kGpuNvidia);
  return 0;
}();

}  // namespace gpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
