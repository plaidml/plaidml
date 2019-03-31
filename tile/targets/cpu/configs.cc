#include "tile/targets/cpu/configs.h"

#include "tile/codegen/driver.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

[[gnu::unused]] auto init = []() {
  codegen::Configs::Register("cpu/cpu", kCpu);
  return 0;
}();

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
