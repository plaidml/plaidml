#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/generate.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

constexpr static size_t k_rng_size = 2048;
void GenSpecial(KernelList& r, const Op& op, const Bindings& bindings,  // NOLINT(runtime/references)
                const std::string& kname, const HardwareSettings& settings);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
