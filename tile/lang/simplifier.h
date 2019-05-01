#pragma once

#include <vector>

#include "tile/lang/generate.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

void Simplify(sem::StmtPtr stmt);
void Simplify(const std::vector<KernelInfo>& kernels);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
