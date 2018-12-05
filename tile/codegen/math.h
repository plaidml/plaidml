// Copyright 2018, Intel Corporation

#pragma once

namespace vertexai {
namespace tile {
namespace codegen {

template <typename X, typename Y>
inline auto IntDivCeil(X x, Y y) -> decltype((x + y - 1) / y) {
  return (x + y - 1) / y;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
