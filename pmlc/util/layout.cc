// Copyright 2020 Intel Corporation
#include "pmlc/util/layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/IR/BuiltinTypes.h"

namespace pmlc {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

MLFramework getMLFramework(StringRef opName) {
  StringRef nGraphPrefix = "ng.";
  if (opName.substr(0, nGraphPrefix.size()) == nGraphPrefix) {
    return MLFramework::NGraph;
  } else {
    return MLFramework::Default;
  }
}

TensorLayout getLayoutType(MLFramework framework, StringRef opName,
                           bool isConst) {
  // TODO: Create a common list of primitives that would be used here and in the
  // plugins so we know they are consistent. For now use namings from OV plugin
  // Const are used to identify the primitives weights or parameters, rest is
  // considered as main flow data type
  if (framework == MLFramework::NGraph) {
    if ((opName.find("Convolution") != StringRef::npos) && isConst) {
      return TensorLayout::KCX;
    } else {
      return TensorLayout::NCX;
    }
  }

  return TensorLayout::NXC;
}

} // namespace pmlc
