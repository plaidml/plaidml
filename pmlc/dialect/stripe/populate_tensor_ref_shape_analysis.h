// Copyright 2019 Intel Corporation

#pragma once

namespace mlir {
class FuncOp;
}  // namespace mlir

namespace pmlc {
namespace dialect {
namespace stripe {

/// This analysis populates the optional shape information of 'tensor_ref' types. This information is found in the
/// 'layout' attribute of 'tensor_ref' BlockArgument/Alloc.
///
/// The algorithm processes all the 'tensor_ref' types in a given FuncOp as follows:
///   1. Traverse use-def chain from the Value associated with the 'tensor_ref' type to a BlockArgument/Alloc.
///   2. Replace 'tensor_ref' type with a new 'tensor_ref' type that contains the shape information.
// NOTE: This analysis is O(n^2) so we should revisit this approach/design if the layout information is needed in
// more cases than the Stripe->Affine dialect conversion.
class PopulateTensorRefShape {
 public:
  explicit PopulateTensorRefShape(mlir::Operation* op);

  void populateWithShapes(mlir::FuncOp func);
};

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
