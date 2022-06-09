// Copyright 2022 Intel Corporation

#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "pmlc/dialect/linalgx/analysis/convolution.h"

using namespace mlir; // NOLINT
using mlir::matchers::m_Val;

namespace pmlc::dialect::linalgx {

Optional<ConvCapture> detectConv(linalg::GenericOp op) {
  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    return None;

  Block *block = op.getBody();
  Block::BlockArgListType args = block->getArguments();
  if (args.size() != 3)
    return None;

  ValueRange values = op.getOperands();
  SmallVector<AffineMap> idxMaps = op.getIndexingMaps();

  Operation *yieldOp = block->getTerminator();
  if (matchPattern(
          yieldOp,
          m_Op<linalg::YieldOp>(m_Op<arith::AddFOp>(
              m_Val(args[2]), m_Op<arith::MulFOp>(m_Val(args[0]), m_Val(args[1]))))) ||
      matchPattern(yieldOp,
                   m_Op<linalg::YieldOp>(m_Op<arith::AddFOp>(m_Op<arith::MulFOp>(
                                             m_Val(args[0]), m_Val(args[1]))),
                                         m_Val(args[2]))))
    return ConvCapture{values, idxMaps, {0, 1, 2}};

  return None;
}

} // namespace pmlc::dialect::linalgx
