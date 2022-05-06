#pragma once

#include "pmlc/dialect/linalgx/ir/ops.h"

namespace pmlc::dialect::linalgx {

struct ConvOperand {
  mlir::Value value;
  mlir::RankedTensorType type;
  mlir::AffineMap idxMap;

  ConvOperand(mlir::ValueRange values, llvm::ArrayRef<mlir::AffineMap> idxMaps,
              int64_t i)
      : value(values[i]), type(value.getType().cast<mlir::RankedTensorType>()),
        idxMap(idxMaps[i]) {}
};

struct ConvCapture {
  ConvOperand input;
  ConvOperand filter;
  ConvOperand output;

  ConvCapture(mlir::ValueRange values, llvm::ArrayRef<mlir::AffineMap> idxMaps,
              llvm::ArrayRef<int64_t> order)
      : input(values, idxMaps, order[0]), filter(values, idxMaps, order[1]),
        output(values, idxMaps, order[2]) {}

  mlir::RankedTensorType getBlockedInputType(int64_t blockSize) {
    llvm::ArrayRef<int64_t> shape = input.type.getShape();
    return mlir::RankedTensorType::get({shape[0],             //
                                        shape[3] / blockSize, //
                                        shape[1],             //
                                        shape[2],             //
                                        blockSize},
                                       input.type.getElementType());
  }

  mlir::RankedTensorType getBlockedFilterType(int64_t blockSize) {
    llvm::ArrayRef<int64_t> shape = filter.type.getShape();
    return mlir::RankedTensorType::get(
        {shape[2] / blockSize,                     //
         shape[3] == 1 ? 1 : shape[3] / blockSize, //
         shape[0],                                 //
         shape[1],                                 //
         blockSize,                                //
         blockSize},
        filter.type.getElementType());
  }

  mlir::RankedTensorType getBlockedOutputType(int64_t blockSize) {
    llvm::ArrayRef<int64_t> shape = output.type.getShape();
    return mlir::RankedTensorType::get({shape[0],             //
                                        shape[3] / blockSize, //
                                        shape[1],             //
                                        shape[2],             //
                                        blockSize},
                                       input.type.getElementType());
  }
};

llvm::Optional<ConvCapture> detectConv(mlir::linalg::GenericOp op);

} // namespace pmlc::dialect::linalgx
