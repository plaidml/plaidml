#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace pmlc::dialect::tile {
struct PaddingInfo;
}

namespace mlir {
class Value;
}

namespace pmlc::conversion::tile_to_pxa {

struct TileToPXATypeConverter : public mlir::TypeConverter {
  TileToPXATypeConverter();
};

mlir::Value
buildSimpleStore(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value scalar, mlir::Value memRef,
                 mlir::Optional<pmlc::dialect::tile::PaddingInfo> maybePadding);

struct BufferAllocator {
  mlir::Value resultMemRef;
  mlir::RankedTensorType rankedTensorType;
  mlir::MemRefType memRefType;
  mlir::Type elementType;

  BufferAllocator(mlir::OpBuilder &builder, mlir::Operation *op,
                  mlir::Type resultType);
};

} // namespace pmlc::conversion::tile_to_pxa
