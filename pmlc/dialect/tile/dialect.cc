// Copyright 2019, Intel Corporation

#include "mlir/IR/Dialect.h"
#include "pmlc/dialect/tile/ops.h"

namespace pmlc {
namespace dialect {
namespace tile {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("tile", ctx) {
    // addTypes<TensorType>();
    // addTypes<AffineType>();
    //   addOperations<
    // #define GET_OP_LIST
    // #include "tile/plaid_ast/ast.cpp.inc"
    //       >();
    // printf("Got to AstDialect::AstDialect\n");
  }
};

static mlir::DialectRegistration<Dialect> EdslOps;

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
