// Copyright 2019, Intel Corporation

#include "mlir/IR/Dialect.h"
#include "pmlc/dialect/hir/ops.h"

namespace pmlc {
namespace dialect {
namespace hir {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("pml_hir", ctx) {
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

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
