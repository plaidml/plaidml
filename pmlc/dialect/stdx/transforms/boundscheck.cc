// Copyright 2020 Intel Corporation

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/stdx/transforms/boundscheck.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

namespace {
template <typename LoadStoreOp>
class BoundsCheckGenerator {
  LoadStoreOp op;
  Location loc;
  ModuleOp module;
  Type i32Type;
  Type indexType;

private:
  BoundsCheckGenerator(LoadStoreOp op, Builder builder)
      : op(op), loc(op.getLoc()),
        module(op.template getParentOfType<ModuleOp>()),
        i32Type(builder.getIntegerType(32)), indexType(builder.getIndexType()) {
  }

public:
  static void generate(LoadStoreOp op) {
    BoundsCheckGenerator generator(op, Builder(op.getContext()));
    generator.generateBoundsChecks();
  }

  FlatSymbolRefAttr getOrInsertFunc() {
    const char *symbol = "plaidml_rt_bounds_check";
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder builder(module.getBodyRegion());
    std::array<Type, 2> inputs{indexType, i32Type};
    ArrayRef<Type> results{};
    auto funcType = builder.getFunctionType(inputs, results);
    ArrayRef<NamedAttribute> attrs{};
    builder.create<FuncOp>(loc, symbol, funcType, attrs);
    return SymbolRefAttr::get(symbol, context);
  }

  Value createConstantIntOp(int64_t value, OpBuilder builder) {
    return builder.create<ConstantIntOp>(loc, value, i32Type);
  }

  void generateBoundsChecks() {
    auto func = getOrInsertFunc();
    OpBuilder opBuilder(op);
    auto dimSizes = op.getMemRefType().getShape();
    for (size_t i = 0; i < dimSizes.size(); i++) {
      auto idxVal = op.getIndices()[i];
      auto rangeVal = createConstantIntOp(dimSizes[i], opBuilder);
      SmallVector<Value, 2> args;
      idxVal.getType().template dyn_cast<mlir::IntegerType>();
      args.push_back(idxVal);
      args.push_back(rangeVal);
      opBuilder.create<CallOp>(op.getOperation()->getLoc(), func,
                               ArrayRef<Type>{}, args);
    }
  }
};

} // namespace

void BoundsCheckPass::runOnFunction() {
  auto func = getFunction();
  func.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<LoadOp>(
            [](auto op) { BoundsCheckGenerator<LoadOp>::generate(op); })
        .Case<StoreOp>(
            [](auto op) { BoundsCheckGenerator<StoreOp>::generate(op); });
  });
}

} // namespace pmlc::dialect::stdx
