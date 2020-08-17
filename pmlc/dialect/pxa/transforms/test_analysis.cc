// Copyright 2020 Intel Corporation

#include <map>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

template <typename T>
static void printStrideInfo(T op) {
  auto info = computeStrideInfo(op);
  llvm::outs() << "strides: ";
  if (!info) {
    llvm::outs() << "none";
  } else {
    auto block = op.getOperation()->getBlock();
    info->print(llvm::outs(), block);
  }
  llvm::outs() << '\n';
  llvm::outs().flush();
}

struct TestStrideInfoPass : public TestStrideInfoBase<TestStrideInfoPass> {
  void runOnOperation() final {
    auto op = getOperation();
    op->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op)
          .Case<PxaLoadOp>([](auto op) { printStrideInfo(op); })
          .Case<PxaReduceOp>([](auto op) { printStrideInfo(op); });
    });
  }
};

struct TestIndirectValuesIteratorPass
    : public TestIndirectValuesIteratorBase<TestIndirectValuesIteratorPass> {
  void runOnOperation() final {
    auto op = getOperation();
    op->walk([&](AllocOp allocOp) {
      llvm::outs() << "alloc: ";
      printDef(allocOp.getOperation());
      for (auto value : getIndirectValues(allocOp)) {
        llvm::outs() << "def: ";
        printDef(value.getDefiningOp());
      }
      llvm::outs() << "alloc end: ";
      printDef(allocOp.getOperation());
      llvm::outs().flush();
    });
  }

  void printDef(Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      llvm::outs() << tag.getValue() << '\n';
    else
      llvm::outs() << debugString(*op) << '\n';
  }
};

struct TestIndirectUsesIteratorPass
    : public TestIndirectUsesIteratorBase<TestIndirectUsesIteratorPass> {
  void runOnOperation() final {
    auto op = getOperation();
    op->walk([&](AllocOp allocOp) {
      llvm::outs() << "alloc: ";
      printDef(allocOp.getOperation());
      for (auto &use : getIndirectUses(allocOp)) {
        llvm::outs() << "use: ";
        printDef(use.getOwner());
      }
      llvm::outs() << "alloc end: ";
      printDef(allocOp.getOperation());
      llvm::outs().flush();
    });
  }

  void printDef(Operation *op) {
    if (auto tag = op->getAttrOfType<StringAttr>("tag"))
      llvm::outs() << tag.getValue() << '\n';
    else
      llvm::outs() << debugString(*op) << '\n';
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTestIndirectUsesIteratorPass() {
  return std::make_unique<TestIndirectUsesIteratorPass>();
}

std::unique_ptr<mlir::Pass> createTestIndirectValuesIteratorPass() {
  return std::make_unique<TestIndirectValuesIteratorPass>();
}

std::unique_ptr<mlir::Pass> createTestStrideInfoPass() {
  return std::make_unique<TestStrideInfoPass>();
}

} // namespace pmlc::dialect::pxa
