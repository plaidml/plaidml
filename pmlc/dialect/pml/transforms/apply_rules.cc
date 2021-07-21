// Copyright 2021 Intel Corporation

#include "pmlc/dialect/pml/transforms/pass_detail.h"

#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pml/ir/dialect.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pml {

struct ApplyRulesPass : public ApplyRulesBase<ApplyRulesPass> {
  void runOnFunction() final {
    FuncOp func = getFunction();
    auto parent = func->getParentOfType<ModuleOp>();
    ModuleOp source = parent.lookupSymbol<ModuleOp>(module);
    if (!source) {
      func.emitError("Source module not found");
      signalPassFailure();
      return;
    }

    ArrayAttr rules;
    for (NamedAttribute attr : source->getDialectAttrs()) {
      if (attr.first.strref() == "pml.rules") {
        rules = attr.second.dyn_cast_or_null<ArrayAttr>();
        break;
      }
    }

    if (!rules) {
      source.emitError("'pml.rules' array not found");
      signalPassFailure();
      return;
    }

    for (ApplyAttr apply : rules.getAsRange<ApplyAttr>()) {
      processRule(apply);
    }

    source.erase();
  }

  void processRule(ApplyAttr apply) {
    PatternAttr pattern = apply.getPattern();
    StringRef opName = pattern.getOp().getValue();
    getFunction().walk([&](Operation *op) {
      if (op->getName().getStringRef() != opName)
        return;

      DictionaryAttr opAttrs = op->getAttrDictionary();
      for (NamedAttribute kvp : pattern.getDict()) {
        if (opAttrs.get(kvp.first) != kvp.second)
          return;
      }

      for (NamedAttribute kvp : apply.getDict()) {
        op->setAttr(kvp.first, kvp.second);
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createApplyRulesPass() {
  return std::make_unique<ApplyRulesPass>();
}

} // namespace pmlc::dialect::pml
