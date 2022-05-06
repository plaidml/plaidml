// Copyright 2021 Intel Corporation

#include "pmlc/dialect/pml/transforms/pass_detail.h"
#include "mlir/IR/BuiltinOps.h"
#include "pmlc/dialect/pml/ir/dialect.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pml {

struct ApplyRulesPass : public ApplyRulesBase<ApplyRulesPass> {
  ApplyRulesPass() = default;

  explicit ApplyRulesPass(StringRef module) { this->module = module.str(); }

  void runOnOperation() final {
    auto source = getOperation().lookupSymbol<ModuleOp>(module);
    if (!source) {
      getOperation().emitError("Source module not found");
      signalPassFailure();
      return;
    }

    ArrayAttr rules;
    for (NamedAttribute attr : source->getDialectAttrs()) {
      if (attr.getName().strref() == "pml.rules") {
        rules = attr.getValue().dyn_cast_or_null<ArrayAttr>();
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
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef() != opName)
        return;

      DictionaryAttr opAttrs = op->getAttrDictionary();
      for (NamedAttribute kvp : pattern.getDict()) {
        if (opAttrs.get(kvp.getName()) != kvp.getValue())
          return;
      }

      for (NamedAttribute kvp : apply.getDict()) {
        op->setAttr(kvp.getName(), kvp.getValue());
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createApplyRulesPass() {
  return std::make_unique<ApplyRulesPass>();
}

std::unique_ptr<mlir::Pass> createApplyRulesPass(StringRef module) {
  return std::make_unique<ApplyRulesPass>(module);
}

} // namespace pmlc::dialect::pml
