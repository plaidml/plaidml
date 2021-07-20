#ifndef __PMLC_DIALECT_PML_PASSES__
#define __PMLC_DIALECT_PML_PASSES__

include "mlir/Pass/PassBase.td"

def ApplyRules : FunctionPass<"pml-apply-rules"> {
  let summary = "Apply rules";
  let constructor = "pmlc::dialect::pml::createApplyRulesPass()";
  let options = [
    Option<"module", "module", "std::string", /*default=*/"\"\"",
           "Module to apply rules from">
  ];
}

#endif // __PMLC_DIALECT_PML_PASSES__
