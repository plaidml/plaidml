#ifndef __PMLC_DIALECT_PML_PASSES__
#define __PMLC_DIALECT_PML_PASSES__

include "mlir/Pass/PassBase.td"

def ApplyRules : Pass<"pml-apply-rules", "mlir::ModuleOp"> {
  let summary = "Apply rules";
  let constructor = "pmlc::dialect::pml::createApplyRulesPass()";
  let options = [
    Option<"module", "module", "std::string", /*default=*/"\"\"",
           "Module to apply rules from">
  ];
}

def LoadModule : Pass<"pml-load-module", "mlir::ModuleOp"> {
  let summary = "Load a module from an external file";
  let constructor = "pmlc::dialect::pml::createLoadModulePass()";
  let options = [
    Option<"path", "path", "std::string", /*default=*/"\"\"",
           "The path to load a module from">
  ];
  let dependentDialects = ["pmlc::dialect::pml::PMLDialect"];
}

#endif // __PMLC_DIALECT_PML_PASSES__
