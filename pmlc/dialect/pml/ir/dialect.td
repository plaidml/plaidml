// Copyright 2021, Intel Corporation

#ifndef __PML_SCHEDULE__
#define __PML_SCHEDULE__

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def PML_Dialect : Dialect {
  let name = "pml";
  let cppNamespace = "pmlc::dialect::pml";
}

def PML_AxisAttr : AttrDef<PML_Dialect, "Axis"> {
  let mnemonic = "axis";

  let parameters = (ins
    "mlir::StringAttr":$name,
    "int64_t":$range
  );
  let hasCustomAssemblyFormat = 1;
}

def PML_ScheduleAttr : AttrDef<PML_Dialect, "Schedule"> {
  let mnemonic = "schedule";

  let parameters = (ins
    "mlir::AffineMap":$map,
    ArrayRefParameter<"AxisAttr", "Per-axis definition">:$axes
  );

  let extraClassDeclaration = [{
    mlir::Optional<AxisDim> getAxisInputDim(mlir::StringRef name);
    mlir::Optional<AxisDim> getAxisResultDim(mlir::StringRef name);
    ScheduleAttr removeAxes(mlir::DenseSet<mlir::StringRef> names);
  }];
  let hasCustomAssemblyFormat = 1;
}

def PML_PatternAttr : AttrDef<PML_Dialect, "Pattern"> {
  let mnemonic = "pattern";

  let parameters = (ins
    "mlir::StringAttr":$op,
    "mlir::DictionaryAttr":$dict
  );

  let hasCustomAssemblyFormat = 1;
}

def PML_ApplyAttr : AttrDef<PML_Dialect, "Apply"> {
  let mnemonic = "apply";

  let parameters = (ins
    "PatternAttr":$pattern,
    "mlir::DictionaryAttr":$dict
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // __PML_SCHEDULE__
