// Copyright 2021, Intel Corporation

#ifndef __PML_SCHEDULE__
#define __PML_SCHEDULE__

include "mlir/IR/OpBase.td"

def PML_Dialect : Dialect {
  let name = "pml";
  let cppNamespace = "pmlc::util";
}

def PML_AxisAttr : AttrDef<PML_Dialect, "Axis"> {
  let mnemonic = "axis";

  let parameters = (ins
    "mlir::StringAttr":$name,
    "int64_t":$range
  );
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
}

#endif // __PML_SCHEDULE__
