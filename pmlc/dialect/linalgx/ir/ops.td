// Copyright 2021 Intel Corporation

#ifndef __PML_LINALGX_OPS__
#define __PML_LINALGX_OPS__

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LinalgX_Dialect : Dialect {
  let name = "linalgx";
  let cppNamespace = "pmlc::dialect::linalgx";
}

class LinalgX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LinalgX_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def CopyOp : LinalgX_Op<"copy", [
  CopyOpInterface,
  SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">,
  LinalgStructuredInterface,
  NoSideEffect
]> {
  let summary = "tensor copy operation";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$output,
    OptionalAttr<AffineMapAttr>:$inputMap,
    OptionalAttr<AffineMapAttr>:$outputMap);
  let results = (outs AnyRankedTensor:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `,` $output `)` attr-dict `:` type($input) `,` type($output)
      `->` type($result)
      custom<CopyOpRegion>($region, ref(type($input)), ref(type($input)))
  }];

  let builders = [
    OpBuilder<(ins
      "mlir::Value":$input,
      "mlir::Value":$output,
      CArg<"mlir::AffineMap", "mlir::AffineMap()">:$inputMap,
      CArg<"mlir::AffineMap", "mlir::AffineMap()">:$outputMap
    )>
  ];

  let extraClassDeclaration = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() { return false; }

    mlir::ValueRange inputs() { return getOperands().take_front(); }
    mlir::ValueRange outputs() { return getOperands().take_back(); }

    mlir::ArrayAttr indexing_maps();
    mlir::ArrayAttr iterator_types();

    static void regionBuilder(mlir::ImplicitLocOpBuilder &b, mlir::Block &block);
    static std::function<void(mlir::ImplicitLocOpBuilder &b, mlir::Block &block)>
    getRegionBuilder() { return &regionBuilder; }
    static unsigned getNumRegionArgs() { return 2; }

    mlir::Value getSource() { return input(); }
    mlir::Value getTarget() { return output(); }
  }];
}

#endif // __PML_LINALGX_OPS__
