// Copyright 2020, Intel Corporation

#ifndef __PMLC_PXA_INTERFACES__
#define __PMLC_PXA_INTERFACES__

include "mlir/IR/OpBase.td"

def PxaGenericOpInterface : OpInterface<"PxaGenericOpInterface"> {
  let description = [{}];

  let cppNamespace = "pmlc::dialect::pxa";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"inputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"outputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of inputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.inputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of outputs.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumOutputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.outputs().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input operands.
      }],
      /*retTy=*/"::mlir::OpOperandVector",
      /*methodName=*/"getInputOperands",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the output operands.
      }],
      /*retTy=*/"::mlir::OpOperandVector",
      /*methodName=*/"getOutputOperands",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the value type for `opOperand`.
      }],
      /*retTy=*/"::mlir::SmallVector<int64_t>",
      /*methodName=*/"getTiedVectorShape",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the AffineValueMap for `opOperand`.
      }],
      /*retTy=*/"::mlir::AffineValueMap",
      /*methodName=*/"getTiedAffineValueMap",
      /*args=*/(ins "::mlir::OpOperand *":$opOperand)
    >,
    InterfaceMethod<
      /*desc=*/"Return the input PxaMemAccessOperands.",
      /*retTy=*/"::mlir::SmallVector<pmlc::dialect::pxa::PxaMemAccessOperand>",
      /*methodName=*/"getInputMemAccesses",
      /*args=*/(ins),
      /*methodBody=*/[{
        ::mlir::SmallVector<pmlc::dialect::pxa::PxaMemAccessOperand> result;
        result.reserve($_op.getNumInputs());
        for (::mlir::OpOperand *operand : $_op.getInputOperands())
          result.emplace_back(operand);
        return result;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the output PxaMemAccessOperands.",
      /*retTy=*/"::mlir::SmallVector<pmlc::dialect::pxa::PxaMemAccessOperand>",
      /*methodName=*/"getOutputMemAccesses",
      /*args=*/(ins),
      /*methodBody=*/[{
        ::mlir::SmallVector<pmlc::dialect::pxa::PxaMemAccessOperand> result;
        result.reserve($_op.getNumOutputs());
        for (::mlir::OpOperand *operand : $_op.getOutputOperands())
          result.emplace_back(operand);
        return result;
      }]
    >,
  ];
}

def PxaReadOpInterface : OpInterface<"PxaReadOpInterface"> {
  let description = [{
      Interface to query characteristics of read-like ops with affine
      restrictions.
  }];

  let cppNamespace = "pmlc::dialect::pxa";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the memref operand to read from.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMemRef",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getMemRef();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getMemRef().getType().template cast<::mlir::MemRefType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return llvm::drop_begin($_op.getOperands(), 1);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the affine map used to index the memref for this operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getAffineMap",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value read by this operation.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.result();
      }]
    >,
  ];
}

def PxaReduceOpInterface : OpInterface<"PxaReduceOpInterface"> {
  let description = [{
      Interface to query characteristics of write-like ops with affine
      restrictions.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the memref operand to write to.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getMemRef",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getMemRef();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getMemRef().getType().template cast<::mlir::MemRefType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return llvm::drop_begin($_op.getOperands(), 2);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the affine map used to index the memref for this operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getAffineMap",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value to store.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValueToStore",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getOperand($_op.getStoredValOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the agg.",
      /*retTy=*/"::mlir::AtomicRMWKind",
      /*methodName=*/"getAgg",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getAgg();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the idxs.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getIdxs",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.idxs();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the result value.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getReduceResult",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.result();
      }]
    >,
  ];
}

#endif // __PMLC_PXA_INTERFACES__
