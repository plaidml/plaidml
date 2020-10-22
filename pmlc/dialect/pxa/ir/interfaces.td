// Copyright 2020, Intel Corporation

#ifndef __PMLC_PXA_INTERFACES__
#define __PMLC_PXA_INTERFACES__

include "mlir/IR/OpBase.td"

class PxaOpInterface<string name> : OpInterface<name>;

def PxaReadOpInterface : PxaOpInterface<"PxaReadOpInterface"> {
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
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<::mlir::MemRefType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 1);
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
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the AffineMapAttr associated with 'memref'.",
      /*retTy=*/"::mlir::NamedAttribute",
      /*methodName=*/"getAffineMapAttrForMemRef",
      /*args=*/(ins "::mlir::Value":$memref),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        assert(memref == getMemRef());
        return {::mlir::Identifier::get(op.getMapAttrName(), op.getContext()),
                op.getAffineMapAttr()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value read by this operation.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValue",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::cast<ConcreteOp>(this->getOperation());
      }]
    >,
  ];
}

def PxaReduceOpInterface : PxaOpInterface<"PxaReduceOpInterface"> {
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
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the memref operand.",
      /*retTy=*/"::mlir::MemRefType",
      /*methodName=*/"getMemRefType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<::mlir::MemRefType>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns affine map operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getMapOperands",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 2);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the affine map used to index the memref for this operation.
      }],
      /*retTy=*/"::mlir::AffineMap",
      /*methodName=*/"getAffineMap",
      /*args=*/(ins),
      /*methodName=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the AffineMapAttr associated with 'memref'.",
      /*retTy=*/"::mlir::NamedAttribute",
      /*methodName=*/"getAffineMapAttrForMemRef",
      /*args=*/(ins "::mlir::Value":$memref),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        assert(memref == getMemRef());
        return {::mlir::Identifier::get(op.getMapAttrName(), op.getContext()),
                op.getAffineMapAttr()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the value to store.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getValueToStore",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getStoredValOperandIndex());
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the agg.",
      /*retTy=*/"::mlir::AtomicRMWKind",
      /*methodName=*/"getAgg",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ConcreteOp op = ::mlir::cast<ConcreteOp>(this->getOperation());
        return op.getAgg();
      }]
    >,
  ];
}

#endif // __PMLC_PXA_INTERFACES__
