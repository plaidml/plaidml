#ifndef __PML_UTIL_INTERFACES__
#define __PML_UTIL_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def GenericBuilderInterface : OpInterface<"GenericBuilder"> {
  let methods = [
    StaticInterfaceMethod<
      /*desc=*/"TODO",
      /*retTy=*/"Type",
      /*methodName=*/"getResultType",
      /*args=*/(ins "ArrayRef<Value>":$operands)
    >,
    StaticInterfaceMethod<
      /*desc=*/"TODO",
      /*retTy=*/"Operation*",
      /*methodName=*/"create",
      /*args=*/(ins
        "OpBuilder*":$builder,
        "Location":$loc,
        "ArrayRef<Value>":$operands),
      /*methodBody=*/[{
        OperationState state(loc, ConcreteOp::getOperationName());
        state.addOperands(operands);
        state.addTypes(getResultType(operands));
        return builder->createOperation(state);
      }]
    >,
  ];
}

#endif // __PML_UTIL_INTERFACES__
