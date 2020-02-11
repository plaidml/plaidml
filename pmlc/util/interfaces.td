#ifndef __PML_UTIL_INTERFACES__
#define __PML_UTIL_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def GenericBuilderInterface : OpInterface<"GenericBuilder"> {
  let methods = [
    StaticInterfaceMethod<"TODO", "Type", "getResultType", (ins "ArrayRef<Value>":$operands)>,
    StaticInterfaceMethod<"TODO", "Operation*", "create", (ins
      "OpBuilder*":$builder,
      "Location":$loc,
      "Type":$type,
      "ArrayRef<Value>":$operands
    )>,
  ];
}

#endif // __PML_UTIL_INTERFACES__
