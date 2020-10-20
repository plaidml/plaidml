// Copyright 2019, Intel Corporation

#ifndef __PML_ABI_OPS__
#define __PML_ABI_OPS__

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"

include "pmlc/util/enums.td"

def ABI_Dialect : Dialect {
  let name = "abi";
  let cppNamespace = "pmlc::dialect::abi";
}

class ABI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ABI_Dialect, mnemonic, traits>;

def ABI_LoopOp : ABI_Op<"loop", [
  DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  ParentOneOf<["FuncOp", "LLVMFuncOp"]>
]> {
  let summary = "Repeat runs of a network";
  let description = [{
    The "abi.loop" operation represents repeated runs of a program,
    taking new inputs and generating new outputs each time.
    This allows loop-invariant code motion to move setup and teardown
    operations outside of the loop, so that their cost is ammortized over
    many runs of the program.
  }];
  let arguments = (ins Variadic<AnyMemRef>:$args);
  let regions = (region AnyRegion:$region);
}

def ABI_DoneOp : ABI_Op<"done", [Terminator]> {
  let summary = "Terminates an abi.loop operation region";
}

#endif // __PML_ABI_OPS__
