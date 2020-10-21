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
  IsolatedFromAbove, HasParent<"ModuleOp">]> {
  let summary = "Repeat runs of a network";
  let description = [{
    The "abi.loop" operation represents repeated runs of a program,
    taking new inputs and generating new outputs each time.
    This allows loop-invariant code motion to move setup and teardown
    operations outside of the loop, so that their cost is ammortized over
    many runs of the program.
  }];
  let regions = (region AnyRegion:$initRegion, AnyRegion:$bodyRegion, AnyRegion:$finiRegion);
  let extraClassDeclaration = [{
    mlir::Block* bodyEntryBlock() { return &bodyRegion().front(); }
    
    CreateNetworkOp getInitTerminator() {
      return mlir::cast<CreateNetworkOp>(initRegion().back().getTerminator());
    }
  }];
}

def ABI_DoneOp : ABI_Op<"done", [Terminator, HasParent<"pmlc::dialect::abi::LoopOp">]> {
  let summary = "Terminates an abi.loop operation body or fini region";
}

def ABI_CreateNetworkOp : ABI_Op<"create_network", [Terminator, HasParent<"pmlc::dialect::abi::LoopOp">]> {
  let summary = "Terminates an abi.loop operation init region";
  let arguments = (ins Variadic<AnyType>);
}

#endif // __PML_ABI_OPS__
