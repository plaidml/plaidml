// Copyright 2020 Intel Corporation

#ifndef __PML_ABI_OPS__
#define __PML_ABI_OPS__

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pmlc/util/loop_with_epilog.td"

include "pmlc/util/enums.td"

def ABI_Dialect : Dialect {
  let name = "abi";
  let cppNamespace = "pmlc::dialect::abi";
}

class ABI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ABI_Dialect, mnemonic, traits>;

def ABI_LoopOp : ABI_Op<"loop", [
  DeclareOpInterfaceMethods<LoopLikeOpInterface>,
  DeclareOpInterfaceMethods<LoopWithEpilogInterface>,
  IsolatedFromAbove, HasParent<"ModuleOp">, RecursiveSideEffects]> {
  let summary = "Repeat runs of a network";
  let description = [{
    The "abi.loop" operation represents repeated runs of a program,
    taking new inputs and generating new outputs each time.
    This allows loop-invariant code motion to move setup and teardown
    operations outside of the loop, so that their cost is ammortized over
    many runs of the program.
  }];
  let regions = (region AnyRegion:$initRegion, AnyRegion:$bodyRegion, AnyRegion:$finiRegion);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    static mlir::StringRef getNetworkFieldTypesAttrName() { return "networkFieldTypes"; }

    std::vector<mlir::Type> getNetworkFieldTypes() {
      std::vector<mlir::Type> result;
      auto arrayAttr = getAttrOfType<mlir::ArrayAttr>(getNetworkFieldTypesAttrName());
      if (arrayAttr) {
        for (auto attr : arrayAttr) {
          if (auto tyAttr = attr.dyn_cast<mlir::TypeAttr>()) {
            result.emplace_back(tyAttr.getValue());
          }
        }
      }
      return result;
    }

    unsigned getNumNetworkFields() {
      auto arrayAttr = getAttrOfType<mlir::ArrayAttr>(getNetworkFieldTypesAttrName());
      if (arrayAttr) {
        return arrayAttr.size();
      }
      return 0;
    }

    void setNetworkFieldTypes(mlir::TypeRange types) {
      mlir::SmallVector<mlir::Attribute, 8> attrs;
      for (auto ty : types) {
        attrs.emplace_back(mlir::TypeAttr::get(ty));
      }
      auto arrayAttr = mlir::ArrayAttr::get(attrs, getContext());
      setAttr(getNetworkFieldTypesAttrName(), arrayAttr);
    }

    mlir::Block* getBodyEntryBlock() { return &bodyRegion().front(); }
    mlir::Block* getFiniEntryBlock() { return &finiRegion().front(); }
    
    CreateNetworkOp getInitTerminator() {
      return mlir::cast<CreateNetworkOp>(initRegion().back().getTerminator());
    }

    DoneOp getFiniTerminator() {
      return mlir::cast<DoneOp>(finiRegion().back().getTerminator());
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
