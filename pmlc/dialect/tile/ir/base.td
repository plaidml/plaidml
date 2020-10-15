// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
#define __PML_TILE_BASE__

include "mlir/Interfaces/SideEffectInterfaces.td"

include "pmlc/util/enums.td"

#ifndef __PML_ELTWISE_BASE__
include "pmlc/dialect/eltwise/ir/base.td"
#endif

// TODO: this should be defined upstream
def IntegerSetAttr : Attr<
    CPred<"$_self.isa<IntegerSetAttr>()">, "IntegerSet attribute"> {
  let storageType = [{ IntegerSetAttr }];
  let returnType = [{ IntegerSet }];
  let constBuilderCall = "IntegerSetAttr::get($0)";
}

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "pmlc::dialect::tile";
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    static std::string getDialectAttrName(llvm::StringRef name);
    static std::string getCanonicalOpName(llvm::StringRef name);
  }];
}

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

class TileOpWithPPV<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, traits> {
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

#endif // __PML_TILE_BASE__
