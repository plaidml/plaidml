// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
#define __PML_TILE_BASE__

include "mlir/Interfaces/SideEffects.td"

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
#endif

include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"
include "pmlc/dialect/eltwise/ir/interfaces.td"

#ifndef __PML_ELTWISE_PREDICATES__
include "pmlc/dialect/eltwise/ir/predicates.td"
#endif

include "pmlc/dialect/tile/ir/interfaces.td"

// TODO: this should be defined upstream
def IntegerSetAttr : Attr<
    CPred<"$_self.isa<IntegerSetAttr>()">, "IntegerSet attribute"> {
  let storageType = [{ IntegerSetAttr }];
  let returnType = [{ IntegerSet }];
  let constBuilderCall = "IntegerSetAttr::get($0)";
}

// TODO: this should be defined upstream
def IndexAttr : Attr<
      And<[CPred<"$_self.isa<IntegerAttr>()">,
           CPred<"$_self.cast<IntegerAttr>().getType().isIndex()">]>,
      "index attribute"> {
  let constBuilderCall = "$_builder.getIntegerAttr($_builder.getIndexType(), $0)";
  let storageType = "IntegerAttr";
  let returnType = [{ APInt }];
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

def AffineMap : DialectType<TileDialect,
  CPred<"$_self.isa<AffineMapType>()">,
  "affine-map">;

def AffineConstraints : DialectType<TileDialect,
  CPred<"$_self.isa<AffineConstraintsType>()">,
  "affine-constraints">;

def AffineTensorMap : DialectType<TileDialect,
  CPred<"$_self.isa<AffineTensorMapType>()">,
  "affine-tensor-map">;

def StrType : DialectType<TileDialect,
  CPred<"$_self.isa<StringType>()">,
  "string">;

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

class TileOpWithPPV<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, traits> {
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

#endif // __PML_TILE_BASE__
