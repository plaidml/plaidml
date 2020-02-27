// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
#define __PML_TILE_BASE__

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif

include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"
include "pmlc/dialect/eltwise/ir/interfaces.td"
include "pmlc/dialect/eltwise/ir/predicates.td"
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
}

def AffineMap : Type<CPred<"$_self.isa<AffineMapType>()">, "affine-map">;
def AffineConstraints : Type<CPred<"$_self.isa<AffineConstraintsType>()">, "affine-constraints">;
def AffineTensorMap : Type<CPred<"$_self.isa<AffineTensorMapType>()">, "affine-tensor-map">;
def StrType : Type<CPred<"$_self.isa<StringType>()">, "string">;

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

class TileOpWithPPV<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, traits> {
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

#endif // __PML_TILE_BASE__
