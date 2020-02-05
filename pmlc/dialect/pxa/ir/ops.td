// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif

#ifndef MLIR_LOOPLIKEINTERFACE
include "mlir/Transforms/LoopLikeInterface.td"
#endif

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

def PXA_Dialect : Dialect {
  let name = "pxa";
  let cppNamespace = "pmlc::dialect::pxa";
}

class PXA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PXA_Dialect, mnemonic, traits>;

class PXA_OpWithPP<string mnemonic, list<OpTrait> traits = []> :
    PXA_Op<mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def AnyStdScalar : AnyTypeOf<[AnyFloat, AnyInteger]> {}

def AffineReduceOp : PXA_OpWithPP<"reduce"> {
  let summary = "Reduce into memory";
  let arguments = (ins
    AggregationKind:$agg,
    AnyStdScalar:$val,
    AnyMemRef:$out,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
}

def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineTerminatorOp">;

def AffineParallelOp : PXA_Op<"parallel", [ImplicitAffineTerminator]> {
  let summary = "multi-index parallel for operation";
  let arguments = (ins 
     AffineMapAttr:$lowerBoundsMap,
     AffineMapAttr:$upperBoundsMap,
     I64ArrayAttr:$steps,
     Variadic<Index>:$mapOperands);
  let regions = (region SizedRegion<1>:$region);
  let verifier = [{ return verifyAffineParallelOp(*this); }];

  let extraClassDeclaration = [{
    operand_range getLowerBoundsOperands();
    operand_range getUpperBoundsOperands();

    // Get the number of dimensions (i.e. number of induction variables)    
    size_t getNumDims();

    // Get various AffineValueMaps
    AffineValueMap getLowerBoundsValueMap(); 
    AffineValueMap getUpperBoundsValueMap(); 
    AffineValueMap getRangesValueMap();

    // Get ranges as constants, may fail in dynamic case
    llvm::Optional<llvm::SmallVector<int64_t, 8>> getConstantRanges();
    
    // Some helpers for rewrites
    mlir::Block *getBody();
    mlir::OpBuilder getBodyBuilder();

    void setSteps(ArrayRef<int64_t> newSteps);
  }];
  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, ArrayRef<int64_t> ranges">,
    OpBuilder<"Builder* builder, OperationState& result, AffineMap lbMap, ValueRange lbArgs, AffineMap ubMap, ValueRange ubArgs">,
    OpBuilder<"Builder* builder, OperationState& result, AffineMap lbMap, ValueRange lbArgs, AffineMap ubMap, ValueRange ubArgs, ArrayRef<int64_t> steps">
  ];
}

#endif // __PML_PXA_OPS__
