// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

//include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
#endif

def XSMM_Dialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

//class XSMM_Op<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
class XSMM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XSMM_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def XSMM_GemmOp : XSMM_Op<"gemm"> {
  let summary = "general matrix multiply operation";
  let description = [{
    Perform a dense matrix-matrix multiplication.

    C_m,n = A_m,k * B_k,n

    Example:
      affine.parallel (%i, %j, %k) = (0, 0, 0) to (8, 8, 8) step (2, 2, 2) {
        xsmm.gemm %C[%i, %j]:#tileC = %A[%i, %k]:#tileA, %B[%k, %j]:#tileB, [2, 2, 2]
          : memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>
      }
  }];

  let arguments = (ins
    AnyMemRef:$c, AffineMapAttr:$cAccessMap, AffineMapAttr:$cTileMap,
    AnyMemRef:$a, AffineMapAttr:$aAccessMap, AffineMapAttr:$aTileMap,
    AnyMemRef:$b, AffineMapAttr:$bAccessMap, AffineMapAttr:$bTileMap,
    I64ArrayAttr:$tile,
    Variadic<Index>:$mapOperands);
  let results = (outs AnyMemRef:$cout);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

//def XSMM_DispatchGemmOp : XSMM_Op<"dispatchgemm", [NoSideEffect]> {
def XSMM_DispatchGemmOp : XSMM_Op<"dispatchgemm"> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";

  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
}

def XSMM_InvokeGemmOp : XSMM_Op<"invokegemm"> {
  let summary = "invoke a preivously queried LIBXSMM kernel";

  let arguments = (ins I64:$ptr,
    AnyMemRef:$c, AffineMapAttr:$cAccessMap, AffineMapAttr:$cTileMap,
    AnyMemRef:$a, AffineMapAttr:$aAccessMap, AffineMapAttr:$aTileMap,
    AnyMemRef:$b, AffineMapAttr:$bAccessMap, AffineMapAttr:$bTileMap,
    I64ArrayAttr:$tile,
    Variadic<Index>:$mapOperands);
  let results = (outs AnyMemRef:$cout);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

#endif // __PML_XSMM_OPS__
