// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

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

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

#endif // __PML_XSMM_OPS__
