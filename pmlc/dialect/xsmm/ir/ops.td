// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def FuncTypeAttr : TypeAttrBase<"::mlir::FunctionType", "function type attribute"> {
  let constBuilderCall = "::mlir::TypeAttr::get($0)";
}

def XSMM_UnaryKind : I64EnumAttr<
    "UnaryKind",
    "Operation to aggregate with",
    [
      I64EnumAttrCase<"NONE", 0>,
      I64EnumAttrCase<"IDENTITY", 1>,
      I64EnumAttrCase<"XOR", 2>,
      I64EnumAttrCase<"X2", 3>,
      I64EnumAttrCase<"SQRT", 4>,
      I64EnumAttrCase<"RELU", 5>,
      I64EnumAttrCase<"RELU_INV", 6>,
      I64EnumAttrCase<"TANH", 7>,
      I64EnumAttrCase<"TANH_INV", 8>,
      I64EnumAttrCase<"SIGMOID", 9>,
      I64EnumAttrCase<"SIGMOID_INV", 10>,
      I64EnumAttrCase<"GELU", 11>,
      I64EnumAttrCase<"GELU_INV", 12>,
      I64EnumAttrCase<"NEGATE", 13>,
      I64EnumAttrCase<"INC", 14>,
      I64EnumAttrCase<"RECIPROCAL", 15>,
      I64EnumAttrCase<"RECIPROCAL_SQRT", 16>,
      I64EnumAttrCase<"EXP", 17>,
      I64EnumAttrCase<"REDUCE_X_OP_ADD", 18>,
      I64EnumAttrCase<"REDUCE_X2_OP_ADD", 19>,
      I64EnumAttrCase<"REDUCE_X_X2_OP_ADD", 20>,
      I64EnumAttrCase<"REDUCE_X_OP_MAX", 21>,
      I64EnumAttrCase<"REDUCE_X_OP_MUL", 22>,
      I64EnumAttrCase<"REDUCE_X_OP_ADD_NCNC_FORMAT", 23>,
      I64EnumAttrCase<"REDUCE_TO_SCALAR_OP_ADD", 24>,
      I64EnumAttrCase<"DROPOUT", 25>,
      I64EnumAttrCase<"DROPOUT_INV", 26>,
      I64EnumAttrCase<"REPLICATE_COL_VAR", 27>,
      I64EnumAttrCase<"TRANSFORM_NORM_TO_VNNI", 28>,
      I64EnumAttrCase<"TRANSFORM_NORM_TO_NORMT", 29>,
      I64EnumAttrCase<"TRANSFORM_VNNI_TO_VNNIT", 30>,
      I64EnumAttrCase<"TRANSFORM_NORM_TO_VNNIT", 31>,
      I64EnumAttrCase<"TRANSFORM_NORM_TO_VNNI_PAD", 32>,
      I64EnumAttrCase<"UNPACK_TO_BLOCKS", 33>,
      I64EnumAttrCase<"LEAKY_RELU", 34>,
      I64EnumAttrCase<"LEAKY_RELU_INV", 35>,
      I64EnumAttrCase<"ELU", 36>,
      I64EnumAttrCase<"ELU_INV", 37>,
      I64EnumAttrCase<"STOCHASTIC_ROUND", 38>,
    ]> {
  let cppNamespace = "pmlc::dialect::xsmm";
}


def XSMM_BinaryKind : I64EnumAttr<
    "BinaryKind",
    "Operation to aggregate with",
    [
      I64EnumAttrCase<"NONE", 0>,
      I64EnumAttrCase<"ADD", 1>,
      I64EnumAttrCase<"MUL", 2>,
      I64EnumAttrCase<"SUB", 3>,
      I64EnumAttrCase<"DIV", 4>,
      I64EnumAttrCase<"MULADD", 5>,
      I64EnumAttrCase<"MATMUL", 6>,
      I64EnumAttrCase<"MUL_AND_REDUCE_TO_SCALAR_OP_ADD", 7>,
      I64EnumAttrCase<"PACK", 8>,

    ]> {
  let cppNamespace = "pmlc::dialect::xsmm";
}





def XSMM_Dialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

class XSMM_Op<string mnemonic, list<Trait> traits = []>
    : Op<XSMM_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

def XSMM_GemmDispatchF32Op : XSMM_Op<"gemm.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_GemmInvokeF32Op : XSMM_Op<"gemm.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def XSMM_Gemm_ReluDispatchF32Op : XSMM_Op<"gemm.relu.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM and relu shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_Gemm_ReluInvokeF32Op : XSMM_Op<"gemm.relu.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to add bias to C from", [MemRead]>:$d,
   Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
 }];
}

def XSMM_Gemm_Relu_Beta1DispatchF32Op : XSMM_Op<"gemm.relu.beta.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM and relu shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_Gemm_Relu_Beta1InvokeF32Op : XSMM_Op<"gemm.relu.beta.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to add bias to C from", [MemRead]>:$d,
   Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
 }];
}

def XSMM_Gemm_BiasDispatchF32Op : XSMM_Op<"gemm.bias.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM and relu shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_Gemm_BiasInvokeF32Op : XSMM_Op<"gemm.bias.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to add bias to C from", [MemRead]>:$d,
   Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
 }];
}

def XSMM_BRGemmDispatchF32Op : XSMM_Op<"brgemm.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld, I64Attr:$strideA, I64Attr:$strideB);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemmInvokeF32Op : XSMM_Op<"brgemm.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    I64Attr:$numBatches,
    Variadic<Index>:$indices);


  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}


def XSMM_BRGemm_ReluDispatchF32Op : XSMM_Op<"brgemm.relu.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld, I64Attr:$strideA, I64Attr:$strideB);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemm_Relu_Beta1DispatchF32Op : XSMM_Op<"brgemm.beta.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld, I64Attr:$strideA, I64Attr:$strideB);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemm_BiasDispatchF32Op : XSMM_Op<"brgemm.bias.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld, I64Attr:$strideA, I64Attr:$strideB);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemm_ReluInvokeF32Op : XSMM_Op<"brgemm.relu.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to D from", [MemRead]>:$d,
    I64Attr:$numBatches,
    Variadic<Index>:$indices);


  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}

def XSMM_BRGemm_Relu_Beta1InvokeF32Op : XSMM_Op<"brgemm.relu.beta.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to D from", [MemRead]>:$d,
    I64Attr:$numBatches,
    Variadic<Index>:$indices);


  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}

def XSMM_BRGemm_BiasInvokeF32Op : XSMM_Op<"brgemm.bias.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to D from", [MemRead]>:$d,
    I64Attr:$numBatches,
    Variadic<Index>:$indices);


  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}


def XSMM_BRGemmOffsDispatchF32Op : XSMM_Op<"brgemm.offs.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemmOffsInvokeF32Op : XSMM_Op<"brgemm.offs.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    I64Attr:$numBatches,
    I64ArrayAttr:$aOffsets,
    I64ArrayAttr:$bOffsets,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def XSMM_BRGemmOffs_ReluDispatchF32Op : XSMM_Op<"brgemm.offs.relu.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemmOffs_BiasDispatchF32Op : XSMM_Op<"brgemm.offs.bias.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_BRGemmOffs_Relu_Beta1DispatchF32Op : XSMM_Op<"brgemm.offs.relu.beta1.dispatch.f32", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}


def XSMM_BRGemmOffs_ReluInvokeF32Op : XSMM_Op<"brgemm.offs.relu.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to C from", [MemRead]>:$d,
    I64Attr:$numBatches,
    I64ArrayAttr:$aOffsets,
    I64ArrayAttr:$bOffsets,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}

def XSMM_BRGemmOffs_Relu_Beta1InvokeF32Op : XSMM_Op<"brgemm.offs.relu.beta1.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to C from", [MemRead]>:$d,
    I64Attr:$numBatches,
    I64ArrayAttr:$aOffsets,
    I64ArrayAttr:$bOffsets,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}

def XSMM_BRGemmOffs_BiasInvokeF32Op : XSMM_Op<"brgemm.offs.bias.invoke.f32"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    Arg<F32MemRef, "the C memref to store to", [MemWrite]>:$c,
    Arg<F32MemRef, "the A memref to load from", [MemRead]>:$a,
    Arg<F32MemRef, "the B memref to load from", [MemRead]>:$b,
    Arg<F32MemRef, "the D memref to load bias to add to C from", [MemRead]>:$d,
    I64Attr:$numBatches,
    I64ArrayAttr:$aOffsets,
    I64ArrayAttr:$bOffsets,
    Variadic<Index>:$indices);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
    operand_range getOperandsForD();
  }];
}

def XSMM_UnaryDispatchOp : XSMM_Op<"unary.dispatch", [NoSideEffect]> {
  let summary = "Generate a unary kernel for a specific tile size.";
  let arguments = (ins
    XSMM_UnaryKind:$kind,
    TypeAttr:$compute_type,
    I64ArrayAttr:$tile,
    I64Attr:$ldi,
    I64Attr:$ldo,
    FuncTypeAttr:$func_type,
    I32Attr:$bcastType,
    I32Attr:$reduceType
  );
  let results = (outs I64:$ptr);
  let assemblyFormat = [{
    $kind `(` $compute_type `,` $tile `,` $ldi `,` $ldo `,` $bcastType `,` $reduceType `)`
    `:` $func_type attr-dict
  }];
}


def XSMM_UnaryInvokeOp : XSMM_Op<"unary.invoke"> {
  let summary = "Invoke a previously generated unary kernel.";
  let arguments = (ins
    I64:$ptr,
    Arg<AnyMemRef, "the output memref to store to", [MemWrite]>:$output,
    Arg<AnyMemRef, "the input memref to load from", [MemRead]>:$input,
    Variadic<Index>:$indices
  );

  let extraClassDeclaration = [{
    operand_range getOperandsForInput();
    operand_range getOperandsForOutput();
  }];
}

def XSMM_BinaryDispatchOp : XSMM_Op<"binary.dispatch", [NoSideEffect]> {
  let summary = "Generate a binary kernel for a specific tile size.";
  let arguments = (ins
    XSMM_BinaryKind:$kind,
    TypeAttr:$compute_type,
    I64ArrayAttr:$tile,
    I64Attr:$ldi1,
    I64Attr:$ldi2,
    I64Attr:$ldo,
    FuncTypeAttr:$func_type,
    I32Attr:$bcastType1,
    I32Attr:$bcastType2
  );
  let results = (outs I64:$ptr);
  let assemblyFormat = [{
    $kind `(` `bcast1` $bcastType1 `bcast2` $bcastType2 `ldo`
              $ldo `ldi2` $ldi2 `ldi1` $ldi1 `tile` $tile `compute` $compute_type `func` $func_type `)` attr-dict `:` type($ptr)
  }];
}
 
def XSMM_BinaryInvokeOp : XSMM_Op<"binary.invoke"> {
  let summary = "Invoke a previously generated binary kernel.";
  let arguments = (ins
    I64:$ptr,
    Arg<AnyMemRef, "the output memref to store to", [MemWrite]>:$output,
    Arg<AnyMemRef, "the input memref to load from", [MemRead]>:$input1,
    Arg<AnyMemRef, "the input memref to load from", [MemRead]>:$input2,
    Variadic<Index>:$indices
  );
}
 
#endif // __PML_XSMM_OPS__
