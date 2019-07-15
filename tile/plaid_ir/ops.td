// Copyright 2019, Intel Corporation

include "mlir/IR/OpBase.td"

def PlaidDialect : Dialect {
  let name = "plaid";
  let cppNamespace = "vertexai::tile::plaid_ir";
}

def TensorType : Type<CPred<"$_self.isa<vertexai::tile::plaid_ir::TensorType>()">, "Plaid Tensor Type"> { }

def AffineType : Type<CPred<"$_self.isa<vertexai::tile::plaid_ir::AffineType>()">, "Plaid Affine Type"> { }

def FloatType : Type<CPred<"$_self.isa<vertexai::tile::plaid_ir::FloatType>()">, "Plaid Float Type"> { }

class PlaidOp<string mnemonic, list<OpTrait> traits = []> : Op<PlaidDialect, mnemonic, traits> { }

class NoResult<string mnemonic, list<OpTrait> traits = []> : PlaidOp<mnemonic, traits> {
  let results = (outs);
} 

class AffineResult<string mnemonic, list<OpTrait> traits = [NoSideEffect]> : PlaidOp<mnemonic, traits> {
  let results = (outs AffineType:$result);
} 

def AllocateOp : PlaidOp<"allocate" > {
  let summary = "Allocate a new tensor and set it's intial shape / size";
  let arguments = (ins);
  let results = (outs TensorType:$res);
}

def RefineOp : PlaidOp<"refine", [NoSideEffect]> {
  let summary = "Modify the offset into a tensor in an affine way";
  let arguments = (ins TensorType:$in, Variadic<AffineType>:$offsets);
  let results = (outs TensorType:$result);
}

def LoadOp : PlaidOp<"load"> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorType:$from);
  let results = (outs AnyType:$into);
}

def StoreOp : NoResult<"store"> {
  let summary = "Store to a tensor";
  let arguments = (ins TensorType:$into, AnyType:$from);
}

def AffineConstOp : AffineResult<"affine_const"> {
  let summary = "A constant affine value";
  let arguments = (ins I64Attr:$value);
}

def AffineMulOp : AffineResult<"affine_mul"> {
  let summary = "Multiply an affine by a constant value";
  let arguments = (ins AffineType:$input, I64Attr:$scale);
}

def AffineAddOp : AffineResult<"affine_add"> {
  let summary = "Multiply an affine by a constant value";
  let arguments = (ins Variadic<AffineType>:$inputs);
}

// TODO: Make this a closed rather than open set at some point.
// or at least handle the common cases (fp32 mul) as actual operations
def IntrinsicOp : PlaidOp<"intrinsic", [NoSideEffect]> {
  let summary = "Do an intrinsic";
  let arguments = (ins Variadic<AnyType>:$in, StrAttr:$name);
  let results = (outs Variadic<AnyType>:$out);
}


def ParallelForOp : NoResult<"parallel_for"> {
  let summary = "Parallel for loop";
  // At some point, we should be able to dynamically control ranges
  // Variadic<IntegerType>:$dyn_ranges
  let arguments = (ins I64ArrayAttr:$ranges);
  let regions = (region
    SizedRegion<1>:$inner
  );
}

def AffineIfOp : NoResult<"affine_if"> {
  let arguments = (ins AffineType:$input);
  let regions = (region
    SizedRegion<1>:$tcase
    // TODO: Add false case
    // SizedRegion<1>:$fcase
  );
}

def TerminateOp : NoResult<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
  let arguments = (ins);
}


