// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/predicates.td"

def StripeDialect : Dialect {
  let name = "stripe";
  let cppNamespace = "pmlc::dialect::stripe";
}

def AffineType : Type<CPred<"$_self.isa<AffineType>()">, "Plaid Affine Type">;
def TensorType : Type<CPred<"$_self.isa<TensorType>()">, "Plaid Tensor Type">;
def TensorRefType : Type<CPred<"$_self.isa<TensorRefType>()">, "Plaid Tensor Reference Type">;

class StripeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StripeDialect, mnemonic, traits>;

def AllocateOp : StripeOp<"alloc"> {
  let summary = "Allocate a new tensor and set its intial shape / size";
  let results = (outs TensorType:$result);
}

def TensorRefOp : StripeOp<"ref", [NoSideEffect]> {
  let summary = "Create a default tensor_ref from a tensor";
  let arguments = (ins TensorType:$in);
  let results = (outs TensorRefType:$result);
}

def RefineOp : StripeOp<"refine", [NoSideEffect]> {
  let summary = "Modify the offset into a tensor in an affine way";
  let arguments = (ins TensorRefType:$in, Variadic<AffineType>:$offsets);
  let results = (outs TensorRefType:$result);
}

def LoadOp : StripeOp<"load"> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorRefType:$from);
  let results = (outs Eltwise_AnyTensor:$into);
}

def LoadIndexOp : StripeOp<"load_index"> {
  let summary = "Convert an affine expression into a scalar value";
  let arguments = (ins AffineType:$from);
  let results = (outs Eltwise_AnyTensor:$into);
}

def StoreOp : StripeOp<"store"> {
  let summary = "Store into a tensor";
  let arguments = (ins TensorRefType:$into, Eltwise_AnyTensor:$from);
}

def AggTypeEnum : I64EnumAttr<
    "AggTypeEnum",
    "Operation to aggregate with",
    [
      I64EnumAttrCase<"add", 1>,
      I64EnumAttrCase<"mul", 2>,
      I64EnumAttrCase<"min", 3>,
      I64EnumAttrCase<"max", 4>,
    ]> {
}

def AggregateOp : StripeOp<"aggregate"> {
  let summary = "A accumulating store with a commutative, associative accumulation";
  let arguments = (ins TensorRefType:$into, Eltwise_AnyTensor:$from, AggTypeEnum:$agg_type);
  let results = (outs);
}

class AffineOp<string mnemonic> : StripeOp<mnemonic, [NoSideEffect]> {
  let results = (outs AffineType:$result);
}

def AffineConstOp : AffineOp<"affine_const"> {
  let summary = "A constant affine value";
  let arguments = (ins I64Attr:$value);
}

def AffineMulOp : AffineOp<"affine_mul"> {
  let summary = "Multiply an affine by a constant value";
  let arguments = (ins AffineType:$input, I64Attr:$scale);
}

def AffineAddOp : AffineOp<"affine_add"> {
  let summary = "Add multiple affines together";
  let arguments = (ins Variadic<AffineType>:$inputs);
}

def ParallelForOp : StripeOp<"parallel_for"> {
  let summary = "Parallel for loop";
  // At some point, we should be able to dynamically control ranges
  // Variadic<IntegerType>:$dyn_ranges
  let arguments = (ins I64ArrayAttr:$ranges);
  let regions = (region SizedRegion<1>:$inner);
}

def ConstraintOp : StripeOp<"constraint"> {
  let arguments = (ins AffineType:$input);
  let regions = (region AnyRegion:$ge_case, AnyRegion:$lt_case);
}

def SpecialOpInterface : OpInterface<"SpecialOp"> {
  let methods = [
    InterfaceMethod<"size_t", "getNumInputs">,
    InterfaceMethod<"size_t", "getNumOutputs">,
  ];
}

class SpecialOp<string mnemonic, int num_out, int num_in> :
    StripeOp<mnemonic, [SpecialOpInterface]> {
  let results = (outs);
  let extraClassDeclaration = 
    "  size_t getNumOutputs() { return " # num_out # "; }\n" #
    "  size_t getNumInputs() { return " # num_in # "; }\n";
}

def ReshapeOp : SpecialOp<"reshape", 1, 1> {
  let arguments = (ins
    TensorRefType:$into,
    TensorRefType:$from
  );
}

def ShapeOp : SpecialOp<"shape", 1, 1> {
  let arguments = (ins
    TensorRefType:$shape,
    TensorRefType:$tensor
  );
}

def GatherOp : SpecialOp<"gather", 1, 2> {
  let arguments = (ins
    TensorRefType:$out,
    TensorRefType:$idxs,
    TensorRefType:$table
  );
}

def ScatterOp : SpecialOp<"scatter", 1, 3> {
  let arguments = (ins
    TensorRefType:$out,
    TensorRefType:$expr,
    TensorRefType:$idxs,
    TensorRefType:$val
  );
}

def PrngStepOp : SpecialOp<"prng_step", 2, 1> {
  let arguments = (ins
    TensorRefType:$state_out, 
    TensorRefType:$buffer_out,
    TensorRefType:$state_in
  );
}

def TerminateOp : StripeOp<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
  let parser = [{ return mlir::success(); }];
  let printer = [{ *p << getOperationName(); }];
}

def ReturnOp : StripeOp<"return", [Terminator, HasParent<"ParallelForOp">]> {
  let summary = "Indicates where a ParallelForOp should execute";

  // 'from' represents the executor value being returned.  We're using a
  // Variadic here in lieu of Optional; a Stripe Return op may only have zero
  // or one argument/s.
  let arguments = (ins Variadic<TensorRefType>:$from);
}
