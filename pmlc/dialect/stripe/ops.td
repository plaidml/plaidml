// Copyright 2019, Intel Corporation

#ifndef __PML_STRIPE_OPS__
#define __PML_STRIPE_OPS__

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

#ifndef __PML_ELTWISE_PREDICATES__
include "pmlc/dialect/eltwise/predicates.td"
#endif

def StripeDialect : Dialect {
  let name = "stripe";
  let cppNamespace = "pmlc::dialect::stripe";
}

def AffineType : Type<CPred<"$_self.isa<AffineType>()">, "Plaid Affine Type">;
def TensorType : Type<CPred<"$_self.isa<TensorType>()">, "Plaid Tensor Type">;
def TensorRefType : Type<CPred<"$_self.isa<TensorRefType>()">, "Plaid Tensor Reference Type">;
def TensorTypeAttr : TypeAttrBase<"TensorType", "tensor type attribute">;

class StripeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StripeDialect, mnemonic, traits>;

class StripeOpWithPP<string mnemonic, list<OpTrait> traits = []> 
    : StripeOp<mnemonic, traits> {
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
}

def TerminateOp : StripeOpWithPP<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
}

def AllocateOp : StripeOpWithPP<"alloc"> {
  let summary = "Allocate a new tensor and set its intial shape / size";
  let arguments = (ins TensorTypeAttr:$layout);
  let results = (outs TensorRefType:$result);
  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, TensorType type">
  ];
}

def RefineOp : StripeOpWithPP<"refine", [NoSideEffect]> {
  let summary = "Modify the offset into a tensor in an affine way";
  let arguments = (ins TensorRefType:$in, Variadic<AffineType>:$offsets);
  let results = (outs TensorRefType:$result);
  let extraClassDeclaration = [{
    Value* getOffset(unsigned i) {
      // skip the $in operand
      return getOperand(i + 1);
    }
  }];
  let hasCanonicalizer = 1;
}

def LoadOp : StripeOpWithPP<"load", [NoSideEffect]> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorRefType:$from);
  let results = (outs EltwiseAny:$into);
}

def LoadIndexOp : StripeOp<"load_index"> {
  let summary = "Convert an affine expression into a scalar value";
  let arguments = (ins AffineType:$from);
  let results = (outs EltwiseAny:$into);
}

def StoreOp : StripeOp<"store"> {
  let summary = "Store into a tensor";
  let arguments = (ins TensorRefType:$into, EltwiseAny:$from);
}

def AggregateOp : StripeOpWithPP<"aggregate"> {
  let summary = "A accumulating store with a commutative, associative accumulation";
  let arguments = (ins TensorRefType:$into, EltwiseAny:$from, AggregationKind:$agg);
  let results = (outs);
}

def AffinePolyOp : StripeOpWithPP<"affine_poly", [NoSideEffect]> {
  let arguments = (ins Variadic<AffineType>:$inputs, I64ArrayAttr:$coeffs, I64Attr:$offset); 
  let results = (outs AffineType:$result);
  let builders = [OpBuilder<"Builder* builder, OperationState& result, const AffinePolynomial& poly">];
  let extraClassDeclaration = [{
    int64_t getCoeff(unsigned i) {
      return coeffs().getValue()[i].cast<IntegerAttr>().getInt();
    }
  }];
  let hasCanonicalizer = 1;
}

def ParallelForOp : StripeOpWithPP<"parallel_for"> {
  let summary = "Parallel for loop";
  // At some point, we should be able to dynamically control ranges
  // Variadic<IntegerType>:$dyn_ranges
  let arguments = (ins I64ArrayAttr:$ranges);
  let regions = (region SizedRegion<1>:$inner);
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, ArrayRef<int64_t> ranges">
  ];
  let extraClassDeclaration = [{
    int64_t getRange(unsigned i) {
      return ranges().getValue()[i].cast<IntegerAttr>().getInt();
    }
    Block *getBody() { return &inner().front(); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
  }];
  let hasCanonicalizer = 1;
}

def ConstraintOp : StripeOpWithPP<"constraint"> {
  let arguments = (ins AffineType:$input);
  let regions = (region AnyRegion:$ge_case, AnyRegion:$lt_case);
  let hasCanonicalizer = 1;
}

def SpecialOpInterface : OpInterface<"SpecialOp"> {
  let methods = [
    StaticInterfaceMethod<"TODO", "size_t", "getNumInputs">,
    StaticInterfaceMethod<"TODO", "size_t", "getNumOutputs">,
    StaticInterfaceMethod<"TODO", "Operation*", "create", (ins
      "OpBuilder*":$builder,
      "Location":$loc,
      "ArrayRef<Value*>":$inputs,
      "ArrayRef<Value*>":$outputs
    ), [{
      OperationState state(loc, ConcreteOp::getOperationName());
      state.addOperands(outputs);
      state.addOperands(inputs);
      return builder->createOperation(state);
    }]>,
  ];
}

class SpecialOp<string mnemonic, int num_out, int num_in> :
    StripeOp<mnemonic, [SpecialOpInterface]> {
  let results = (outs);
  let extraClassDeclaration = [{
    static size_t getNumOutputs() { return }] # num_out # [{;}
    static size_t getNumInputs() { return }] # num_in # [{;}
  }];
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

def ExecuteOnOp : StripeOpWithPP<"execute_on", [Terminator, HasParent<"ParallelForOp">]> {
  let summary = "Indicates where a ParallelForOp should execute";
  // 'from' represents the executor value being returned.
  let arguments = (ins TensorRefType:$from);
}

#endif // __PML_STRIPE_OPS__
