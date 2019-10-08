// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/enums.td"
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
  let printer = [{ PrintSimple(getOperation(), p, 1, {}, in()->getType(), true); }];
  let parser = [{
    TensorRefType refType;
    auto aff_type = AffineType::get(parser->getBuilder().getContext());
    llvm::SmallVector<OpAsmParser::OperandType, 8> operands;
    bool r = ParseSimple(parser, result, &operands, 1, {}, &refType, true);
    r = r || parser->resolveOperand(operands[0], refType, result->operands);
    for (size_t i = 1; i < operands.size(); i++) {
      r = r || parser->resolveOperand(operands[i], aff_type, result->operands);
    }
    r = r || parser->addTypeToList(refType, result->types);
    return mlir::failure(r); 
  }];
}

def LoadOp : StripeOp<"load"> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorRefType:$from);
  let results = (outs Eltwise_AnyTensor:$into);
  let printer = [{ PrintSimple(getOperation(), p, 1, {}, from()->getType(), false); }];
  let parser = [{
    TensorRefType refType;
    llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
    bool r = ParseSimple(parser, result, &operands, 1, {}, &refType, false);
    Type eltType = RankedTensorType::get({}, refType.getElementType());
    r = r || parser->resolveOperand(operands[0], refType, result->operands);
    r = r || parser->addTypeToList(eltType, result->types);
    return mlir::failure(r); 
  }];
}

def LoadIndexOp : StripeOp<"load_index"> {
  let summary = "Convert an affine expression into a scalar value";
  let arguments = (ins AffineType:$from);
  let results = (outs Eltwise_AnyTensor:$into);
}

def StoreOp : StripeOp<"store"> {
  let summary = "Store into a tensor";
  let arguments = (ins TensorRefType:$into, Eltwise_AnyTensor:$from);
  let printer = [{ PrintSimple(getOperation(), p, 2, {}, into()->getType(), false); }];
  let parser = [{
    TensorRefType refType;
    llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
    bool r = ParseSimple(parser, result, &operands, 2, {}, &refType, false);
    Type eltType = RankedTensorType::get({}, refType.getElementType());
    r = r || parser->resolveOperand(operands[0], refType, result->operands);
    r = r || parser->resolveOperand(operands[1], eltType, result->operands);
    return mlir::failure(r); 
  }];
}

def AggregateOp : StripeOp<"aggregate"> {
  let summary = "A accumulating store with a commutative, associative accumulation";
  let arguments = (ins TensorRefType:$into, Eltwise_AnyTensor:$from, AggregationKind:$agg);
  let results = (outs);
  let printer = [{ PrintSimple(getOperation(), p, 2, {}, into()->getType(), false); }];
  let parser = [{
    TensorRefType refType;
    llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
    bool r = ParseSimple(parser, result, &operands, 2, {}, &refType, false);
    Type eltType = RankedTensorType::get({}, refType.getElementType());
    r = r || parser->resolveOperand(operands[0], refType, result->operands);
    r = r || parser->resolveOperand(operands[1], eltType, result->operands);
    return mlir::failure(r); 
  }];
}

class AffineOp<string mnemonic> : StripeOp<mnemonic, [NoSideEffect]> {
  let results = (outs AffineType:$result);
}

def AffineConstOp : AffineOp<"affine_const"> {
  let summary = "A constant affine value";
  let arguments = (ins I64Attr:$value);
  let printer = [{ PrintSimple(getOperation(), p, 0, {"value"}, Type(), false); }];
  let parser = [{
    auto aff_type = AffineType::get(parser->getBuilder().getContext());
    llvm::SmallVector<OpAsmParser::OperandType, 0> operands;
    bool r = ParseSimple(parser, result, &operands, 0, {"value"}, static_cast<Type*>(nullptr), false);
    r = r || parser->addTypeToList(aff_type, result->types);
    return mlir::failure(r); 
  }];
}

def AffineMulOp : AffineOp<"affine_mul"> {
  let summary = "Multiply an affine by a constant value";
  let arguments = (ins AffineType:$input, I64Attr:$scale);
  let printer = [{ PrintSimple(getOperation(), p, 1, {"scale"}, Type(), false); }];
  let parser = [{
    auto aff_type = AffineType::get(parser->getBuilder().getContext());
    llvm::SmallVector<OpAsmParser::OperandType, 1> operands;
    bool r = ParseSimple(parser, result, &operands, 1, {"scale"}, static_cast<Type*>(nullptr), false);
    r = r || parser->resolveOperand(operands[0], aff_type, result->operands);
    r = r || parser->addTypeToList(aff_type, result->types);
    return mlir::failure(r); 
  }];
}

def AffineAddOp : AffineOp<"affine_add"> {
  let summary = "Add multiple affines together";
  let arguments = (ins Variadic<AffineType>:$inputs); 
  let printer = [{ PrintSimple(getOperation(), p, 0, {}, Type(), true); }];
  let parser = [{
    auto aff_type = AffineType::get(parser->getBuilder().getContext());
    llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
    bool r = ParseSimple(parser, result, &operands, 0, {}, static_cast<Type*>(nullptr), true);
    r = r || parser->resolveOperands(operands, aff_type, result->operands);
    r = r || parser->addTypeToList(aff_type, result->types);
    return mlir::failure(r); 
  }];
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
    InterfaceMethod<"TODO", "size_t", "getNumInputs">,
    InterfaceMethod<"TODO", "size_t", "getNumOutputs">,
  ];
}

class SpecialOp<string mnemonic, int num_out, int num_in> :
    StripeOp<mnemonic, [SpecialOpInterface]> {
  let results = (outs);
  let extraClassDeclaration = [{
    size_t getNumOutputs() { return }] # num_out # [{;}
    size_t getNumInputs() { return }] # num_in # [{;}
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

def TerminateOp : StripeOp<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
  let parser = [{ return mlir::success(); }];
  let printer = [{ p << getOperationName(); }];
}

def ExecuteOnOp : StripeOp<"execute_on", [Terminator, HasParent<"ParallelForOp">]> {
  let summary = "Indicates where a ParallelForOp should execute";

  // 'from' represents the executor value being returned.
  let arguments = (ins TensorRefType:$from);
}
