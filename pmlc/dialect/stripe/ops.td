// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/predicates.td"

def StripeDialect : Dialect {
  let name = "stripe";
  let cppNamespace = "pmlc::dialect::stripe";
}

def AffineType : Type<CPred<"$_self.isa<AffineType>()">, "Plaid Affine Type">;
def DeviceIDType : Type<CPred<"$_self.isa<DeviceIDType>()">, "Plaid DeviceIDType">;
def DevicePathType : Type<CPred<"$_self.isa<DevicePathType>()">, "Plaid DevicePathType">;
def PrngType : Type<CPred<"$_self.isa<PrngType>()">, "Plaid PrngType">;
def TensorType : Type<CPred<"$_self.isa<TensorType>()">, "Plaid Tensor Type">;
def TensorRefType : Type<CPred<"$_self.isa<TensorRefType>()">, "Plaid Tensor Reference Type">;

class StripeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StripeDialect, mnemonic, traits>;

def AllocateOp : StripeOp<"alloc"> {
  let summary = "Allocate a new tensor and set it's intial shape / size";
  let arguments = (ins DevicePathType:$devicePath);
  let results = (outs TensorType:$result);
}

def TensorRefOp : StripeOp<"ref", [NoSideEffect]> {
  let summary = "Create a default tensor_ref from a tensor";
  let arguments = (ins TensorType:$in, DevicePathType:$devicePath);
  let results = (outs TensorRefType:$result);
}

def RefineOp : StripeOp<"refine", [NoSideEffect]> {
  let summary = "Modify the offset into a tensor in an affine way";
  let arguments = (ins
    TensorRefType:$in,
    Variadic<AffineType>:$offsets,
    DictionaryAttr:$attrs,
    DevicePathType:$devicePath
  );
  let results = (outs TensorRefType:$result);
}

def LoadOp : StripeOp<"load"> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorRefType:$from, DictionaryAttr:$attr);
  let results = (outs Eltwise_AnyTensor:$into);
  /*
  let parser = [{
    mlir::SmallVector<OpAsmParser::OperandType, 1> ops;
    Type type;
    return mlir::failure(parser->parseOperandList(ops, 1) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperands(ops, type, result->operands) ||
                 parser->addTypeToList(type, result->types));
  }];
  let printer = [{
    auto op = this->getOperation();
    *p << op->getName() << ' ' << *op->getOperand(0);
    *p << " : " << op->getResult(0)->getType();
  }];
  */
}

def StoreOp : StripeOp<"store"> {
  let summary = "Store into a tensor";
  let arguments = (ins
    TensorRefType:$into,
    Eltwise_AnyTensor:$from,
    DictionaryAttr:$attr
  );
  /*
  let parser = [{
    mlir::SmallVector<OpAsmParser::OperandType, 2> ops;
    Type type;
    return mlir::failure(parser->parseOperandList(ops, 2) ||
                 parser->resolveOperands(ops, type, result->operands));
  }];
  let printer = [{
    auto op = this->getOperation();
    *p << op->getName() << ' ' << *op->getOperand(0) << ", " << *op->getOperand(1);
  }];
  */
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
  let arguments = (ins TensorRefType:$into, Eltwise_AnyTensor:$from, AggTypeEnum:$agg_type, DictionaryAttr:$attr);
  let results = (outs);
}

class AffineOp<string mnemonic> : StripeOp<mnemonic, [NoSideEffect]> {
  let results = (outs AffineType:$result);
}

def AffineConstOp : AffineOp<"affine_const"> {
  let summary = "A constant affine value";
  let arguments = (ins I64Attr:$value);
  /*
  let printer = [{
    auto op = this->getOperation();
    *p << op->getName() << ' ';
    *p << op->getAttrOfType<IntegerAttr>("value").getInt();
  }];
  */
}

def AffineMulOp : AffineOp<"affine_mul"> {
  let summary = "Multiply an affine by a constant value";
  let arguments = (ins AffineType:$input, I64Attr:$scale);
}

def AffineAddOp : AffineOp<"affine_add"> {
  let summary = "Add multiple affines together";
  let arguments = (ins Variadic<AffineType>:$inputs);
}

def AffineMeta : AffineOp<"affine_meta"> {
  let summary = "Attach metadata to an affine";
  let arguments = (ins AffineType:$input, DictionaryAttr:$attrs);
}

def DeviceIDOp : StripeOp<"devid", [NoSideEffect]> {
  let arguments = (ins StrAttr:$name, Variadic<AffineType>:$unit);
  let results = (outs DeviceIDType:$result);
}

def DevicePathOp : StripeOp<"devpath", [NoSideEffect]> {
  let arguments = (ins Variadic<DeviceIDType>:$dev_ids);
  let results = (outs DevicePathType:$result);
}

def ParallelForOp : StripeOp<"parallel_for"> {
  let summary = "Parallel for loop";
  // At some point, we should be able to dynamically control ranges
  // Variadic<IntegerType>:$dyn_ranges
  let arguments = (ins
    I64ArrayAttr:$ranges,
    DictionaryAttr:$attrs
    // TypedArrayAttrBase<DictionaryAttr, "per index tags">:$idx_attrs
  ); 
  let regions = (region SizedRegion<1>:$inner);
}

def ConstraintOp : StripeOp<"constraint"> {
  let arguments = (ins AffineType:$input);
  let regions = (region AnyRegion:$ge_case, AnyRegion:$lt_case);
}

def ExecutorOp : StripeOp<"executor"> {
  let arguments = (ins DevicePathType:$devicePath);
  let regions = (region AnyRegion:$body);
}

def TerminateOp : StripeOp<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
  let parser = [{ return mlir::success(); }];
  let printer = [{ *p << this->getOperation()->getName(); }];
}

