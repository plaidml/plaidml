// Copyright 2019, Intel Corporation

include "pmlc/dialect/scalar/predicates.td"

def StripeDialect : Dialect {
  let name = "stripe";
  let cppNamespace = "pmlc::dialect::stripe";
}

def AffineType : Type<CPred<"$_self.isa<AffineType>()">, "Plaid Affine Type">;
def TensorType : Type<CPred<"$_self.isa<TensorType>()">, "Plaid Tensor Reference Type">;
def PrngType : Type<CPred<"$_self.isa<PrngType>()">, "Plaid PrngType">;

def DictionaryAttr : Attr<CPred<"$_self.isa<mlir::DictionaryAttr>()">, "Dictionary Attribute"> {
  let storageType = "DictionaryAttr";
  let returnType = "DictionaryAttr";
  let convertFromStorage = "$_self";
  let defaultValue = "{}";
  let constBuilderCall = "$_builder.getDictionaryAttr($0)";
}

def TensorLayoutAttr : Attr<CPred<"$_self.isa<TensorLayoutAttr>()">, "Tensor Layout Attribute"> {
  let storageType = "TensorLayoutAttr";
  let returnType = "TensorLayoutAttr";
  let convertFromStorage = "$_self";
}

class StripeOp<string mnemonic, list<OpTrait> traits = []> :
    Op<StripeDialect, mnemonic, traits>;

def AllocateOp : StripeOp<"allocate"> {
  let summary = "Allocate a new tensor and set it's intial shape / size";
  let arguments = (ins TensorLayoutAttr:$layout);
  let results = (outs TensorType:$res);
}

def RefineOp : StripeOp<"refine", [NoSideEffect]> {
  let summary = "Modify the offset into a tensor in an affine way";
  let arguments = (ins TensorType:$in, Variadic<AffineType>:$offsets, DictionaryAttr:$attrs);
  let results = (outs TensorType:$result);
}

def LoadOp : StripeOp<"load"> {
  let summary = "Load from a tensor";
  let arguments = (ins TensorType:$from, DictionaryAttr:$attr);
  let results = (outs ScalarType:$into);
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
  let summary = "Store to a tensor";
  let arguments = (ins TensorType:$into, ScalarType:$from, DictionaryAttr:$attr);
  let results = (outs);
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

def ParallelForOp : StripeOp<"parallel_for"> {
  let summary = "Parallel for loop";
  // At some point, we should be able to dynamically control ranges
  // Variadic<IntegerType>:$dyn_ranges
  let arguments = (ins I64ArrayAttr:$ranges, DictionaryAttr:$attrs); //, TypedArrayAttrBase<DictionaryAttr, "per index tags">:$idx_attrs);
  let results = (outs);
  let regions = (region
    SizedRegion<1>:$inner
  );
}

def ConstraintOp : StripeOp<"constraint"> {
  let arguments = (ins AffineType:$input);
  let results = (outs);
  let regions = (region
    AnyRegion:$ge_case,
    AnyRegion:$lt_case
  );
}

def TerminateOp : StripeOp<"terminate", [Terminator]> {
  let summary = "A block terminator since MLIR requires one";
  let results = (outs);
  let arguments = (ins);
  let parser = [{ return mlir::success(); }];
  let printer = [{ *p << this->getOperation()->getName(); }];
}
