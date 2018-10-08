// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/emitllvm.h"

#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "tile/lang/exprtype.h"
#include "tile/lang/fnv1a64.h"
#include "tile/lang/generate.h"
#include "tile/lang/semprinter.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

Emit::Emit()
    : context_(llvm::getGlobalContext()),
      builder_{context_},
      module_{new llvm::Module("tile", context_)},
      funcopt_{new llvm::legacy::FunctionPassManager(module_.get())},
      int32type_{llvm::IntegerType::get(context_, 32)},
      booltype_{llvm::IntegerType::get(context_, 1)},
      blocks_{1} {
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  ssizetype_ = llvm::IntegerType::get(context_, archbits);
  auto gridSizeCount = std::tuple_size<lang::GridSize>::value;
  gridSizeType_ = llvm::ArrayType::get(ssizetype_, gridSizeCount);

  // Generate an external reference for the barrier function.
  std::vector<llvm::Type*> no_args;
  llvm::Type* voidType = llvm::Type::getVoidTy(context_);
  auto functype = llvm::FunctionType::get(voidType, no_args, false);
  auto link = llvm::Function::ExternalLinkage;
  std::string name = "Barrier";
  barrier_func_ = llvm::Function::Create(functype, link, name, module_.get());

  // Configure the function pass manager for specific optimization passes which
  // might be relevant for Tile code.
  funcopt_->add(llvm::createEarlyCSEPass());
  funcopt_->add(llvm::createLICMPass());
  funcopt_->add(llvm::createLoopInstSimplifyPass());
  funcopt_->add(llvm::createGVNPass());
  funcopt_->add(llvm::createDeadStoreEliminationPass());
  funcopt_->add(llvm::createSCCPPass());
  funcopt_->add(llvm::createReassociatePass());
  funcopt_->add(llvm::createInstructionCombiningPass());
  funcopt_->add(llvm::createInstructionSimplifierPass());
  funcopt_->add(llvm::createAggressiveDCEPass());
  funcopt_->add(llvm::createCFGSimplificationPass());
  funcopt_->doInitialization();
}

void Emit::Visit(const sem::IntConst& n) {
  llvm::Value* ret = llvm::ConstantInt::get(int32type_, n.value);
  Resolve(value{ret, {sem::Type::VALUE, DataType::INT32}});
}

void Emit::Visit(const sem::FloatConst& n) {
  llvm::Type* ty = llvm::Type::getFloatTy(context_);
  llvm::Value* ret = llvm::ConstantFP::get(ty, n.value);
  Resolve(value{ret, {sem::Type::VALUE, DataType::FLOAT32}});
}

void Emit::Visit(const sem::LookupLVal& n) {
  value ret = Lookup(n.name);
  Resolve(ret);
}

void Emit::Visit(const sem::LoadExpr& n) {
  value ptr = LVal(n.inner);
  llvm::Value* ret = builder_.CreateLoad(ptr.v);
  Resolve(value{ret, ptr.t});
}

void Emit::Visit(const sem::StoreStmt& n) {
  value lhs = LVal(n.lhs);
  assert(lhs.v->getType()->isPointerTy());
  value rhs = Eval(n.rhs);
  llvm::Value* rval = CastTo(rhs, lhs.t);
  builder_.CreateStore(rval, lhs.v);
}

void Emit::Visit(const sem::SubscriptLVal& n) {
  // Compute the address of an array element, given the array's base address
  // and the element offset number.
  value ptr = LVal(n.ptr);
  std::vector<llvm::Value*> indexes;
  llvm::Type* element = ptr.v->getType()->getPointerElementType();
  // A variable holding the address of an external array will have a pointer
  // to pointer type; we must dereference it to get the base address.
  if (element->isPointerTy()) {
    ptr.v = builder_.CreateLoad(ptr.v);
  }
  // If the array was defined inside Tile, our variable is a pointer to the
  // actual storage array, so we can compute the element address in a single
  // operation by first "indexing" the pointer with offset zero. Yes, it's
  // weird, but it's a standard LLVM-ism.
  if (element->isArrayTy()) {
    indexes.push_back(llvm::ConstantInt::get(int32type_, 0));
  }
  // Since Index is itself an LVal, our result is not the element value, but
  // a pointer to the element value, which just happens to be the same thing
  // GetElementPtr wants to return.
  value offset = Eval(n.offset);
  indexes.push_back(CastTo(offset, {sem::Type::VALUE, DataType::INT32}));
  llvm::Value* resptr = builder_.CreateGEP(ptr.v, indexes);
  sem::Type restype = ptr.t;
  restype.base = sem::Type::VALUE;
  restype.array = 0;
  Resolve(value{resptr, restype});
}

void Emit::Visit(const sem::DeclareStmt& n) {
  llvm::Value* lhs = Define(n.name, n.type);
  if (n.init) {
    // The declaration provides a single initialization value. If this is an
    // array, we'll assign that value to every element.
    value rhs = Eval(n.init);
    if (n.type.array) {
      sem::Type element_type = n.type;
      element_type.array = 0;
      llvm::Value* val = CastTo(rhs, element_type);
      // The var is a pointer to an array, so we need to "index" the pointer
      // with an offset of zero before indexing the element we actually want.
      llvm::Value* zero = llvm::ConstantInt::get(int32type_, 0);
      for (uint64_t i = 0; i < n.type.array; ++i) {
        llvm::Value* index = llvm::ConstantInt::get(ssizetype_, i);
        llvm::Value* element = builder_.CreateGEP(lhs, {zero, index});
        builder_.CreateStore(val, element);
      }
    } else {
      llvm::Value* val = CastTo(rhs, n.type);
      builder_.CreateStore(val, lhs);
    }
  }
}

void Emit::Visit(const sem::UnaryExpr& n) {
  value inner = Eval(n.inner);
  using fnv1a64::hashlit;
  switch (fnv1a64::hash(n.op.c_str())) {
    case hashlit("!"):
      Resolve(value{builder_.CreateNot(inner.v), inner.t});
      return;
    case hashlit("-"):
      if (IsFloatingPointType(inner.t)) {
        Resolve(value{builder_.CreateFNeg(inner.v), inner.t});
      } else {
        Resolve(value{builder_.CreateNeg(inner.v), inner.t});
      }
      return;
  }
  throw Error("Invalid unary op: " + n.op + print(inner.v->getType()));
}

void Emit::Visit(const sem::BinaryExpr& n) {
  value lhs = Eval(n.lhs);
  value rhs = Eval(n.rhs);

  // Pointer offsets are encoded as an addition, but have little to do with the
  // arithmetic sort of addition, so we'll handle them as a special case.
  if (n.op == "+" && PointerAddition(lhs, rhs)) return;

  sem::Type comtype = ConvergeOperands(&lhs, &rhs);
  sem::Type booltype{sem::Type::VALUE, DataType::BOOLEAN};

  // Create the instruction that represents this operator, which depends on
  // whether our operands are floats, signed ints, or unsigned ints.
  uint64_t ophash = fnv1a64::hash(n.op.c_str());
  using fnv1a64::hashlit;
  if (IsFloatingPointType(comtype)) {
    switch (ophash) {
      case hashlit("+"):
        Resolve(value{builder_.CreateFAdd(lhs.v, rhs.v), comtype});
        return;
      case hashlit("-"):
        Resolve(value{builder_.CreateFSub(lhs.v, rhs.v), comtype});
        return;
      case hashlit("*"):
        Resolve(value{builder_.CreateFMul(lhs.v, rhs.v), comtype});
        return;
      case hashlit("/"):
        Resolve(value{builder_.CreateFDiv(lhs.v, rhs.v), comtype});
        return;
      case hashlit("=="):
        Resolve(value{builder_.CreateFCmpOEQ(lhs.v, rhs.v), booltype});
        return;
      case hashlit("!="):
        Resolve(value{builder_.CreateFCmpONE(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<"):
        Resolve(value{builder_.CreateFCmpOLT(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">"):
        Resolve(value{builder_.CreateFCmpOGT(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<="):
        Resolve(value{builder_.CreateFCmpOLE(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">="):
        Resolve(value{builder_.CreateFCmpOGE(lhs.v, rhs.v), booltype});
        return;
    }
  } else if (IsUnsignedIntegerType(comtype)) {
    switch (ophash) {
      case hashlit("+"):
        Resolve(value{builder_.CreateAdd(lhs.v, rhs.v), comtype});
        return;
      case hashlit("-"):
        Resolve(value{builder_.CreateSub(lhs.v, rhs.v), comtype});
        return;
      case hashlit("*"):
        Resolve(value{builder_.CreateMul(lhs.v, rhs.v), comtype});
        return;
      case hashlit("/"):
        Resolve(value{builder_.CreateUDiv(lhs.v, rhs.v), comtype});
        return;
      case hashlit("%"):
        Resolve(value{builder_.CreateURem(lhs.v, rhs.v), comtype});
        return;
      case hashlit("<<"):
        Resolve(value{builder_.CreateShl(lhs.v, rhs.v), comtype});
        return;
      case hashlit(">>"):
        Resolve(value{builder_.CreateLShr(lhs.v, rhs.v), comtype});
        return;
      case hashlit("=="):
        Resolve(value{builder_.CreateICmpEQ(lhs.v, rhs.v), booltype});
        return;
      case hashlit("!="):
        Resolve(value{builder_.CreateICmpNE(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<"):
        Resolve(value{builder_.CreateICmpULT(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">"):
        Resolve(value{builder_.CreateICmpUGT(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<="):
        Resolve(value{builder_.CreateICmpULE(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">="):
        Resolve(value{builder_.CreateICmpUGE(lhs.v, rhs.v), booltype});
        return;
      case hashlit("&"):
        Resolve(value{builder_.CreateAnd(lhs.v, rhs.v), comtype});
        return;
      case hashlit("|"):
        Resolve(value{builder_.CreateOr(lhs.v, rhs.v), comtype});
        return;
      case hashlit("^"):
        Resolve(value{builder_.CreateXor(lhs.v, rhs.v), comtype});
        return;
      case hashlit("&&"):
        Resolve(value{builder_.CreateAnd(ToBool(lhs.v), ToBool(rhs.v)), booltype});
        return;
      case hashlit("||"):
        Resolve(value{builder_.CreateOr(ToBool(lhs.v), ToBool(rhs.v)), booltype});
        return;
    }
  } else {
    switch (ophash) {
      case hashlit("+"):
        Resolve(value{builder_.CreateAdd(lhs.v, rhs.v), comtype});
        return;
      case hashlit("-"):
        Resolve(value{builder_.CreateSub(lhs.v, rhs.v), comtype});
        return;
      case hashlit("*"):
        Resolve(value{builder_.CreateMul(lhs.v, rhs.v), comtype});
        return;
      case hashlit("/"):
        Resolve(value{builder_.CreateSDiv(lhs.v, rhs.v), comtype});
        return;
      case hashlit("%"):
        Resolve(value{builder_.CreateSRem(lhs.v, rhs.v), comtype});
        return;
      case hashlit("<<"):
        Resolve(value{builder_.CreateShl(lhs.v, rhs.v), comtype});
        return;
      case hashlit(">>"):
        Resolve(value{builder_.CreateAShr(lhs.v, rhs.v), comtype});
        return;
      case hashlit("=="):
        Resolve(value{builder_.CreateICmpEQ(lhs.v, rhs.v), booltype});
        return;
      case hashlit("!="):
        Resolve(value{builder_.CreateICmpNE(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<"):
        Resolve(value{builder_.CreateICmpSLT(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">"):
        Resolve(value{builder_.CreateICmpSGT(lhs.v, rhs.v), booltype});
        return;
      case hashlit("<="):
        Resolve(value{builder_.CreateICmpSLE(lhs.v, rhs.v), booltype});
        return;
      case hashlit(">="):
        Resolve(value{builder_.CreateICmpSGE(lhs.v, rhs.v), booltype});
        return;
      case hashlit("&"):
        Resolve(value{builder_.CreateAnd(lhs.v, rhs.v), comtype});
        return;
      case hashlit("|"):
        Resolve(value{builder_.CreateOr(lhs.v, rhs.v), comtype});
        return;
      case hashlit("^"):
        Resolve(value{builder_.CreateXor(lhs.v, rhs.v), comtype});
        return;
      case hashlit("&&"):
        Resolve(value{builder_.CreateAnd(ToBool(lhs.v), ToBool(rhs.v)), booltype});
        return;
      case hashlit("||"):
        Resolve(value{builder_.CreateOr(ToBool(lhs.v), ToBool(rhs.v)), booltype});
        return;
    }
  }
  std::string desc = print(lhs.v->getType()) + n.op + print(rhs.v->getType());
  throw Error("Invalid operation: " + desc);
}

void Emit::Visit(const sem::CondExpr& n) {
  // Tile semantics allow but do not require conditional evaluation of the
  // result operands in a CondExpr. We'll use conditional evaluation for single
  // booleans, but boolean vectors will be implemented as a select.
  value cond = Eval(n.cond);
  llvm::Type* condtype = cond.v->getType();
  if (condtype->isVectorTy()) {
    // Vector of booleans: operands must also be vectors of equal size.
    llvm::Value* selexpr = ToBool(cond.v);
    value tcase = Eval(n.tcase);
    value fcase = Eval(n.fcase);
    sem::Type comtype = ConvergeOperands(&tcase, &fcase);
    Resolve(value{builder_.CreateSelect(selexpr, tcase.v, fcase.v), comtype});
  } else {
    // Scalar boolean: evaluate "then" or "else" expression accordingly.
    auto thenblock = llvm::BasicBlock::Create(context_, "then");
    auto elseblock = llvm::BasicBlock::Create(context_, "else");
    auto nextblock = llvm::BasicBlock::Create(context_, "next");
    builder_.CreateCondBr(ToBool(cond.v), thenblock, elseblock);

    // Evaluate the value to return if the condition is true.
    function_->getBasicBlockList().push_back(thenblock);
    builder_.SetInsertPoint(thenblock);
    value thenval = Eval(n.tcase);
    thenblock = builder_.GetInsertBlock();

    // Evaluate the value to return if the condition is false.
    function_->getBasicBlockList().push_back(elseblock);
    builder_.SetInsertPoint(elseblock);
    value elseval = Eval(n.fcase);
    elseblock = builder_.GetInsertBlock();

    // What common type can we coerce both operand values into?
    sem::Type comtype = lang::Promote({thenval.t, elseval.t});

    // Convert the true value to the common type and exit.
    builder_.SetInsertPoint(thenblock);
    thenval = value{CastTo(thenval, comtype), comtype};
    builder_.CreateBr(nextblock);
    thenblock = builder_.GetInsertBlock();

    // Convert the false value to the common type and exit.
    builder_.SetInsertPoint(elseblock);
    elseval = value{CastTo(elseval, comtype), comtype};
    builder_.CreateBr(nextblock);
    elseblock = builder_.GetInsertBlock();

    // Merge both instruction streams back together.
    function_->getBasicBlockList().push_back(nextblock);
    builder_.SetInsertPoint(nextblock);
    llvm::PHINode* phi = builder_.CreatePHI(CType(comtype), 2, "cond");
    phi->addIncoming(thenval.v, thenblock);
    phi->addIncoming(elseval.v, elseblock);
    Resolve(value{phi, thenval.t});
  }
}

void Emit::Visit(const sem::SelectExpr& n) {
  value cond = Eval(n.cond);
  value tcase = Eval(n.tcase);
  value fcase = Eval(n.fcase);
  sem::Type comtype = ConvergeOperands(&tcase, &fcase);
  Resolve(value{builder_.CreateSelect(ToBool(cond.v), tcase.v, fcase.v), comtype});
}

void Emit::Visit(const sem::ClampExpr& n) {
  value val = Eval(n.val);
  value min = value{CastTo(Eval(n.min), val.t), val.t};
  value max = value{CastTo(Eval(n.max), val.t), val.t};
  llvm::Value* infra = nullptr;
  llvm::Value* supra = nullptr;
  if (IsFloatingPointType(val.t)) {
    infra = builder_.CreateFCmpOLT(val.v, min.v);
    supra = builder_.CreateFCmpOGT(val.v, max.v);
  } else if (IsUnsignedIntegerType(val.t)) {
    infra = builder_.CreateICmpULT(val.v, min.v);
    supra = builder_.CreateICmpUGT(val.v, max.v);
  } else {
    infra = builder_.CreateICmpSLT(val.v, min.v);
    supra = builder_.CreateICmpSGT(val.v, max.v);
  }
  llvm::Value* ret = builder_.CreateSelect(infra, min.v, val.v);
  ret = builder_.CreateSelect(supra, max.v, ret);
  Resolve(value{ret, val.t});
}

void Emit::Visit(const sem::CastExpr& n) {
  value val = Eval(n.val);
  // Broadcasting a singular value to a vector is legal when explicit, but
  // must be implemented here and not in CastTo() because it is not allowed
  // as an implicit conversion.
  if (val.t.vec_width == 1 && n.type.vec_width > 1) {
    sem::Type nonvec = n.type;
    nonvec.vec_width = 1;
    llvm::Value* v = CastTo(val, nonvec);
    Resolve(value{builder_.CreateVectorSplat(n.type.vec_width, v), n.type});
  } else {
    Resolve(value{CastTo(val, n.type), n.type});
  }
}

void Emit::Visit(const sem::CallExpr& n) {
  // Apply one of the builtin functions to the argument(s). All arguments must
  // have the same type, which is 'double' by default, and the result will have
  // the same type. If one or more of the arguments are vectors, we will convert
  // non-vector arguments up to the vector size.
  sem::Type gentype{sem::Type::VALUE, DataType::FLOAT64};
  std::string typefix = ".";
  std::vector<value> vals;
  for (auto& expr : n.vals) {
    auto val = Eval(expr);
    if (val.t.vec_width > gentype.vec_width) {
      gentype.vec_width = val.t.vec_width;
      typefix = ".v" + std::to_string(gentype.vec_width);
    }
    vals.push_back(val);
  }
  typefix += "f64";
  // Cast each argument value to the common type used for this call. This may
  // require vector-expansion.
  std::vector<llvm::Value*> args;
  for (auto& val : vals) {
    args.push_back(CastTo(val, gentype));
  }
  std::string linkName;
  bool devectorize = false;
  switch (n.function) {
    case sem::CallExpr::Function::CEIL:
    case sem::CallExpr::Function::COS:
    case sem::CallExpr::Function::EXP:
    case sem::CallExpr::Function::FLOOR:
    case sem::CallExpr::Function::LOG:
    case sem::CallExpr::Function::POW:
    case sem::CallExpr::Function::SIN:
    case sem::CallExpr::Function::SQRT:
      linkName = "llvm." + n.name + typefix;
      break;
    case sem::CallExpr::Function::MAD:
      linkName = "llvm.fma" + typefix;
      break;
    case sem::CallExpr::Function::ROUND:
      linkName = (gentype.vec_width > 1) ? ("llvm.rint" + typefix) : "round";
      break;
    case sem::CallExpr::Function::ACOS:
    case sem::CallExpr::Function::ASIN:
    case sem::CallExpr::Function::ATAN:
    case sem::CallExpr::Function::COSH:
    case sem::CallExpr::Function::SINH:
    case sem::CallExpr::Function::TAN:
    case sem::CallExpr::Function::TANH:
      linkName = n.name;
      devectorize = true;
      break;
  }
  // Find a reference to that builtin function, or generate a reference if this
  // is the first time we've called it.
  llvm::Function* func = nullptr;
  auto loc = builtins_.find(linkName);
  if (loc != builtins_.end()) {
    func = loc->second;
  } else {
    sem::Type restype = gentype;
    if (devectorize) {
      restype.vec_width = 1;
    }
    llvm::Type* ltype = CType(restype);
    std::vector<llvm::Type*> paramTypes(args.size(), ltype);
    auto funcType = llvm::FunctionType::get(ltype, paramTypes, false);
    auto link = llvm::Function::ExternalLinkage;
    func = llvm::Function::Create(funcType, link, linkName, module_.get());
    builtins_[linkName] = func;
  }
  // Generate an invocation, which will produce the same output type.
  if (gentype.vec_width == 1 || !devectorize) {
    Resolve(value{builder_.CreateCall(func, args, ""), gentype});
  } else {
    assert(1 == args.size());
    // Apply the target function to each vector element.
    auto apzero = llvm::APFloat::getZero(llvm::APFloat::IEEEdouble);
    auto zero = llvm::ConstantFP::get(context_, apzero);
    auto ltype = CType(gentype);
    auto op = llvm::CastInst::getCastOpcode(zero, true, ltype, true);
    llvm::Value* outvec = builder_.CreateCast(op, zero, ltype);
    for (unsigned i = 0; i < gentype.vec_width; ++i) {
      llvm::Value* index = llvm::ConstantInt::get(int32type_, i);
      llvm::Value* inelement = builder_.CreateExtractElement(args[0], index);
      std::vector<llvm::Value*> callarg(1, inelement);
      llvm::Value* outelement = builder_.CreateCall(func, callarg, "");
      outvec = builder_.CreateInsertElement(outvec, outelement, index);
    }
    Resolve(value{outvec, gentype});
  }
}

void Emit::LimitConstSInt(unsigned bits, sem::LimitConst::Which which) {
  llvm::APInt apval;
  switch (which) {
    case sem::LimitConst::ZERO:
      apval = llvm::APInt::getNullValue(bits);
      break;
    case sem::LimitConst::ONE:
      apval = llvm::APInt(bits, 1);
      break;
    case sem::LimitConst::MIN:
      apval = llvm::APInt::getSignedMinValue(bits);
      break;
    case sem::LimitConst::MAX:
      apval = llvm::APInt::getSignedMaxValue(bits);
      break;
  }
  llvm::IntegerType* ty = llvm::IntegerType::get(context_, bits);
  sem::Type comtype{sem::Type::VALUE};
  if (bits <= 8) {
    comtype.dtype = DataType::INT8;
  } else if (bits <= 16) {
    comtype.dtype = DataType::INT16;
  } else if (bits <= 32) {
    comtype.dtype = DataType::INT32;
  } else {
    comtype.dtype = DataType::INT64;
  }
  Resolve(value{llvm::ConstantInt::get(ty, apval), comtype});
}

void Emit::LimitConstUInt(unsigned bits, sem::LimitConst::Which which) {
  llvm::APInt apval;
  switch (which) {
    case sem::LimitConst::ZERO:
      apval = llvm::APInt::getNullValue(bits);
      break;
    case sem::LimitConst::ONE:
      apval = llvm::APInt(bits, 1);
      break;
    case sem::LimitConst::MIN:
      apval = llvm::APInt::getMinValue(bits);
      break;
    case sem::LimitConst::MAX:
      apval = llvm::APInt::getMaxValue(bits);
      break;
  }
  llvm::IntegerType* ty = llvm::IntegerType::get(context_, bits);
  sem::Type comtype{sem::Type::VALUE};
  if (bits == 1) {
    comtype.dtype = DataType::BOOLEAN;
  } else if (bits <= 8) {
    comtype.dtype = DataType::UINT8;
  } else if (bits <= 16) {
    comtype.dtype = DataType::UINT16;
  } else if (bits <= 32) {
    comtype.dtype = DataType::UINT32;
  } else {
    comtype.dtype = DataType::UINT64;
  }
  Resolve(value{llvm::ConstantInt::get(ty, apval), comtype});
}

void Emit::LimitConstFP(const llvm::fltSemantics& sem, sem::LimitConst::Which which) {
  llvm::APFloat apval(llvm::APFloat::Bogus);
  switch (which) {
    case sem::LimitConst::ZERO:
      apval = llvm::APFloat::getZero(sem);
      break;
    case sem::LimitConst::ONE:
      apval = llvm::APFloat(1.0);
      break;
    case sem::LimitConst::MIN:
      apval = llvm::APFloat::getLargest(sem, /*Negative*/ true);
      break;
    case sem::LimitConst::MAX:
      apval = llvm::APFloat::getLargest(sem, /*Negative*/ false);
      break;
  }
  sem::Type comtype{sem::Type::VALUE, DataType::FLOAT64};
  Resolve(value{llvm::ConstantFP::get(context_, apval), comtype});
}

void Emit::Visit(const sem::LimitConst& n) {
  switch (n.type) {
    case DataType::BOOLEAN:
      LimitConstUInt(1, n.which);
      break;
    case DataType::INT8:
      LimitConstSInt(8, n.which);
      break;
    case DataType::INT16:
      LimitConstSInt(16, n.which);
      break;
    case DataType::INT32:
      LimitConstSInt(32, n.which);
      break;
    case DataType::INT64:
      LimitConstSInt(64, n.which);
      break;
    case DataType::UINT8:
      LimitConstUInt(8, n.which);
      break;
    case DataType::UINT16:
      LimitConstUInt(16, n.which);
      break;
    case DataType::UINT32:
      LimitConstUInt(32, n.which);
      break;
    case DataType::UINT64:
      LimitConstUInt(64, n.which);
      break;
    case DataType::FLOAT16:
      LimitConstFP(llvm::APFloat::IEEEhalf, n.which);
      break;
    case DataType::FLOAT32:
      LimitConstFP(llvm::APFloat::IEEEsingle, n.which);
      break;
    case DataType::FLOAT64:
      LimitConstFP(llvm::APFloat::IEEEdouble, n.which);
      break;
    case DataType::INVALID:
    case DataType::PRNG:
      throw Error("Unknown type has no constants");
  }
}

void Emit::Visit(const sem::IndexExpr& n) {
  // The local index is always zero on the CPU, since we don't have anything
  // analogous to OpenCL's local work items: that is, there is one "work item"
  // per "work group", which is what we call a thread. Furthermore, since there
  // is exactly one work item per work group, the global work-item ID always
  // equals the work-group ID. The current work index must be supplied as an
  // implicit trailing parameter to the kernel invocation.
  sem::Type idxType{sem::Type::INDEX};
  llvm::Value* zero = llvm::ConstantInt::get(int32type_, 0);
  switch (n.type) {
    case sem::IndexExpr::LOCAL:
      Resolve(value{zero, idxType});
      break;
    case sem::IndexExpr::GROUP:
    case sem::IndexExpr::GLOBAL:
      llvm::Value* ndim = llvm::ConstantInt::get(int32type_, n.dim);
      llvm::Value* resptr = builder_.CreateGEP(workIndex_, {zero, ndim});
      Resolve(value{builder_.CreateLoad(resptr), idxType});
      break;
  }
}

void Emit::Visit(const sem::Block& n) {
  Enter();
  for (const sem::StmtPtr& s : n.statements) {
    s->Accept(*this);
  }
  Leave();
}

void Emit::Visit(const sem::IfStmt& n) {
  auto thenblock = llvm::BasicBlock::Create(context_, "then");
  auto elseblock = llvm::BasicBlock::Create(context_, "else");
  auto doneblock = llvm::BasicBlock::Create(context_, "done");
  value cond = Eval(n.cond);
  builder_.CreateCondBr(ToBool(cond.v), thenblock, elseblock);

  function_->getBasicBlockList().push_back(thenblock);
  builder_.SetInsertPoint(thenblock);
  Enter();
  if (nullptr != n.iftrue) {
    n.iftrue->Accept(*this);
  }
  Leave();
  if (!CurrentBlockIsTerminated()) {
    builder_.CreateBr(doneblock);
  }

  function_->getBasicBlockList().push_back(elseblock);
  builder_.SetInsertPoint(elseblock);
  Enter();
  if (nullptr != n.iffalse) {
    n.iffalse->Accept(*this);
  }
  Leave();
  if (!CurrentBlockIsTerminated()) {
    builder_.CreateBr(doneblock);
  }

  function_->getBasicBlockList().push_back(doneblock);
  builder_.SetInsertPoint(doneblock);
}

void Emit::Visit(const sem::ForStmt& n) {
  auto bodyblock = llvm::BasicBlock::Create(context_, "body");
  auto testblock = llvm::BasicBlock::Create(context_, "test");
  auto iterblock = llvm::BasicBlock::Create(context_, "iter");
  auto doneblock = llvm::BasicBlock::Create(context_, "done");

  Enter();
  llvm::Value* ivar = Define(n.var, {sem::Type::INDEX});
  builder_.CreateStore(llvm::ConstantInt::get(ssizetype_, 0), ivar);
  builder_.CreateBr(testblock);

  function_->getBasicBlockList().push_back(testblock);
  builder_.SetInsertPoint(testblock);
  llvm::Value* limit = llvm::ConstantInt::get(ssizetype_, n.num * n.step);
  llvm::Value* cur_ivar = builder_.CreateLoad(ivar);
  llvm::Value* cond = builder_.CreateICmpSLT(cur_ivar, limit);
  builder_.CreateCondBr(cond, bodyblock, doneblock);

  function_->getBasicBlockList().push_back(bodyblock);
  builder_.SetInsertPoint(bodyblock);
  EnterLoop(doneblock, iterblock);
  n.inner->Accept(*this);
  Leave();
  if (!CurrentBlockIsTerminated()) {
    builder_.CreateBr(iterblock);
  }

  function_->getBasicBlockList().push_back(iterblock);
  builder_.SetInsertPoint(iterblock);
  llvm::Value* step = llvm::ConstantInt::get(ssizetype_, n.step);
  llvm::Value* next = builder_.CreateAdd(cur_ivar, step);
  builder_.CreateStore(next, ivar);
  builder_.CreateBr(testblock);

  function_->getBasicBlockList().push_back(doneblock);
  builder_.SetInsertPoint(doneblock);
  Leave();
}

void Emit::Visit(const sem::WhileStmt& n) {
  auto bodyblock = llvm::BasicBlock::Create(context_, "body");
  auto testblock = llvm::BasicBlock::Create(context_, "test");
  auto doneblock = llvm::BasicBlock::Create(context_, "done");
  builder_.CreateBr(testblock);

  function_->getBasicBlockList().push_back(testblock);
  builder_.SetInsertPoint(testblock);
  value cond = Eval(n.cond);
  builder_.CreateCondBr(ToBool(cond.v), bodyblock, doneblock);

  function_->getBasicBlockList().push_back(bodyblock);
  builder_.SetInsertPoint(bodyblock);
  EnterLoop(doneblock, testblock);
  n.inner->Accept(*this);
  Leave();
  builder_.CreateBr(testblock);

  function_->getBasicBlockList().push_back(doneblock);
  builder_.SetInsertPoint(doneblock);
}

void Emit::Visit(const sem::BarrierStmt& n) {
  std::vector<llvm::Value*> args;
  builder_.CreateCall(barrier_func_, args, "");
}

void Emit::Visit(const sem::ReturnStmt& n) {
  if (CurrentBlockIsTerminated()) {
    throw Error("unreachable duplicate return in this block");
  }
  if (returntype_.base != sem::Type::TVOID) {
    if (!n.value) {
      throw Error("must return non-void value from this function");
    }
    value rval = Eval(n.value);
    builder_.CreateRet(CastTo(rval, returntype_));
  } else {
    if (n.value) {
      throw Error("must not return a value from a void function");
    }
    builder_.CreateRet(nullptr);
  }
}

void Emit::Visit(const sem::Function& n) {
  // Convert parameter and return types. In addition to the formal parameters,
  // we will append an implicit parameter carrying the work item indexes.
  std::vector<llvm::Type*> param_types;
  for (const auto& p : n.params) {
    param_types.push_back(CType(p.first));
  }
  param_types.push_back(gridSizeType_->getPointerTo());
  returntype_ = n.ret;
  llvm::Type* ret = CType(returntype_);

  // Build the function type and create its entrypoint block.
  auto ftype = llvm::FunctionType::get(ret, param_types, false);
  auto linkage = llvm::Function::ExternalLinkage;
  const char* nstr = n.name.c_str();
  function_ = llvm::Function::Create(ftype, linkage, nstr, module_.get());
  auto bb = llvm::BasicBlock::Create(context_, "entry", function_);
  builder_.SetInsertPoint(bb);
  Enter();

  // Create all the parameters as specified in the function signature.
  for (auto ai = function_->arg_begin(); ai != function_->arg_end(); ++ai) {
    unsigned idx = ai->getArgNo();
    if (idx < n.params.size()) {
      // One of the formal parameters
      std::string paramName = n.params[idx].second;
      sem::Type paramType = n.params[idx].first;
      llvm::Value* alloc = Define(paramName, paramType);
      ai->setName(paramName);
      builder_.CreateStore(&(*ai), alloc);
    } else {
      // The implicit work item index parameter
      ai->setName("_workIndex");
      workIndex_ = &(*ai);
    }
  }

  // Emit the body of the function, which is probably a block.
  n.body->Accept(*this);

  // If this block has not yet been terminated, generate an implicit return.
  if (!CurrentBlockIsTerminated()) {
    builder_.CreateRet(nullptr);
  }

  Leave();
  returntype_.base = sem::Type::TVOID;

  funcopt_->run(*function_);
}

std::string Emit::str() const {
  std::string r;
  llvm::raw_string_ostream os(r);
  os << *module_;
  os.flush();
  return r;
}

Emit::value Emit::Process(const sem::Node& n) {
  value nv;
  std::swap(result_, nv);
  try {
    n.Accept(*this);
  } catch (Error err) {
    VLOG(1) << "cpu::Emit::Process failed on this node:\n" << sem::Print(n).str();
    throw;
  }
  std::swap(result_, nv);
  return nv;
}

Emit::value Emit::Eval(sem::ExprPtr e) {
  assert(e);
  return Process(*e);
}

Emit::value Emit::LVal(sem::LValPtr l) {
  assert(l);
  value out = Process(*l);
  assert(nullptr != out.v && out.v->getType()->isPointerTy());
  return out;
}

void Emit::Resolve(value out) {
  assert(nullptr == result_.v);
  assert(nullptr != out.v);
  result_ = out;
}

llvm::Type* Emit::CType(const sem::Type& type) {
  if (type.base == sem::Type::INDEX) {
    assert(1 == type.vec_width && 0 == type.array);
    return ssizetype_;
  }
  if (type.base == sem::Type::TVOID) {
    assert(1 == type.vec_width && 0 == type.array);
    return llvm::Type::getVoidTy(context_);
  }
  llvm::Type* t = nullptr;
  switch (type.dtype) {
    case DataType::FLOAT16:
      t = llvm::Type::getHalfTy(context_);
      break;
    case DataType::FLOAT32:
      t = llvm::Type::getFloatTy(context_);
      break;
    case DataType::FLOAT64:
      t = llvm::Type::getDoubleTy(context_);
      break;
    default:
      t = llvm::IntegerType::get(context_, bit_width(type.dtype));
  }
  if (type.vec_width > 1) {
    t = llvm::VectorType::get(t, type.vec_width);
  }
  if (type.base == sem::Type::POINTER_MUT || type.base == sem::Type::POINTER_CONST) {
    t = llvm::PointerType::get(t, 0);
  }
  if (type.array > 0) {
    t = llvm::ArrayType::get(t, type.array);
  }
  return t;
}

llvm::Value* Emit::Define(const std::string& name, sem::Type type) {
  auto& symbols = blocks_.front().symbols;
  if (symbols.find(name) != symbols.end()) {
    throw Error("Duplicate definitions in same block");
  }
  llvm::BasicBlock& entry = function_->getEntryBlock();
  llvm::IRBuilder<> top(&entry, entry.begin());
  llvm::Value* ptr = top.CreateAlloca(CType(type), nullptr, name.c_str());
  symbols.emplace(name, value{ptr, type});
  return ptr;
}

Emit::value Emit::Lookup(const std::string& name) {
  for (const auto& m : blocks_) {
    const auto it = m.symbols.find(name);
    if (it != m.symbols.end()) {
      return it->second;
    }
  }
  throw Error("Lookup of variable \"" + name + "\" failed");
}

llvm::Value* Emit::CastTo(value val, sem::Type to) {
  llvm::Type* from_type = val.v->getType();
  llvm::Type* to_type = CType(to);
  if (from_type == to_type) {
    return val.v;
  }
  if (!llvm::CastInst::isCastable(from_type, to_type)) {
    std::string desc = print(from_type) + " to " + print(to_type);
    throw Error("Illegal typecast: " + desc);
  }
  bool src_signed = !IsUnsignedIntegerType(val.t);
  bool dst_signed = !IsUnsignedIntegerType(to);
  auto op = llvm::CastInst::getCastOpcode(val.v, src_signed, to_type, dst_signed);
  return builder_.CreateCast(op, val.v, to_type);
}

llvm::Value* Emit::ToBool(llvm::Value* val) {
  // Compare this value against zero. Return a bool (int1).
  if (val->getType() == booltype_) {
    return val;
  }
  llvm::Value* zero = llvm::ConstantInt::get(val->getType(), 0);
  return builder_.CreateICmpNE(val, zero);
}

void Emit::Enter() { blocks_.emplace_front(); }

void Emit::EnterLoop(llvm::BasicBlock* doneblock, llvm::BasicBlock* testblock) {
  Enter();
  blocks_.front().doneblock = doneblock;
  blocks_.front().testblock = testblock;
}

void Emit::Leave() { blocks_.pop_front(); }

bool Emit::CurrentBlockIsTerminated() { return nullptr != builder_.GetInsertBlock()->getTerminator(); }

bool Emit::IsUnsignedIntegerType(const sem::Type& t) {
  if (t.base != sem::Type::VALUE) return false;
  if (t.array > 0) return false;
  switch (t.dtype) {
    case DataType::BOOLEAN:
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
      return true;
    default:
      return false;
  }
}

bool Emit::IsFloatingPointType(const sem::Type& t) {
  if (t.base != sem::Type::VALUE) return false;
  if (t.array > 0) return false;
  switch (t.dtype) {
    case DataType::FLOAT16:
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return true;
    default:
      return false;
  }
}

bool Emit::PointerAddition(value left, value right) {
  llvm::Type* ltype = left.v->getType();
  llvm::Type* rtype = right.v->getType();
  if (ltype->isPointerTy() && rtype->isIntegerTy()) {
    std::vector<llvm::Value*> idxList{right.v};
    Resolve(value{builder_.CreateGEP(left.v, idxList), left.t});
    return true;
  } else if (ltype->isIntegerTy() && rtype->isPointerTy()) {
    std::vector<llvm::Value*> idxList{left.v};
    Resolve(value{builder_.CreateGEP(right.v, idxList), right.t});
    return true;
  }
  return false;
}

sem::Type Emit::ConvergeOperands(value* left, value* right) {
  // Find a common type for these operands. Convert both to the common type.
  // Modify our arguments, replacing the originals with the new values. Return
  // the common type we settled on.
  sem::Type comtype = lang::Promote({left->t, right->t});
  *left = value{CastTo(*left, comtype), comtype};
  *right = value{CastTo(*right, comtype), comtype};
  return comtype;
}

std::string Emit::print(const llvm::Type* t) {
  // LLVM objects don't have methods to return a string description, so we
  // have to print them to an output stream and capture the result.
  std::string str;
  llvm::raw_string_ostream rso(str);
  t->print(rso);
  return rso.str();
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
