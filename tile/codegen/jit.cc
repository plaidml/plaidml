// Copyright 2018, Intel Corp.

#include "tile/codegen/jit.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/DynamicLibrary.h>

#include <algorithm>
#include <deque>
#include <memory>

#include <half.hpp>

#include "base/util/printstring.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

namespace {
const char invoker_name_[] = "__invoke_";
}

class Executable {
 public:
  Executable(std::unique_ptr<llvm::Module>&& module, const std::vector<std::string>& parameters);
  void Run(const std::map<std::string, void*>& buffers);

 private:
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  std::vector<std::string> parameters_;
};

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Runtime : public llvm::RuntimeDyld::SymbolResolver {
 public:
  llvm::RuntimeDyld::SymbolInfo findSymbol(const std::string&) override;
  llvm::RuntimeDyld::SymbolInfo findSymbolInLogicalDylib(const std::string&) override;
};

class Compiler : private stripe::ConstStmtVisitor {
 public:
  Compiler();
  std::unique_ptr<Executable> CompileProgram(const stripe::Block& program);

 protected:
  explicit Compiler(llvm::Module* module);
  void GenerateInvoker(const stripe::Block& program, llvm::Function* main);
  llvm::Function* CompileBlock(const stripe::Block& block);
  void Visit(const stripe::Load&) override;
  void Visit(const stripe::Store&) override;
  void Visit(const stripe::Constant&) override;
  void Visit(const stripe::Special&) override;
  void Visit(const stripe::Intrinsic&) override;
  void Visit(const stripe::Block&) override;
  void Add(const stripe::Intrinsic&);
  void Subtract(const stripe::Intrinsic&);
  void Negate(const stripe::Intrinsic&);
  void Multiply(const stripe::Intrinsic&);
  void Divide(const stripe::Intrinsic&);
  void Mod(const stripe::Intrinsic&);
  void LessThan(const stripe::Intrinsic&);
  void LessThanOrEqualTo(const stripe::Intrinsic&);
  void GreaterThan(const stripe::Intrinsic&);
  void GreaterThanOrEqualTo(const stripe::Intrinsic&);
  void Equal(const stripe::Intrinsic&);
  void Unequal(const stripe::Intrinsic&);
  void Conditional(const stripe::Intrinsic&);
  void VisitZERO(const stripe::Special&);
  void VisitCOPY(const stripe::Special&);

  struct scalar {
    llvm::Value* value = nullptr;
    DataType type = DataType::INVALID;
  };

  struct buffer {
    llvm::Value* base = nullptr;
    const stripe::Refinement* refinement = nullptr;
  };

  struct index {
    llvm::Value* variable = nullptr;
  };

  struct loop {
    llvm::BasicBlock* init = nullptr;
    llvm::BasicBlock* test = nullptr;
    llvm::BasicBlock* body = nullptr;
    llvm::BasicBlock* done = nullptr;
  };

 private:
  scalar Cast(scalar, DataType);
  llvm::Type* CType(DataType);
  llvm::Type* CType(const TensorShape&);
  llvm::Value* IndexElement(const buffer& buf);
  void OutputType(llvm::Value* ret, const stripe::Intrinsic&);
  void OutputBool(llvm::Value* ret, const stripe::Intrinsic&);

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  llvm::Module* module_ = nullptr;

  std::map<std::string, scalar> scalars_;
  std::map<std::string, buffer> buffers_;
  std::map<std::string, index> indexes_;
};

Compiler::Compiler() : context_(llvm::getGlobalContext()), builder_{context_} {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    LLVMInitializeNativeTarget();
    LLVMLinkInMCJIT();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
}

std::unique_ptr<Executable> Compiler::CompileProgram(const stripe::Block& program) {
  std::cerr << "Compiler::CompileProgram:" << std::endl;
  std::cerr << program << std::endl << "----------------" << std::endl;
  module_ = new llvm::Module("stripe", context_);
  llvm::Function* main = CompileBlock(program);
  GenerateInvoker(program, main);
  std::vector<std::string> param_names;
  for (auto& ref : program.refs) {
    param_names.push_back(ref.into);
  }
  module_->print(llvm::errs(), nullptr);
  std::unique_ptr<llvm::Module> xfermod(module_);
  module_ = nullptr;
  return std::make_unique<Executable>(std::move(xfermod), param_names);
}

Compiler::Compiler(llvm::Module* module) : context_(llvm::getGlobalContext()), builder_{context_}, module_(module) {
  // This private constructor sets up a nested compiler instance which will
  // process a nested block, generating output into the same module as its
  // containing compiler instance.
}

void Compiler::GenerateInvoker(const stripe::Block& program, llvm::Function* main) {
  // Generate a wrapper function for this program, so that we can call it
  // generically through a single C function pointer type, no matter how
  // many buffer parameters it expects. From C, we will prepare a vector
  // of void*, containing the parameter buffer data pointers; the wrapper
  // will extract each data pointer, then pass each one as a parameter when
  // it calls the program block function.
  // LLVM doesn't have the notion of a void pointer, so we'll pretend all of
  // these buffers are arrays of int32.
  llvm::Type* arrayptr = builder_.getInt8PtrTy()->getPointerTo();
  llvm::Type* voidtype = builder_.getVoidTy();
  auto invoker_type = llvm::FunctionType::get(voidtype, {arrayptr}, false);
  auto linkage = llvm::Function::ExternalLinkage;
  auto invoker = llvm::Function::Create(invoker_type, linkage, invoker_name_, module_);
  auto block = llvm::BasicBlock::Create(context_, "block", invoker);
  builder_.SetInsertPoint(block);
  // We'll look up the kernel by name and implicitly bitcast it so we can call
  // it using our int32-pointers in place of whatever it actually expects;
  // LLVM will tolerate this mismatch when we use getOrInsertFunction.
  size_t param_count = program.refs.size();
  auto ai = invoker->arg_begin();
  llvm::Value* argvec = &(*ai);
  // The body of the invoker will simply compute the element pointer for each
  // argument value in order, then load the value.
  std::vector<llvm::Value*> args;
  for (unsigned i = 0; i < param_count; ++i) {
    llvm::Value* index = builder_.getInt32(i);
    std::vector<llvm::Value*> idxList{index};
    llvm::Value* elptr = builder_.CreateGEP(argvec, idxList);
    llvm::Value* elval = builder_.CreateLoad(elptr);
    llvm::Type* eltype = CType(program.refs[i].shape)->getPointerTo();
    args.push_back(builder_.CreateBitCast(elval, eltype));
  }
  // Having built the argument list, we'll call the actual kernel using the
  // parameter signature it expects.
  builder_.CreateCall(main, args, "");
  builder_.CreateRetVoid();
}

llvm::Function* Compiler::CompileBlock(const stripe::Block& block) {
  // Generate a function implementing the body of this block.
  // Buffers (refinements) will be passed in as function parameters.
  // create an array of parameter types, one for each buffer
  std::vector<llvm::Type*> param_types;
  for (const auto& ref : block.refs) {
    param_types.push_back(CType(ref.shape)->getPointerTo());
  }
  // create a type for the function which will contain the block
  llvm::Type* return_type = builder_.getVoidTy();
  auto func_type = llvm::FunctionType::get(return_type, param_types, false);
  // create the LLVM function which will implement the Stripe block
  auto linkage = llvm::Function::ExternalLinkage;
  auto name = block.name;
  auto function = llvm::Function::Create(func_type, linkage, name, module_);
  // create a basic block; configure the builder to start there
  auto bb = llvm::BasicBlock::Create(context_, "entry", function);
  builder_.SetInsertPoint(bb);
  // associate parameter values with buffer names
  for (auto ai = function->arg_begin(); ai != function->arg_end(); ++ai) {
    unsigned idx = ai->getArgNo();
    std::string param_name = block.refs[idx].into;
    ai->setName(param_name);
    buffers_[param_name] = buffer{&(*ai), &block.refs[idx]};
  }
  // allocate storage for each loop index
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  llvm::Type* ssizetype = llvm::IntegerType::get(context_, archbits);
  for (auto& idx : block.idxs) {
    llvm::Value* variable = builder_.CreateAlloca(ssizetype);
    variable->setName(idx.name);
    indexes_[idx.name] = index{variable};
  }
  // generate the basic blocks for each nested loop's evaluation stages
  std::vector<loop> loops;
  for (auto& idx : block.idxs) {
    std::string name = idx.name;
    auto init = llvm::BasicBlock::Create(context_, "init_" + name, function);
    auto test = llvm::BasicBlock::Create(context_, "test_" + name, function);
    auto body = llvm::BasicBlock::Create(context_, "body_" + name, function);
    auto done = llvm::BasicBlock::Create(context_, "done_" + name, function);
    loops.push_back({init, test, body, done});
  }
  // initialize each loop index and generate the termination check
  llvm::Value* zero = llvm::ConstantInt::get(ssizetype, 0);
  for (size_t i = 0; i < block.idxs.size(); ++i) {
    builder_.CreateBr(loops[i].init);
    builder_.SetInsertPoint(loops[i].init);
    llvm::Value* variable = indexes_[block.idxs[i].name].variable;
    builder_.CreateStore(zero, variable);
    builder_.CreateBr(loops[i].test);
    builder_.SetInsertPoint(loops[i].test);
    llvm::Value* index = builder_.CreateLoad(variable);
    llvm::Value* range = llvm::ConstantInt::get(ssizetype, block.idxs[i].range);
    llvm::Value* go = builder_.CreateICmpULT(index, range);
    builder_.CreateCondBr(go, loops[i].body, loops[i].done);
    builder_.SetInsertPoint(loops[i].body);
  }

  // process each statement in the block body, generating code to modify the
  // parameter buffer contents
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }

  // increment each index, from innermost to outermost, then jump back to the test
  for (size_t i = block.idxs.size(); i-- > 0;) {
    llvm::Value* variable = indexes_[block.idxs[i].name].variable;
    llvm::Value* index = builder_.CreateLoad(variable);
    llvm::Value* increment = llvm::ConstantInt::get(ssizetype, 1);
    index = builder_.CreateAdd(index, increment);
    builder_.CreateStore(index, variable);
    builder_.CreateBr(loops[i].test);
    builder_.SetInsertPoint(loops[i].done);
  }

  // advance the indexes and repeat if necessary
  builder_.CreateRetVoid();
  return function;
}

void Compiler::Visit(const stripe::Load& load) {
  // op->from is the name of a source buffer
  // op->into is the name of a destination scalar
  buffer from = buffers_[load.from];
  // Look up the address of the target element.
  // Load the value from that address and use it to redefine the
  // destination scalar.
  llvm::Value* element = IndexElement(from);
  llvm::Value* value = builder_.CreateLoad(element);
  scalars_[load.into] = scalar{value, from.refinement->shape.type};
}

void Compiler::Visit(const stripe::Store& store) {
  // op->from is the name of a source scalar
  // op->into is the name of a destination buffer
  // get the offset into the destination buffer from the scope context
  // look up the expected aggregation operation for the destination from the context
  // load the value to be stored from the source variable
  // use GEP to compute the destination element address
  // use the specified aggregation to store the value
  buffer into = buffers_[store.into];
  scalar from = Cast(scalars_[store.from], into.refinement->shape.type);
  llvm::Value* value = from.value;
  llvm::Value* element = IndexElement(into);
  std::string agg_op = into.refinement->agg_op;
  if ("add" == agg_op) {
    llvm::Value* prev = builder_.CreateLoad(element);
    if (is_float(from.type)) {
      value = builder_.CreateFAdd(value, prev);
    } else if (is_int(from.type) || is_uint(from.type)) {
      value = builder_.CreateAdd(value, prev);
    } else {
      throw Error("Invalid addition type: " + to_string(from.type));
    }
  } else if (!agg_op.empty()) {
    throw Error("Unimplemented agg_op: " + to_string(agg_op));
  }
  builder_.CreateStore(value, element);
}

void Compiler::Visit(const stripe::Constant& constant) {
  // store a constant integer or float value into a scalar
  switch (constant.type) {
    case stripe::ConstType::Integer: {
      auto ty = builder_.getInt64Ty();
      auto value = llvm::ConstantInt::get(ty, constant.iconst);
      scalars_[constant.name] = scalar{value, DataType::INT64};
    } break;
    case stripe::ConstType::Float: {
      auto ty = builder_.getDoubleTy();
      auto value = llvm::ConstantFP::get(ty, constant.fconst);
      scalars_[constant.name] = scalar{value, DataType::FLOAT64};
    } break;
  }
}

void Compiler::Visit(const stripe::Special& special) {
  std::map<std::string, std::function<void(void)>> handlers{
      {"zero", [&]() { VisitZERO(special); }},
      {"copy", [&]() { VisitCOPY(special); }},
  };
  handlers[special.name]();
}

void Compiler::Visit(const stripe::Intrinsic& intrinsic) {
  // Find the correct handler for this intrinsic.
  // Note that stripe::Intrinsic defines a bunch of strings which are not actually intrinsics;
  // they are actually values for stripe::Refinement::agg_op. Not sure why they are located in
  // the wrong structure. There are no constants defined for actual intrinsic names; these have
  // been derived experimentally.
  std::map<std::string, std::function<void(void)>> handlers{
      {"add", [&]() { Add(intrinsic); }},
      {"sub", [&]() { Subtract(intrinsic); }},
      {"neg", [&]() { Negate(intrinsic); }},
      {"mul", [&]() { Multiply(intrinsic); }},
      {"div", [&]() { Divide(intrinsic); }},
      {"mod", [&]() { Mod(intrinsic); }},
      {"lt", [&]() { LessThan(intrinsic); }},
      {"lte", [&]() { LessThanOrEqualTo(intrinsic); }},
      {"gt", [&]() { GreaterThan(intrinsic); }},
      {"gte", [&]() { GreaterThanOrEqualTo(intrinsic); }},
      {"eq", [&]() { Equal(intrinsic); }},
      {"neq", [&]() { Unequal(intrinsic); }},
      //    {"and", [&](){ And(intrinsic); }},
      //    {"or", [&](){ Or(intrinsic); }},
      //    {"not", [&](){ Not(intrinsic); }},
      //    {"xor", [&](){ Xor(intrinsic); }},
      {"cond", [&]() { Conditional(intrinsic); }},
  };
  handlers[intrinsic.name]();
}

void Compiler::Visit(const stripe::Block& block) {
  // Compile a nested block as a function in the same module
  Compiler nested(module_);
  auto function = nested.CompileBlock(block);
  // Generate a list of args which are the buffers the block expects to receive.
  std::vector<llvm::Value*> args;
  for (auto& ref : block.refs) {
    std::string name = ref.from.empty() ? ref.into : ref.from;
    llvm::Value* base = buffers_[name].base;
    llvm::Type* eltype = CType(ref.shape)->getPointerTo();
    args.push_back(builder_.CreateBitCast(base, eltype));
  }
  // Invoke the function
  builder_.CreateCall(function, args, "");
}

void Compiler::Add(const stripe::Intrinsic& add) {
  // Accepts two inputs, cast to operation type
  assert(2 == add.inputs.size());
  scalar lhs = Cast(scalars_[add.inputs[0]], add.type);
  scalar rhs = Cast(scalars_[add.inputs[1]], add.type);
  // Sum placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_float(add.type)) {
    ret = builder_.CreateFAdd(lhs.value, rhs.value);
  } else if (is_int(add.type) || is_uint(add.type)) {
    ret = builder_.CreateAdd(lhs.value, rhs.value);
  } else {
    throw Error("Invalid addition type: " + to_string(add.type));
  }
  OutputType(ret, add);
}

void Compiler::Subtract(const stripe::Intrinsic& sub) {
  // Accepts two inputs, cast to operation type
  assert(2 == sub.inputs.size());
  scalar lhs = Cast(scalars_[sub.inputs[0]], sub.type);
  scalar rhs = Cast(scalars_[sub.inputs[1]], sub.type);
  // Difference placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_float(sub.type)) {
    ret = builder_.CreateFSub(lhs.value, rhs.value);
  } else if (is_int(sub.type) || is_uint(sub.type)) {
    ret = builder_.CreateSub(lhs.value, rhs.value);
  } else {
    throw Error("Invalid subtraction type: " + to_string(sub.type));
  }
  OutputType(ret, sub);
}

void Compiler::Negate(const stripe::Intrinsic& neg) {
  // Accepts one input
  assert(1 == neg.inputs.size());
  scalar op = Cast(scalars_[neg.inputs[0]], neg.type);
  // Negated operand value placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_float(neg.type)) {
    ret = builder_.CreateFNeg(op.value);
  } else if (is_int(neg.type) || is_uint(neg.type)) {
    ret = builder_.CreateNeg(op.value);
  } else {
    throw Error("Invalid negation type: " + to_string(neg.type));
  }
  OutputType(ret, neg);
}

void Compiler::Multiply(const stripe::Intrinsic& mul) {
  // Accepts two inputs, cast to operation type
  assert(2 == mul.inputs.size());
  scalar lhs = Cast(scalars_[mul.inputs[0]], mul.type);
  scalar rhs = Cast(scalars_[mul.inputs[1]], mul.type);
  // Product placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_float(mul.type)) {
    ret = builder_.CreateFMul(lhs.value, rhs.value);
  } else if (is_int(mul.type) || is_uint(mul.type)) {
    ret = builder_.CreateMul(lhs.value, rhs.value);
  } else {
    throw Error("Invalid multiplication type: " + to_string(mul.type));
  }
  OutputType(ret, mul);
}

void Compiler::Divide(const stripe::Intrinsic& div) {
  // Accepts two inputs, cast to operation type
  assert(2 == div.inputs.size());
  scalar lhs = Cast(scalars_[div.inputs[0]], div.type);
  scalar rhs = Cast(scalars_[div.inputs[1]], div.type);
  // Product placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_float(div.type)) {
    ret = builder_.CreateFDiv(lhs.value, rhs.value);
  } else if (is_int(div.type)) {
    ret = builder_.CreateSDiv(lhs.value, rhs.value);
  } else if (is_uint(div.type)) {
    ret = builder_.CreateUDiv(lhs.value, rhs.value);
  } else {
    throw Error("Invalid division type: " + to_string(div.type));
  }
  OutputType(ret, div);
}

void Compiler::Mod(const stripe::Intrinsic& mod) {
  // Accepts two inputs, cast to operation type
  assert(2 == mod.inputs.size());
  scalar lhs = Cast(scalars_[mod.inputs[0]], mod.type);
  scalar rhs = Cast(scalars_[mod.inputs[1]], mod.type);
  // Product placed into the single output
  // Output type is operation type
  llvm::Value* ret = nullptr;
  if (is_int(mod.type)) {
    ret = builder_.CreateSRem(lhs.value, rhs.value);
  } else if (is_uint(mod.type)) {
    ret = builder_.CreateURem(lhs.value, rhs.value);
  } else {
    throw Error("Invalid modulo type: " + to_string(mod.type));
  }
  OutputType(ret, mod);
}

void Compiler::LessThan(const stripe::Intrinsic& lt) {
  // Accepts two inputs
  assert(2 == lt.inputs.size());
  scalar lhs = Cast(scalars_[lt.inputs[0]], lt.type);
  scalar rhs = Cast(scalars_[lt.inputs[1]], lt.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(lt.type)) {
    ret = builder_.CreateFCmpOLT(lhs.value, rhs.value);
  } else if (is_int(lt.type)) {
    ret = builder_.CreateICmpSLT(lhs.value, rhs.value);
  } else if (is_uint(lt.type)) {
    ret = builder_.CreateICmpULT(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(lt.type));
  }
  OutputBool(ret, lt);
}

void Compiler::LessThanOrEqualTo(const stripe::Intrinsic& lte) {
  // Accepts two inputs
  assert(2 == lte.inputs.size());
  scalar lhs = Cast(scalars_[lte.inputs[0]], lte.type);
  scalar rhs = Cast(scalars_[lte.inputs[1]], lte.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(lte.type)) {
    ret = builder_.CreateFCmpOLE(lhs.value, rhs.value);
  } else if (is_int(lte.type)) {
    ret = builder_.CreateICmpSLE(lhs.value, rhs.value);
  } else if (is_uint(lte.type)) {
    ret = builder_.CreateICmpULE(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(lte.type));
  }
  OutputBool(ret, lte);
}

void Compiler::GreaterThan(const stripe::Intrinsic& gt) {
  // Accepts two inputs
  assert(2 == gt.inputs.size());
  scalar lhs = Cast(scalars_[gt.inputs[0]], gt.type);
  scalar rhs = Cast(scalars_[gt.inputs[1]], gt.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(gt.type)) {
    ret = builder_.CreateFCmpOGT(lhs.value, rhs.value);
  } else if (is_int(gt.type)) {
    ret = builder_.CreateICmpSGT(lhs.value, rhs.value);
  } else if (is_uint(gt.type)) {
    ret = builder_.CreateICmpUGT(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(gt.type));
  }
  OutputBool(ret, gt);
}

void Compiler::GreaterThanOrEqualTo(const stripe::Intrinsic& gte) {
  // Accepts two inputs
  assert(2 == gte.inputs.size());
  scalar lhs = Cast(scalars_[gte.inputs[0]], gte.type);
  scalar rhs = Cast(scalars_[gte.inputs[1]], gte.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(gte.type)) {
    ret = builder_.CreateFCmpOGE(lhs.value, rhs.value);
  } else if (is_int(gte.type)) {
    ret = builder_.CreateICmpSGE(lhs.value, rhs.value);
  } else if (is_uint(gte.type)) {
    ret = builder_.CreateICmpUGE(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(gte.type));
  }
  OutputBool(ret, gte);
}

void Compiler::Equal(const stripe::Intrinsic& eq) {
  // Accepts two inputs
  assert(2 == eq.inputs.size());
  scalar lhs = Cast(scalars_[eq.inputs[0]], eq.type);
  scalar rhs = Cast(scalars_[eq.inputs[1]], eq.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(eq.type)) {
    ret = builder_.CreateFCmpOEQ(lhs.value, rhs.value);
  } else if (is_int(eq.type) || is_uint(eq.type)) {
    ret = builder_.CreateICmpEQ(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(eq.type));
  }
  OutputBool(ret, eq);
}

void Compiler::Unequal(const stripe::Intrinsic& neq) {
  // Accepts two inputs
  assert(2 == neq.inputs.size());
  scalar lhs = Cast(scalars_[neq.inputs[0]], neq.type);
  scalar rhs = Cast(scalars_[neq.inputs[1]], neq.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(neq.type)) {
    ret = builder_.CreateFCmpONE(lhs.value, rhs.value);
  } else if (is_int(neq.type) || is_uint(neq.type)) {
    ret = builder_.CreateICmpNE(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(neq.type));
  }
  OutputBool(ret, neq);
}

void Compiler::Conditional(const stripe::Intrinsic& cond) {
  // Three inputs: C, T, F; C is boolean, T and F are operation type
  assert(3 == cond.inputs.size());
  scalar c = scalars_[cond.inputs[0]];
  if (DataType::BOOLEAN != c.type) {
    throw Error("Condition expression is non-boolean: " + to_string(c.type));
  }
  scalar t = Cast(scalars_[cond.inputs[1]], cond.type);
  scalar f = Cast(scalars_[cond.inputs[2]], cond.type);
  // Single output will be one of T or F
  // Output type is operation type
  llvm::Value* ret = builder_.CreateSelect(c.value, t.value, f.value);
  OutputType(ret, cond);
}

void Compiler::VisitZERO(const stripe::Special& zero) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation ZERO is not yet specified");
}

void Compiler::VisitCOPY(const stripe::Special& copy) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation COPY is not yet specified");
}

Compiler::scalar Compiler::Cast(scalar v, DataType to_type) {
  if (v.type == to_type) {
    return v;
  }
  llvm::Type* to_llvmtype = CType(to_type);
  bool from_signed = is_int(v.type) || is_float(v.type);
  bool to_signed = is_int(to_type) || is_float(to_type);
  auto op = llvm::CastInst::getCastOpcode(v.value, from_signed, to_llvmtype, to_signed);
  llvm::Value* ret = builder_.CreateCast(op, v.value, to_llvmtype);
  return scalar{ret, to_type};
}

llvm::Type* Compiler::CType(DataType type) {
  switch (type) {
    case DataType::BOOLEAN:
      return builder_.getInt1Ty();
    case DataType::INT8:
    case DataType::UINT8:
      return builder_.getInt8Ty();
    case DataType::INT16:
    case DataType::UINT16:
      return builder_.getInt16Ty();
    case DataType::INT32:
    case DataType::UINT32:
      return builder_.getInt32Ty();
    case DataType::INT64:
    case DataType::UINT64:
      return builder_.getInt64Ty();
    case DataType::FLOAT16:
      return builder_.getHalfTy();
    case DataType::FLOAT32:
      return builder_.getFloatTy();
    case DataType::FLOAT64:
      return builder_.getDoubleTy();
    case DataType::INVALID:
    case DataType::PRNG:
      throw Error("Invalid type: " + to_string(type));
  }
}

llvm::Type* Compiler::CType(const TensorShape& shape) {
  llvm::Type* type = CType(shape.type);
  for (auto bound : shape.dims) {
    type = llvm::ArrayType::get(type, bound.size);
  }
  return type;
}

llvm::Value* Compiler::IndexElement(const buffer& buf) {
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  llvm::Type* ssizetype = llvm::IntegerType::get(context_, archbits);
  llvm::Value* zero = llvm::ConstantInt::get(ssizetype, 0);
  std::vector<llvm::Value*> idxList{zero};
  // Iterate through the source refinement's "access" elements to find
  // the names of the element indexes. Load the current value of each
  // index variable and add it to the GEP index list.
  if (buf.refinement->access.size()) {
    for (const auto& access : buf.refinement->access) {
      std::string indexName = access.toString();
      llvm::Value* indexVar = indexes_[indexName].variable;
      idxList.push_back(builder_.CreateLoad(indexVar));
    }
  } else {
    // Special case for a one-dimensional, one-element array
    idxList.push_back(zero);
  }
  return builder_.CreateGEP(buf.base, idxList);
}

void Compiler::OutputType(llvm::Value* ret, const stripe::Intrinsic& intrinsic) {
  assert(1 == intrinsic.outputs.size());
  scalars_[intrinsic.outputs[0]] = scalar{ret, intrinsic.type};
}

void Compiler::OutputBool(llvm::Value* ret, const stripe::Intrinsic& intrinsic) {
  assert(1 == intrinsic.outputs.size());
  scalars_[intrinsic.outputs[0]] = scalar{ret, DataType::BOOLEAN};
}

Executable::Executable(std::unique_ptr<llvm::Module>&& module, const std::vector<std::string>& parameters)
    : parameters_(parameters) {
  std::string errStr;
  std::unique_ptr<llvm::RuntimeDyld::SymbolResolver> rez(new Runtime);
  auto ee = llvm::EngineBuilder(std::move(module))
                .setErrorStr(&errStr)
                .setEngineKind(llvm::EngineKind::JIT)
                .setVerifyModules(true)
                .setSymbolResolver(std::move(rez))
                .create();
  if (ee) {
    ee->finalizeObject();
    engine_.reset(ee);
  } else {
    throw Error("Failed to create ExecutionEngine: " + errStr);
  }
}

void Executable::Run(const std::map<std::string, void*>& buffers) {
  std::vector<void*> args(parameters_.size());
  for (size_t i = 0; i < args.size(); ++i) {
    args[i] = buffers.at(parameters_[i]);
  }
  void* argvec = args.data();
  uint64_t entrypoint = engine_->getFunctionAddress(invoker_name_);
  ((void (*)(void*))entrypoint)(argvec);
}

namespace rt {
// Implementations of support functions the tile backend will link against,
// that we won't be able to resolve from system libraries.
float h2f(half_float::half n) { return n; }
half_float::half f2h(float n) { return half_float::half_cast<half_float::half>(n); }
}  // namespace rt

template <typename T>
llvm::RuntimeDyld::SymbolInfo symInfo(T ptr) {
  auto flags = llvm::JITSymbolFlags::None;
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  return llvm::RuntimeDyld::SymbolInfo(addr, flags);
}

llvm::RuntimeDyld::SymbolInfo Runtime::findSymbol(const std::string& name) {
  static std::map<std::string, llvm::RuntimeDyld::SymbolInfo> symbols{
      {"__gnu_h2f_ieee", symInfo(rt::h2f)},
      {"__gnu_f2h_ieee", symInfo(rt::f2h)},
      {"___truncsfhf2", symInfo(rt::f2h)},
      {"___extendhfsf2", symInfo(rt::h2f)},
  };
  auto loc = symbols.find(name);
  if (loc != symbols.end()) {
    return loc->second;
  }
  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name);
  // If we failed to resolve the symbol, and its first character is an underscore, try again without
  // the underscore, because the code may have been generated for a system whose loader expects every
  // symbol to have an underscore prefix, but the DynamicLibrary module expects not to have a prefix.
  if (!ptr && name[0] == '_' && name.size() > 1) {
    ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name.substr(1));
  }
  if (ptr) {
    auto info = symInfo(ptr);
    symbols.emplace(name, info);
    return info;
  }
  throw Error("failed to resolve external symbol reference: \"" + name + "\"");
}

llvm::RuntimeDyld::SymbolInfo Runtime::findSymbolInLogicalDylib(const std::string& name) {
  return llvm::RuntimeDyld::SymbolInfo(nullptr);
}

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers) {
  Compiler compiler;
  auto executable = compiler.CompileProgram(program);
  executable->Run(buffers);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
