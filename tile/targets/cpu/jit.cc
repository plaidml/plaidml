// Copyright 2018, Intel Corp.

#include "tile/targets/cpu/jit.h"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <algorithm>
#include <deque>
#include <memory>

#include <half.hpp>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

namespace {
const char invoker_name_[] = "__invoke_";
}

struct ProgramModule {
  std::unique_ptr<llvm::Module> module;
  std::vector<std::string> parameters;
};

class Executable {
 public:
  explicit Executable(ProgramModule&& module);
  void Run(const std::map<std::string, void*>& buffers);
  void Save(const std::string& filename);

 private:
  std::unique_ptr<llvm::ExecutionEngine> engine_;
  std::vector<std::string> parameters_;
};

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Runtime : public llvm::LegacyJITSymbolResolver {
 public:
  llvm::JITSymbol findSymbol(const std::string&) override;
  llvm::JITSymbol findSymbolInLogicalDylib(const std::string&) override;
};

class Compiler : private stripe::ConstStmtVisitor {
 public:
  explicit Compiler(llvm::LLVMContext* context);
  ProgramModule CompileProgram(const stripe::Block& program);

 protected:
  explicit Compiler(llvm::LLVMContext* context, llvm::Module* module);
  void GenerateInvoker(const stripe::Block& program, llvm::Function* main);
  llvm::Function* CompileBlock(const stripe::Block& block);
  void Visit(const stripe::Load&) override;
  void Visit(const stripe::Store&) override;
  void Visit(const stripe::LoadIndex&) override;
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
  void And(const stripe::Intrinsic&);
  void Or(const stripe::Intrinsic&);
  void Not(const stripe::Intrinsic&);
  void Xor(const stripe::Intrinsic&);
  void Assign(const stripe::Intrinsic&);
  void BitLeft(const stripe::Intrinsic&);
  void BitRight(const stripe::Intrinsic&);
  void Zero(const stripe::Special&);
  void Copy(const stripe::Special&);
  void Reshape(const stripe::Special&);

  struct scalar {
    llvm::Value* value = nullptr;
    DataType type = DataType::INVALID;
  };

  struct buffer {
    const stripe::Refinement* refinement = nullptr;
    llvm::Value* base = nullptr;
  };

  struct index {
    const stripe::Index* index = nullptr;
    llvm::Value* variable = nullptr;
    llvm::Value* init = nullptr;
  };

  struct loop {
    llvm::BasicBlock* init = nullptr;
    llvm::BasicBlock* test = nullptr;
    llvm::BasicBlock* body = nullptr;
    llvm::BasicBlock* done = nullptr;
  };

 private:
  scalar Cast(scalar, DataType);
  scalar CheckBool(scalar);
  llvm::Type* CType(DataType);
  llvm::Value* ElementPtr(const buffer& buf);
  llvm::Value* Eval(const stripe::Affine& access);
  void OutputType(llvm::Value* ret, const stripe::Intrinsic&);
  void OutputBool(llvm::Value* ret, const stripe::Intrinsic&);
  llvm::Type* IndexType();
  llvm::Value* IndexConst(ssize_t val);
  llvm::FunctionType* BlockType(const stripe::Block&);
  llvm::Value* MallocFunction();
  llvm::Value* FreeFunction();

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  llvm::Module* module_ = nullptr;

  std::map<std::string, scalar> scalars_;
  std::map<std::string, buffer> buffers_;
  std::map<std::string, index> indexes_;
};

Compiler::Compiler(llvm::LLVMContext* context) : context_(*context), builder_{context_} {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    LLVMInitializeNativeTarget();
    LLVMLinkInMCJIT();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
}

ProgramModule Compiler::CompileProgram(const stripe::Block& program) {
  // Compile each block in this program into a function within an LLVM module.
  ProgramModule ret;
  ret.module = std::make_unique<llvm::Module>("stripe", context_);
  module_ = ret.module.get();
  llvm::Function* main = CompileBlock(program);
  // Generate a stub function we can invoke from the outside, passing buffers
  // as an array of generic pointers.
  GenerateInvoker(program, main);
  // Improve the simple-minded IR we've just generated by running module-level
  // optimization passes; among many other things, this will streamline our
  // loops to eliminate most branches and inline most block function calls.
  llvm::PassManagerBuilder pmb;
  pmb.OptLevel = 3;
  pmb.SizeLevel = 0;
  pmb.SLPVectorize = true;
  pmb.LoopVectorize = true;
  pmb.MergeFunctions = true;
  llvm::legacy::PassManager modopt;
  pmb.populateModulePassManager(modopt);
  if (VLOG_IS_ON(2)) {
    IVLOG(2, "\n============================================================\n");
    module_->print(llvm::errs(), nullptr);
  }
  modopt.run(*module_);
  if (VLOG_IS_ON(2)) {
    IVLOG(2, "\n============================================================\n");
    module_->print(llvm::errs(), nullptr);
  }
  // Wrap the finished module and the buffer names into an Executable instance.
  for (auto& ref : program.refs) {
    assert(ref.dir != stripe::RefDir::None);
    ret.parameters.push_back(ref.into());
  }
  module_ = nullptr;
  return ret;
}

Compiler::Compiler(llvm::LLVMContext* context, llvm::Module* module)
    : context_(*context), builder_{context_}, module_(module) {
  // This private constructor sets up a nested compiler instance which will
  // process a nested block, generating output into the same module as its
  // containing compiler instance.
}

void Compiler::GenerateInvoker(const stripe::Block& program, llvm::Function* main) {
  // Generate a wrapper function for this program, so that we can call it
  // generically through a single C function pointer type, no matter how many
  // buffer parameters it expects. From C, we will prepare a vector of void*,
  // containing the parameter buffer data pointers; the wrapper will extract
  // each data pointer, then pass each one as a parameter when it calls the
  // program's top-level block function.
  // LLVM doesn't have the notion of a void pointer, so we'll pretend all of
  // these buffers are arrays of int8, then bitcast later.
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
  auto ai = invoker->arg_begin();
  llvm::Value* argvec = &(*ai);
  // The body of the invoker will compute the element pointer for each
  // argument value in order, then load the value.
  std::vector<llvm::Value*> args;
  {
    unsigned i = 0;
    for (auto& ref : program.refs) {
      llvm::Value* index = builder_.getInt32(i++);
      std::vector<llvm::Value*> idxList{index};
      llvm::Value* elptr = builder_.CreateGEP(argvec, idxList);
      llvm::Value* elval = builder_.CreateLoad(elptr);
      llvm::Type* eltype = CType(ref.interior_shape.type)->getPointerTo();
      args.push_back(builder_.CreateBitCast(elval, eltype));
    }
  }
  // After passing in buffer pointers, we also provide an initial value for
  // each index; since this is the outermost block, all indexes begin at zero.
  for (unsigned i = 0; i < program.idxs.size(); ++i) {
    args.push_back(IndexConst(0));
  }
  // Having built the argument list, we'll call the actual kernel using the
  // parameter signature it expects.
  builder_.CreateCall(main, args, "");
  builder_.CreateRetVoid();
}

llvm::Function* Compiler::CompileBlock(const stripe::Block& block) {
  // Generate a function implementing the body of this block.
  // Buffers (refinements) will be passed in as function parameters, as will
  // the initial value for each index.

  for (const auto& ref : block.refs) {
    buffers_[ref.into()] = buffer{&ref};
  }
  for (const auto& idx : block.idxs) {
    indexes_[idx.name] = index{&idx};
  }

  // create the LLVM function which will implement the Stripe block
  auto linkage = llvm::Function::ExternalLinkage;
  auto name = block.name;
  auto func_type = BlockType(block);
  auto function = llvm::Function::Create(func_type, linkage, name, module_);
  // create a basic block; configure the builder to start there
  auto bb = llvm::BasicBlock::Create(context_, "entry", function);
  builder_.SetInsertPoint(bb);

  // associate parameter values with buffers and indexes
  for (auto ai = function->arg_begin(); ai != function->arg_end(); ++ai) {
    unsigned idx = ai->getArgNo();
    if (idx < block.refs.size()) {
      auto it = block.refs.begin();
      std::advance(it, idx);
      std::string param_name = it->into();
      ai->setName(param_name);
      assert(nullptr == buffers_[param_name].base);
      buffers_[param_name].base = &(*ai);
    } else {
      idx -= block.refs.size();
      std::string param_name = block.idxs[idx].name;
      ai->setName(param_name);
      assert(nullptr == indexes_[param_name].init);
      indexes_[param_name].init = &(*ai);
    }
  }

  // allocate storage for each loop index
  for (auto& idx : block.idxs) {
    llvm::Value* variable = builder_.CreateAlloca(IndexType());
    variable->setName(idx.name);
    assert(nullptr == indexes_[idx.name].variable);
    indexes_[idx.name].variable = variable;
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
  for (size_t i = 0; i < block.idxs.size(); ++i) {
    builder_.CreateBr(loops[i].init);
    builder_.SetInsertPoint(loops[i].init);
    llvm::Value* variable = indexes_[block.idxs[i].name].variable;
    llvm::Value* init = indexes_[block.idxs[i].name].init;
    builder_.CreateStore(init, variable);
    builder_.CreateBr(loops[i].test);
    builder_.SetInsertPoint(loops[i].test);
    llvm::Value* index = builder_.CreateLoad(variable);
    llvm::Value* range = IndexConst(block.idxs[i].range);
    llvm::Value* limit = builder_.CreateAdd(init, range);
    llvm::Value* go = builder_.CreateICmpULT(index, limit);
    builder_.CreateCondBr(go, loops[i].body, loops[i].done);
    builder_.SetInsertPoint(loops[i].body);
  }

  // check the constraints against the current index values and decide whether
  // to execute the block body for this iteration
  llvm::Value* go = builder_.getTrue();
  for (auto& constraint : block.constraints) {
    llvm::Value* gateval = Eval(constraint);
    llvm::Value* check = builder_.CreateICmpSGE(gateval, IndexConst(0));
    go = builder_.CreateAnd(check, go);
  }
  auto block_body = llvm::BasicBlock::Create(context_, "block", function);
  auto block_done = llvm::BasicBlock::Create(context_, "next", function);
  builder_.CreateCondBr(go, block_body, block_done);
  builder_.SetInsertPoint(block_body);

  // process each statement in the block body, generating code to modify the
  // parameter buffer contents
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }

  // rejoin instruction flow after the constraint check
  builder_.CreateBr(block_done);
  builder_.SetInsertPoint(block_done);

  // increment each index, from innermost to outermost, then jump back to test
  for (size_t i = block.idxs.size(); i-- > 0;) {
    llvm::Value* variable = indexes_[block.idxs[i].name].variable;
    llvm::Value* index = builder_.CreateLoad(variable);
    llvm::Value* increment = IndexConst(1);
    index = builder_.CreateAdd(index, increment);
    builder_.CreateStore(index, variable);
    builder_.CreateBr(loops[i].test);
    builder_.SetInsertPoint(loops[i].done);
  }

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
  llvm::Value* element = ElementPtr(from);
  llvm::Value* value = builder_.CreateLoad(element);
  scalars_[load.into] = scalar{value, from.refinement->interior_shape.type};
}

void Compiler::Visit(const stripe::Store& store) {
  // op->from is the name of a source scalar
  // op->into is the name of a destination buffer
  // get the offset into the destination buffer from the scope context
  // look up the expected aggregation operation for the destination from the
  // context
  // load the value to be stored from the source variable
  // use GEP to compute the destination element address
  // use the specified aggregation to store the value
  buffer into = buffers_[store.into];
  scalar from = Cast(scalars_[store.from], into.refinement->interior_shape.type);
  llvm::Value* value = from.value;
  llvm::Value* element = ElementPtr(into);
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
  } else if ("max" == agg_op) {
    llvm::Value* prev = builder_.CreateLoad(element);
    llvm::Value* flag = nullptr;
    if (is_float(from.type)) {
      flag = builder_.CreateFCmpUGT(prev, value);
    } else if (is_int(from.type)) {
      flag = builder_.CreateICmpSGT(prev, value);
    } else if (is_uint(from.type)) {
      flag = builder_.CreateICmpUGT(prev, value);
    }
    value = builder_.CreateSelect(flag, prev, value);
  } else if ("assign" == agg_op) {
    // fall through to assignment
  } else if (!agg_op.empty()) {
    throw Error("Unimplemented agg_op: " + to_string(agg_op));
  }
  builder_.CreateStore(value, element);
}

void Compiler::Visit(const stripe::LoadIndex& load_index) {
  // op->from is an affine
  // op->into is the name of a destination scalar
  llvm::Value* rval = Eval(load_index.from);
  llvm::Value* value = builder_.CreateLoad(rval);
  scalars_[load_index.into] = scalar{value, DataType::INT64};
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
  static std::map<std::string, std::function<void(Compiler*, const stripe::Special&)>> handlers{
      {"zero", &Compiler::Zero},
      {"copy", &Compiler::Copy},
      {"reshape", &Compiler::Reshape},
  };
  auto it = handlers.find(special.name);
  if (it == handlers.end()) {
    throw Error("Unknown special \"" + special.name + "\"");
  }
  it->second(this, special);
}

void Compiler::Visit(const stripe::Intrinsic& intrinsic) {
  // Find the correct handler for this intrinsic.
  // Note that stripe::Intrinsic defines a bunch of strings which are not
  // actually intrinsics; they are values for stripe::Refinement::agg_op. Not
  // sure why they are located in the wrong structure. There are no constants
  // defined for actual intrinsic names; these have been derived experimentally.
  static std::map<std::string, std::function<void(Compiler*, const stripe::Intrinsic&)>> handlers{
      {"add", &Compiler::Add},
      {"sub", &Compiler::Subtract},
      {"neg", &Compiler::Negate},
      {"mul", &Compiler::Multiply},
      {"div", &Compiler::Divide},
      {"mod", &Compiler::Mod},
      {"lt", &Compiler::LessThan},
      {"lte", &Compiler::LessThanOrEqualTo},
      {"gt", &Compiler::GreaterThan},
      {"gte", &Compiler::GreaterThanOrEqualTo},
      {"eq", &Compiler::Equal},
      {"neq", &Compiler::Unequal},
      {"and", &Compiler::And},
      {"or", &Compiler::Or},
      {"not", &Compiler::Not},
      {"xor", &Compiler::Xor},
      {"cond", &Compiler::Conditional},
      {"assign", &Compiler::Assign},
      {"bit_right", &Compiler::BitRight},
      {"bit_left", &Compiler::BitLeft},
  };
  auto it = handlers.find(intrinsic.name);
  if (it == handlers.end()) {
    throw Error("Unknown intrinsic \"" + intrinsic.name + "\"");
  }
  it->second(this, intrinsic);
}

void Compiler::Visit(const stripe::Block& block) {
  // Compile a nested block as a function in the same module
  Compiler nested(&context_, module_);
  auto function = nested.CompileBlock(block);
  // Generate a list of args.
  // The argument list begins with a pointer to each refinement. We will either
  // pass along the address of a refinement from the current block, or allocate
  // a new buffer for the nested block's use.
  std::vector<llvm::Value*> args;
  std::vector<llvm::Value*> allocs;
  for (auto& ref : block.refs) {
    llvm::Value* buffer = nullptr;
    // When a refinement is neither in nor out, and it has no "from"
    // name, it represents a local allocation.
    if (ref.dir == stripe::RefDir::None && ref.from.empty()) {
      // Allocate new storage for the buffer.
      size_t size = ref.interior_shape.byte_size();
      std::vector<llvm::Value*> malloc_args;
      malloc_args.push_back(IndexConst(size));
      auto malloc_func = MallocFunction();
      buffer = builder_.CreateCall(malloc_func, malloc_args, "");
      allocs.push_back(buffer);
      llvm::Type* buftype = CType(ref.interior_shape.type)->getPointerTo();
      buffer = builder_.CreateBitCast(buffer, buftype);
    } else {
      // Pass in the current element address from the source buffer.
      // If a "from" name is specified, use that buffer; if not, that means
      // that both blocks use the same name, so use "into".
      std::string name = ref.from.empty() ? ref.into() : ref.from;
      buffer = ElementPtr(buffers_[name]);
    }
    args.push_back(buffer);
  }
  // Following the list of refinement args, we will provide a list of initial
  // values for each of the block's indexes, which are specified as an affine
  // in terms of the current block's indexes.
  for (auto& idx : block.idxs) {
    args.push_back(Eval(idx.affine));
  }
  // Invoke the function. It does not return a value.
  builder_.CreateCall(function, args, "");
  // Free the temporary buffers we allocated as parameter values.
  for (auto ptr : allocs) {
    std::vector<llvm::Value*> free_args;
    free_args.push_back(ptr);
    auto free_func = FreeFunction();
    builder_.CreateCall(free_func, free_args, "");
  }
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
  scalar c = CheckBool(scalars_[cond.inputs[0]]);
  scalar t = Cast(scalars_[cond.inputs[1]], cond.type);
  scalar f = Cast(scalars_[cond.inputs[2]], cond.type);
  // Single output will be one of T or F
  // Output type is operation type
  llvm::Value* ret = builder_.CreateSelect(c.value, t.value, f.value);
  OutputType(ret, cond);
}

void Compiler::And(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateAnd(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Or(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateOr(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Not(const stripe::Intrinsic& stmt) {
  assert(1 == stmt.inputs.size());
  scalar op = CheckBool(scalars_[stmt.inputs[0]]);
  llvm::Value* ret = builder_.CreateNot(op.value);
  OutputBool(ret, stmt);
}

void Compiler::Xor(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateXor(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Assign(const stripe::Intrinsic& stmt) {
  assert(1 == stmt.inputs.size());
  scalar op = Cast(scalars_[stmt.inputs[0]], stmt.type);
  llvm::Value* ret = op.value;
  OutputType(ret, stmt);
}

void Compiler::BitLeft(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  scalar lhs = Cast(scalars_[stmt.inputs[0]], stmt.type);
  scalar rhs = Cast(scalars_[stmt.inputs[1]], stmt.type);
  llvm::Value* ret = builder_.CreateShl(lhs.value, rhs.value);
  OutputType(ret, stmt);
}

void Compiler::BitRight(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  scalar lhs = Cast(scalars_[stmt.inputs[0]], stmt.type);
  scalar rhs = Cast(scalars_[stmt.inputs[1]], stmt.type);
  llvm::Value* ret = nullptr;
  if (is_int(stmt.type)) {
    ret = builder_.CreateAShr(lhs.value, rhs.value);
  } else if (is_uint(stmt.type)) {
    ret = builder_.CreateLShr(lhs.value, rhs.value);
  } else {
    throw Error("Invalid bitshift type: " + to_string(stmt.type));
  }
  OutputType(ret, stmt);
}

void Compiler::Zero(const stripe::Special& zero) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation ZERO is not yet specified");
}

void Compiler::Copy(const stripe::Special& copy) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation COPY is not yet specified");
}

void Compiler::Reshape(const stripe::Special& reshape) {
  throw Error("Special operation RESHAPE is not yet specified");
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

Compiler::scalar Compiler::CheckBool(scalar v) {
  if (v.type == DataType::BOOLEAN) {
    return v;
  }
  throw Error("Expected boolean, actually found " + to_string(v.type));
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
    case DataType::INT128:
    case DataType::PRNG:
    case DataType::INVALID:
      throw Error("Invalid type: " + to_string(type));
  }
  return builder_.getVoidTy();
}

llvm::Value* Compiler::ElementPtr(const buffer& buf) {
  // Ask the source refinement to generate an access path, in the form of
  // a sequence of indexes to scale and sum. Load each index value, multiply,
  // and the result is an offset from the buffer base address.
  llvm::Value* offset = Eval(buf.refinement->FlatAccess());
  std::vector<llvm::Value*> idxList{offset};
  return builder_.CreateGEP(buf.base, idxList);
}

llvm::Value* Compiler::Eval(const stripe::Affine& access) {
  llvm::Value* offset = IndexConst(0);
  for (auto& term : access.getMap()) {
    llvm::Value* indexVal = nullptr;
    if (!term.first.empty()) {
      llvm::Value* indexVar = indexes_[term.first].variable;
      indexVal = builder_.CreateLoad(indexVar);
      llvm::Value* multiplier = IndexConst(term.second);
      indexVal = builder_.CreateMul(indexVal, multiplier);
    } else {
      indexVal = IndexConst(term.second);
    }
    offset = builder_.CreateAdd(offset, indexVal);
  }
  return offset;
}

void Compiler::OutputType(llvm::Value* ret, const stripe::Intrinsic& intrinsic) {
  assert(1 == intrinsic.outputs.size());
  scalars_[intrinsic.outputs[0]] = scalar{ret, intrinsic.type};
}

void Compiler::OutputBool(llvm::Value* ret, const stripe::Intrinsic& intrinsic) {
  assert(1 == intrinsic.outputs.size());
  scalars_[intrinsic.outputs[0]] = scalar{ret, DataType::BOOLEAN};
}

llvm::Type* Compiler::IndexType() {
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  return llvm::IntegerType::get(context_, archbits);
}

llvm::Value* Compiler::IndexConst(ssize_t val) {
  llvm::Type* ssizetype = IndexType();
  return llvm::ConstantInt::get(ssizetype, val);
}

llvm::FunctionType* Compiler::BlockType(const stripe::Block& block) {
  // Generate a type for the function which will implement this block.
  std::vector<llvm::Type*> param_types;
  // Each buffer base address will be provided as a parameter.
  for (const auto& ref : block.refs) {
    param_types.push_back(CType(ref.interior_shape.type)->getPointerTo());
  }
  // Following the buffers, a parameter will provide the initial value for
  // each of the block's indexes.
  for (size_t i = 0; i < block.idxs.size(); ++i) {
    param_types.push_back(IndexType());
  }
  // Blocks never return a value.
  llvm::Type* return_type = builder_.getVoidTy();
  return llvm::FunctionType::get(return_type, param_types, false);
}

llvm::Value* Compiler::MallocFunction(void) {
  std::vector<llvm::Type*> argtypes{IndexType()};
  llvm::Type* rettype = builder_.getInt8PtrTy();
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "malloc";
  return module_->getOrInsertFunction(funcname, functype);
}

llvm::Value* Compiler::FreeFunction(void) {
  llvm::Type* ptrtype = builder_.getInt8PtrTy();
  std::vector<llvm::Type*> argtypes{ptrtype};
  llvm::Type* rettype = llvm::Type::getVoidTy(context_);
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "free";
  return module_->getOrInsertFunction(funcname, functype);
}

Executable::Executable(ProgramModule&& module) : parameters_(module.parameters) {
  std::string errStr;
  std::unique_ptr<llvm::LegacyJITSymbolResolver> rez(new Runtime);
  auto ee = llvm::EngineBuilder(std::move(module.module))
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
llvm::JITEvaluatedSymbol symInfo(T ptr) {
  auto flags = llvm::JITSymbolFlags::None;
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  return llvm::JITEvaluatedSymbol(addr, flags);
}

llvm::JITSymbol Runtime::findSymbol(const std::string& name) {
  static std::map<std::string, llvm::JITEvaluatedSymbol> symbols{
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
  // If we failed to resolve the symbol, and its first character is an
  // underscore, try again without
  // the underscore, because the code may have been generated for a system whose
  // loader expects every
  // symbol to have an underscore prefix, but the DynamicLibrary module expects
  // not to have a prefix.
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

llvm::JITSymbol Runtime::findSymbolInLogicalDylib(const std::string& name) { return llvm::JITSymbol(nullptr); }

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers) {
  llvm::LLVMContext context;
  Compiler compiler(&context);
  auto module = compiler.CompileProgram(program);
  Executable executable(std::move(module));
  executable.Run(buffers);
}

struct Native::Impl {
  llvm::LLVMContext context;
  ProgramModule module;

  void compile(const stripe::Block& program) {
    Compiler compiler(&context);
    module = compiler.CompileProgram(program);
  }

  void run(const std::map<std::string, void*>& buffers) {
    Executable executable(std::move(module));
    executable.Run(buffers);
  }

  void save(const std::string& filename) {
    std::error_code ec;
    llvm::ToolOutputFile result(filename, ec, llvm::sys::fs::F_None);
    WriteBitcodeToFile(*module.module, result.os());
    result.keep();
  }
};

Native::Native() : m_impl(new Native::Impl) {}
Native::~Native() {}
void Native::compile(const stripe::Block& program) { m_impl->compile(program); }
void Native::run(const std::map<std::string, void*>& buffers) { m_impl->run(buffers); }
void Native::save(const std::string& filename) { m_impl->save(filename); }

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
