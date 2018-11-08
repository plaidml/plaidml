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
  void Equal(const stripe::Intrinsic&);
  void Conditional(const stripe::Intrinsic&);
  void VisitZERO(const stripe::Special&);
  void VisitCOPY(const stripe::Special&);

  struct scalar {
    llvm::Value* value = nullptr;
    DataType type = DataType::INVALID;
  };

  struct buffer {
    llvm::Value* base = nullptr;
    TensorShape shape;
  };

 private:
  scalar Cast(scalar, DataType);
  llvm::Type* CType(DataType);
  llvm::Type* CType(const TensorShape&);

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  llvm::Module* module_ = nullptr;

  std::map<std::string, scalar> scalars_;
  std::map<std::string, buffer> buffers_;
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
  module_ = new llvm::Module("stripe", context_);
  llvm::Function* main = CompileBlock(program);
  GenerateInvoker(program, main);
  std::vector<std::string> param_names;
  for (auto& ref : program.refs) {
    param_names.push_back(ref.into);
  }
  // module_->print(llvm::errs(), nullptr);
  std::unique_ptr<llvm::Module> xfermod(module_);
  module_ = nullptr;
  return std::make_unique<Executable>(std::move(xfermod), param_names);
}

Compiler::Compiler(llvm::Module* module) : context_(llvm::getGlobalContext()), builder_{context_}, module_(module) {}

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
    buffers_[param_name] = buffer{&(*ai), block.refs[idx].shape};
  }
  // process each statement in the block body, generating code to modify the
  // parameter buffer contents
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }
  // advance the indexes and repeat if necessary
  builder_.CreateRetVoid();
  return function;
}

void Compiler::Visit(const stripe::Load& load) {
  // op->from is the name of a source buffer
  // op->into is the name of a destination scalar
  // get the current offset into the source buffer from the scope context
  // use GEP to compute the source element address
  // load the value from that address
  // store the value into the destination variable
  buffer from = buffers_[load.from];
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  llvm::Type* ssizetype = llvm::IntegerType::get(context_, archbits);
  llvm::Value* zero = llvm::ConstantInt::get(ssizetype, 0);
  std::vector<llvm::Value*> idxList{zero, zero};
  llvm::Value* element = builder_.CreateGEP(from.base, idxList);
  llvm::Value* value = builder_.CreateLoad(element);
  scalars_[load.into] = scalar{value, from.shape.type};
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
  llvm::Value* value = scalars_[store.from].value;
  unsigned archbits = module_->getDataLayout().getPointerSizeInBits();
  llvm::Type* ssizetype = llvm::IntegerType::get(context_, archbits);
  llvm::Value* zero = llvm::ConstantInt::get(ssizetype, 0);
  std::vector<llvm::Value*> idxList{zero, zero};
  llvm::Value* element = builder_.CreateGEP(into.base, idxList);
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
      //    {"mod", [&](){ Mod(intrinsic); }},
      //    {"lt", [&](){ LessThan(intrinsic); }},
      //    {"lte", [&](){ LessThanOrEqualTo(intrinsic); }},
      //    {"gt", [&](){ GreaterThan(intrinsic); }},
      //    {"gte", [&](){ GreaterThanOrEqualTo(intrinsic); }},
      {"eq", [&]() { Equal(intrinsic); }},
      //    {"neq", [&](){ Unequal(intrinsic); }},
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
  // Generate a list of args which are the buffers the block expects to receive
  std::vector<llvm::Value*> callarg;
  // Invoke the function
  builder_.CreateCall(function, callarg, "");
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
  assert(1 == add.outputs.size());
  scalars_[add.outputs[0]] = scalar{ret, add.type};
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
  assert(1 == sub.outputs.size());
  scalars_[sub.outputs[0]] = scalar{ret, sub.type};
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
  assert(1 == neg.outputs.size());
  scalars_[neg.outputs[0]] = scalar{ret, neg.type};
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
  assert(1 == mul.outputs.size());
  scalars_[mul.outputs[0]] = scalar{ret, mul.type};
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
  assert(1 == div.outputs.size());
  scalars_[div.outputs[0]] = scalar{ret, div.type};
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
  } else if (DataType::BOOLEAN == eq.type) {
    ret = builder_.CreateICmpEQ(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type: " + to_string(eq.type));
  }
  assert(1 == eq.outputs.size());
  scalars_[eq.outputs[0]] = scalar{ret, DataType::BOOLEAN};
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
  assert(1 == cond.outputs.size());
  scalars_[cond.outputs[0]] = scalar{ret, cond.type};
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
