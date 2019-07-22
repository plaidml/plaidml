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
#include <llvm/Transforms/Utils/Cloning.h>

#include <algorithm>
#include <deque>
#include <memory>

#include <half.hpp>

#include "base/util/lookup.h"
#include "tile/stripe/stripe.h"

// libxsmm
#include "libxsmm_source.h"  // NOLINT

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
  std::map<std::string, void*> externals;
};

class Executable {
 public:
  explicit Executable(const ProgramModule& module);
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
  explicit Runtime(const std::map<std::string, void*> externals) : externals_(externals) {}
  llvm::JITSymbol findSymbol(const std::string&) override;
  llvm::JITSymbol findSymbolInLogicalDylib(const std::string&) override;

 private:
  std::map<std::string, void*> externals_;
};

class Compiler : private stripe::ConstStmtVisitor {
 public:
  Compiler(llvm::LLVMContext* context, const std::map<std::string, External>& externals);
  ProgramModule CompileProgram(const stripe::Block& program);
  virtual ~Compiler() {
    // make sure the blockStack is drained.
    blockStack_.clear();
  }

 protected:
  explicit Compiler(llvm::LLVMContext* context, llvm::Module* module, const std::map<std::string, External>& externals,
                    std::vector<std::shared_ptr<stripe::Block>> blockStack);

  void GenerateInvoker(const stripe::Block& program, llvm::Function* main);
  llvm::Function* CompileXSMMBlock(const stripe::Block& block, const stripe::Block* const dimBlock, DataType dataType);
  llvm::Function* CompileBlock(const stripe::Block& block);
  void Visit(const stripe::Load&) override;
  void Visit(const stripe::Store&) override;
  void Visit(const stripe::LoadIndex&) override;
  void Visit(const stripe::Constant&) override;
  void Visit(const stripe::Special&) override;
  void Visit(const stripe::Intrinsic&) override;
  void Visit(const stripe::Block&) override;
  void Intrinsic(const stripe::Intrinsic&, External handler);
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
  void Sqrt(const stripe::Intrinsic&);
  void Exp(const stripe::Intrinsic&);
  void Log(const stripe::Intrinsic&);
  void Pow(const stripe::Intrinsic&);
  void Tanh(const stripe::Intrinsic&);
  void Cos(const stripe::Intrinsic&);
  void Zero(const stripe::Special&);
  void Copy(const stripe::Special&);
  void Reshape(const stripe::Special&);
  void PrngStep(const stripe::Special&);

  struct Scalar {
    llvm::Value* value = nullptr;
    DataType type = DataType::INVALID;
  };

  struct Buffer {
    const stripe::Refinement* refinement = nullptr;
    llvm::Value* base = nullptr;
  };

  struct Index {
    const stripe::Index* index = nullptr;
    llvm::Value* variable = nullptr;
    llvm::Value* init = nullptr;
  };

  struct Loop {
    llvm::BasicBlock* init = nullptr;
    llvm::BasicBlock* test = nullptr;
    llvm::BasicBlock* body = nullptr;
    llvm::BasicBlock* done = nullptr;
  };

 private:
  const stripe::Block* const GetBlockWithDimensionAttributes() const;
  Scalar Cast(Scalar, DataType);
  Scalar CheckBool(Scalar);
  llvm::Type* CType(DataType);
  llvm::Value* ElementPtr(const Buffer& buf);
  llvm::Value* Eval(const stripe::Affine& access);
  void OutputType(llvm::Value* ret, const stripe::Intrinsic&);
  void OutputBool(llvm::Value* ret, const stripe::Intrinsic&);
  void CallIntrinsicFunc(const stripe::Intrinsic&, const char* name_f32, const char* name_f64);
  llvm::Type* IndexType();
  llvm::Value* IndexConst(ssize_t val);
  llvm::FunctionType* BlockType(const stripe::Block&);
  llvm::Value* XSMMDispatchFunction(llvm::Type* alphaBetaPrtrType, const std::string& funcionName);
  llvm::Value* MallocFunction();
  llvm::Value* CallocFunction();
  llvm::Value* FreeFunction();
  llvm::Value* PrngStepFunction();
  bool isXSMMSuppotedDataType(DataType dataType);
  const DataType GetBlockRefsDataType(const stripe::Block& block);

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  llvm::Module* module_ = nullptr;
  std::map<std::string, External> external_handlers_;
  std::map<std::string, void*> external_funcptrs_;

  std::map<std::string, Scalar> scalars_;
  std::map<std::string, Buffer> buffers_;
  std::map<std::string, Index> indexes_;
  std::vector<std::shared_ptr<stripe::Block>> blockStack_;
};

Compiler::Compiler(llvm::LLVMContext* context, const std::map<std::string, External>& externals)
    : context_(*context), builder_{context_}, external_handlers_{externals} {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    LLVMInitializeNativeTarget();
    LLVMLinkInMCJIT();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();
  });
}

ProgramModule Compiler::CompileProgram(const stripe::Block& program) {
  IVLOG(4, program);
  // Compile each block in this program into a function within an LLVM module.
  ProgramModule ret;
  ret.module = std::make_unique<llvm::Module>("stripe", context_);
  module_ = ret.module.get();
  llvm::Function* main = CompileBlock(program);
  ret.externals = external_funcptrs_;
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
  if (VLOG_IS_ON(4)) {
    IVLOG(4, "\n============================================================\n");
    module_->print(llvm::errs(), nullptr);
  }
  modopt.run(*module_);
  if (VLOG_IS_ON(4)) {
    IVLOG(4, "\n============================================================\n");
    module_->print(llvm::errs(), nullptr);
  }
  // Wrap the finished module and the parameter names into a ProgramModule.
  for (auto& ref : program.refs) {
    if (ref.has_tag("user")) {
      ret.parameters.push_back(ref.into());
    }
  }
  module_ = nullptr;
  assert(ret.module);
  return ret;
}

Compiler::Compiler(llvm::LLVMContext* context, llvm::Module* module, const std::map<std::string, External>& externals,
                   std::vector<std::shared_ptr<stripe::Block>> blockStack)
    : context_(*context), builder_{context_}, module_(module), external_handlers_{externals}, blockStack_{blockStack} {
  // This private constructor sets up a nested instance which will
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
  std::vector<llvm::Value*> allocs;
  {
    unsigned i = 0;
    for (auto& ref : program.refs) {
      if (ref.has_tag("user")) {
        // The refinement must be provided as a parameter by the user
        llvm::Value* index = builder_.getInt32(i++);
        std::vector<llvm::Value*> idxList{index};
        llvm::Value* elptr = builder_.CreateGEP(argvec, idxList);
        llvm::Value* elval = builder_.CreateLoad(elptr);
        llvm::Type* eltype = CType(ref.interior_shape.type)->getPointerTo();
        args.push_back(builder_.CreateBitCast(elval, eltype));
      } else if (ref.has_tag("tmp")) {
        // Allocate a temporary buffer for this refinement
        size_t size = ref.interior_shape.byte_size();
        std::vector<llvm::Value*> calloc_args{IndexConst(size), IndexConst(1)};
        auto buffer = builder_.CreateCall(CallocFunction(), calloc_args, "");
        allocs.push_back(buffer);
        llvm::Type* buftype = CType(ref.interior_shape.type)->getPointerTo();
        args.push_back(builder_.CreateBitCast(buffer, buftype));
      } else {
        throw std::runtime_error("Top-level refinement missing #user or #tmp");
      }
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
  // Free any temporary buffers we may have allocated.
  for (auto ptr : allocs) {
    std::vector<llvm::Value*> free_args{ptr};
    builder_.CreateCall(FreeFunction(), free_args, "");
  }
  builder_.CreateRetVoid();
}

// Gets the parent block that has the GEMM dimension attributes.
const stripe::Block* const Compiler::GetBlockWithDimensionAttributes() const {
  for (auto it = blockStack_.crbegin(); it != blockStack_.crend(); ++it) {
    const std::shared_ptr<stripe::Block> block = *it;
    const stripe::Block* retBlock = block.get();
    if (retBlock->has_attr("m_idx") && retBlock->has_attr("n_idx") && retBlock->has_attr("k_idx")) {
      return retBlock;
    }
  }
  throw std::runtime_error("No dimensions set in the GEMM block hierarchy.");
}

llvm::Function* Compiler::CompileXSMMBlock(const stripe::Block& block, const stripe::Block* const dimBlock,
                                           const DataType dataType) {
  // Validate incomingparams.
  if (!dimBlock->has_attr("m_idx")) {
    throw std::runtime_error("Expected m_idx attribute not set!");
  }

  if (!dimBlock->has_attr("k_idx")) {
    throw std::runtime_error("Expected k_idx attribute not set!");
  }

  if (!dimBlock->has_attr("n_idx")) {
    throw std::runtime_error("Expected n_idx attribute not set!");
  }

  // Generate a function that implements the body for this block of statements.
  // Refinements (their buffers) and initial indexes
  // will be passed as parameters (to the function).
  for (const auto& ref : block.refs) {
    buffers_[ref.into()] = Buffer{&ref};
  }
  for (const auto& idx : block.idxs) {
    indexes_[idx.name] = Index{&idx};
  }

  // Create the LLVM function which will implement the Stripe block
  auto linkage = llvm::Function::ExternalLinkage;
  auto name = block.name;
  auto func_type = BlockType(block);
  auto function = llvm::Function::Create(func_type, linkage, name, module_);

  // Areate a basic block; configure the builder to start there
  auto bb = llvm::BasicBlock::Create(context_, "entry", function);
  builder_.SetInsertPoint(bb);

  // Associate parameter values with buffers and indexes
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
  auto i32t = builder_.getInt32Ty();
  llvm::Value* lda = builder_.CreateAlloca(i32t);
  llvm::Value* ldb = builder_.CreateAlloca(i32t);
  llvm::Value* ldc = builder_.CreateAlloca(i32t);

  llvm::Value* alpha = nullptr;
  llvm::Value* beta = nullptr;
  llvm::Value* one = nullptr;
  llvm::Type* alpha_beta_ptr_type = nullptr;
  std::string functionName("Invalid");
  switch (dataType) {
    case DataType::FLOAT32:
      one = llvm::ConstantFP::get(builder_.getFloatTy(), 1.0);
      alpha = builder_.CreateAlloca(builder_.getFloatTy());
      beta = builder_.CreateAlloca(builder_.getFloatTy());
      alpha_beta_ptr_type = llvm::Type::getFloatPtrTy(context_);
      functionName = "libxsmm_smmdispatch";
      break;

    case DataType::FLOAT64:
      one = llvm::ConstantFP::get(builder_.getDoubleTy(), 1.0L);
      alpha = builder_.CreateAlloca(builder_.getDoubleTy());
      beta = builder_.CreateAlloca(builder_.getDoubleTy());
      alpha_beta_ptr_type = llvm::Type::getDoublePtrTy(context_);
      functionName = "libxsmm_dmmdispatch";
      break;

    default:
      throw std::runtime_error("Unsupported DataType for XSMM.");
  }

  builder_.CreateStore(one, alpha);
  builder_.CreateStore(one, beta);

  builder_.CreateStore(llvm::ConstantInt::get(i32t, dimBlock->get_attr_int("m_idx")), lda);
  builder_.CreateStore(llvm::ConstantInt::get(i32t, dimBlock->get_attr_int("k_idx")), ldb);
  builder_.CreateStore(llvm::ConstantInt::get(i32t, dimBlock->get_attr_int("n_idx")), ldc);
  llvm::Value* nptr = llvm::ConstantPointerNull::get(llvm::Type::getInt32PtrTy(context_));
  llvm::Value* dispatch = XSMMDispatchFunction(alpha_beta_ptr_type, functionName);
  std::vector<llvm::Value*> args1 = {llvm::ConstantInt::get(i32t, FindIndexByTag(block, "stencil_m")->range),
                                     llvm::ConstantInt::get(i32t, FindIndexByTag(block, "stencil_n")->range),
                                     llvm::ConstantInt::get(i32t, FindIndexByTag(block, "stencil_k")->range),
                                     lda,
                                     ldb,
                                     ldc,
                                     alpha,
                                     beta,
                                     nptr,
                                     nptr};
  llvm::Value* func = builder_.CreateCall(dispatch, args1);
  IVLOG(1, block);
  for (const auto& kvp : buffers_) {
    IVLOG(1, "key:" << kvp.first);
    IVLOG(1, "value:" << kvp.second.base);
    // kvp.second.base->dump();
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    kvp.second.base->getType()->print(rso);
    IVLOG(1, "llvm_type:" << type_str);
    // kvp.second.base->dump();
  }
  IVLOG(1, block.ref_ins()[1]->into());
  IVLOG(1, block.ref_ins()[0]->into());
  IVLOG(1, block.ref_outs()[0]->into());
  std::vector<llvm::Type*> param_types{
      alpha_beta_ptr_type,  // a
      alpha_beta_ptr_type,  // b
      alpha_beta_ptr_type,  // c
  };
  llvm::FunctionType* rftype = llvm::FunctionType::get(builder_.getVoidTy(), param_types, false);
  std::vector<llvm::Value*> args2 = {buffers_[block.ref_ins()[1]->into()].base,
                                     buffers_[block.ref_ins()[0]->into()].base,
                                     buffers_[block.ref_outs()[0]->into()].base};
  builder_.CreateCall(rftype, func, args2);
  builder_.CreateRetVoid();
  return function;
}

bool Compiler::isXSMMSuppotedDataType(DataType dataType) {
  switch (dataType) {
    case DataType::FLOAT32:
    case DataType::FLOAT64:
      return true;

    default:
      return false;
  }
}

// Make sure all the refinments of this block are of the same type.
// If they are not, XSMM functions can't be called and we should
// to slower GEMM calculation process.
const DataType Compiler::GetBlockRefsDataType(const stripe::Block& block) {
  DataType retDataType = DataType::INVALID;
  bool firstIteration = true;
  const auto allRefs = block.refs;
  for (auto it = allRefs.cbegin(); it != allRefs.cend(); ++it) {
    if (firstIteration) {
      retDataType = it->interior_shape.type;
      firstIteration = false;
    } else {
      if (retDataType != it->interior_shape.type) {
        // Refinments with tdifferent DataType detected.
        // Return INVALID, so the XSMM logic detects XSMM
        // should not be used.
        return DataType::INVALID;
      }
    }
  }

  return retDataType;
}

llvm::Function* Compiler::CompileBlock(const stripe::Block& block) {
  if (block.has_tag("xsmm")) {
    const stripe::Block* const pBlock = GetBlockWithDimensionAttributes();
    DataType dataType = GetBlockRefsDataType(block);
    if (nullptr != pBlock && isXSMMSuppotedDataType(dataType)) {
      return CompileXSMMBlock(block, pBlock, dataType);
    }
  }

  // Generate a function implementing the body of this block.
  // Buffers (refinements) will be passed in as function parameters, as will
  // the initial value for each index.

  for (const auto& ref : block.refs) {
    buffers_[ref.into()] = Buffer{&ref};
  }
  for (const auto& idx : block.idxs) {
    indexes_[idx.name] = Index{&idx};
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
  std::vector<Loop> loops;
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
    assert(block.idxs[i].affine == stripe::Affine());
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
  std::shared_ptr<stripe::Block> pBlock = std::make_shared<stripe::Block>(block);
  blockStack_.push_back(pBlock);
  for (const auto& stmt : block.stmts) {
    stmt->Accept(this);
  }
  blockStack_.pop_back();

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
  Buffer from = buffers_[load.from];
  // Look up the address of the target element.
  // Load the value from that address and use it to redefine the
  // destination scalar.
  llvm::Value* element = ElementPtr(from);
  llvm::Value* value = builder_.CreateLoad(element);
  scalars_[load.into] = Scalar{value, from.refinement->interior_shape.type};
}

void Compiler::Visit(const stripe::Store& store) {
  // op->from is the name of a source scalar
  // op->into is the name of a destination buffer
  // get the offset into the destination buffer from the scope context
  // look up the expected aggregation operation for the destination from the
  // context (assign, add, product/mul, min, max)
  // load the value to be stored from the source variable
  // use GEP to compute the destination element address
  // use the specified aggregation to store the value
  Buffer into = buffers_[store.into];
  Scalar from = Cast(scalars_[store.from], into.refinement->interior_shape.type);
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
  } else if ("mul" == agg_op) {
    llvm::Value* prev = builder_.CreateLoad(element);
    if (is_float(from.type)) {
      value = builder_.CreateFMul(value, prev);
    } else if (is_int(from.type) || is_uint(from.type)) {
      value = builder_.CreateMul(value, prev);
    } else {
      throw Error("Invalid multiplication type: " + to_string(from.type));
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
  } else if ("min" == agg_op) {
    llvm::Value* prev = builder_.CreateLoad(element);
    llvm::Value* flag = nullptr;
    if (is_float(from.type)) {
      flag = builder_.CreateFCmpULT(prev, value);
    } else if (is_int(from.type)) {
      flag = builder_.CreateICmpSLT(prev, value);
    } else if (is_uint(from.type)) {
      flag = builder_.CreateICmpULT(prev, value);
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
  scalars_[load_index.into] = Scalar{rval, DataType::INT64};
}

void Compiler::Visit(const stripe::Constant& constant) {
  // store a constant integer or float value into a scalar
  switch (constant.type) {
    case stripe::ConstType::Integer: {
      auto ty = builder_.getInt64Ty();
      auto value = llvm::ConstantInt::get(ty, constant.iconst);
      scalars_[constant.name] = Scalar{value, DataType::INT64};
    } break;
    case stripe::ConstType::Float: {
      auto ty = builder_.getDoubleTy();
      auto value = llvm::ConstantFP::get(ty, constant.fconst);
      scalars_[constant.name] = Scalar{value, DataType::FLOAT64};
    } break;
  }
}

void Compiler::Visit(const stripe::Special& special) {
  // The list of specials defined in the spec differs from the list defined in
  // tile/lang/gen_special.cc. The spec lists "zero", "copy", and "reshape",
  // while gen_special.cc uses "gather", "scatter", "shape", and "prng_step".
  static std::map<std::string, std::function<void(Compiler*, const stripe::Special&)>> handlers{
      {"zero", &Compiler::Zero},
      {"copy", &Compiler::Copy},
      {"reshape", &Compiler::Reshape},
      {"prng_step", &Compiler::PrngStep},
  };
  auto it = handlers.find(special.name);
  if (it == handlers.end()) {
    throw Error("Unknown special \"" + special.name + "\"");
  }
  it->second(this, special);
}

void Compiler::Visit(const stripe::Intrinsic& intrinsic) {
  // Find the correct handler for this intrinsic.
  // If the context has provided an external handler for this intrinsic name,
  // we'll use it - that allows the context to override builtin definitions.
  // If there is no external handler, look up a builtin handler definition.
  // Note that stripe::Intrinsic defines a bunch of strings which are not
  // actually intrinsics; they are values for stripe::Refinement::agg_op. Not
  // sure why they are located in the wrong structure. There are no constants
  // defined for actual intrinsic names; these have been derived experimentally.
  static std::map<std::string, std::function<void(Compiler*, const stripe::Intrinsic&)>> builtins{
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
      {"ident", &Compiler::Assign},
      // Extra operations defined in tile/lang/ops.cc, which are apparently
      // passed along directly into Stripe
      {"cmp_eq", &Compiler::Equal},
      {"cmp_ne", &Compiler::Unequal},
      {"cmp_lt", &Compiler::LessThan},
      {"cmp_gt", &Compiler::GreaterThan},
      {"cmp_le", &Compiler::LessThanOrEqualTo},
      {"cmp_ge", &Compiler::GreaterThanOrEqualTo},
      {"bit_right", &Compiler::BitRight},
      {"bit_left", &Compiler::BitLeft},
      // Other undocumented intrinsics, which are apparently necessary in order
      // to successfully run the backend_test:
      {"sqrt", &Compiler::Sqrt},
      {"exp", &Compiler::Exp},
      {"log", &Compiler::Log},
      {"pow", &Compiler::Pow},
      {"tanh", &Compiler::Tanh},
      {"cos", &Compiler::Cos},
  };
  auto externiter = external_handlers_.find(intrinsic.name);
  if (externiter != external_handlers_.end()) {
    Intrinsic(intrinsic, externiter->second);
  } else {
    auto builtiniter = builtins.find(intrinsic.name);
    if (builtiniter == builtins.end()) {
      throw Error("Unknown intrinsic \"" + intrinsic.name + "\"");
    }
    builtiniter->second(this, intrinsic);
  }
}

void Compiler::Visit(const stripe::Block& block) {
  // Compile a nested block as a function in the same module
  Compiler nested(&context_, module_, external_handlers_, blockStack_);
  auto function = nested.CompileBlock(block);
  for (auto& fptr_iter : nested.external_funcptrs_) {
    external_funcptrs_.emplace(fptr_iter);
  }
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
      std::vector<llvm::Value*> calloc_args{IndexConst(size), IndexConst(1)};
      buffer = builder_.CreateCall(CallocFunction(), calloc_args, "");
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

void Compiler::Intrinsic(const stripe::Intrinsic& intrinsic, External handler) {
  // Process an intrinsic statement using an external handler function.
  // Load all the input scalars. Create a vector containing their types.
  // We will provide this as input to the handler, so the handler may perform
  // overloading if it so desires. The handler must replace the input types
  // with its own list of desired inputs. We will use this for argument count
  // verification, then cast each input scalar accordingly.
  std::vector<Scalar> inputs;
  std::vector<DataType> input_types;
  for (auto& input_name : intrinsic.inputs) {
    Scalar input = scalars_[input_name];
    inputs.push_back(input);
    input_types.push_back(input.type);
  }
  DataType output_type = intrinsic.type;
  // Call the handler. It will process the input and output types, then return
  // an entrypoint address. If it does not like the types, it may throw, or
  // return nullptr_t, in which case we will throw.
  auto funcptr = handler(&input_types, &output_type);
  if (!funcptr) {
    throw Error("External intrinsic rejected for " + intrinsic.name);
  }
  // Verify that we have the expected number of inputs. Cast each input to the
  // type specified by the handler.
  if (inputs.size() != input_types.size()) {
    throw Error("External intrinsic " + intrinsic.name + " expects " + std::to_string(input_types.size()) +
                " input(s), but the invocation " + "provided " + std::to_string(inputs.size()));
  }
  std::vector<llvm::Type*> argtypes(inputs.size());
  std::vector<llvm::Value*> argvals(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = Cast(inputs[i], input_types[i]);
    argvals[i] = inputs[i].value;
    assert(argvals[i]);
    argtypes[i] = CType(inputs[i].type);
  }
  // Build a function type signature for this list of input and output types
  llvm::Type* rtype = CType(output_type);
  auto functype = llvm::FunctionType::get(rtype, argtypes, false);
  // llvm::Type* fptrtype = functype->getPointerTo();
  // Embed the funcptr as a constant, cast to the relevant function type
  // llvm::Value* funcval = llvm::ConstantInt::get(fptrtype, (intptr_t)funcptr);
  // Generate a call to the funcptr
  std::string funcname = "external_" + intrinsic.name;
  external_funcptrs_[funcname] = funcptr;
  auto funcval = module_->getOrInsertFunction(funcname.c_str(), functype).getCallee();
  auto ret = builder_.CreateCall(funcval, argvals, "");
  // If we have an output, store the new scalar value.
  size_t expected_outputs = 0;
  if (output_type != DataType::INVALID) {
    expected_outputs = 1;
    scalars_[intrinsic.outputs[0]] = Scalar{ret, output_type};
  }
  // Verify that we have the expected number of outputs.
  if (expected_outputs != intrinsic.outputs.size()) {
    throw Error("External intrinsic " + intrinsic.name + " expects " + std::to_string(expected_outputs) +
                " output(s), but the invocation " + "provided " + std::to_string(intrinsic.outputs.size()));
  }
}

void Compiler::Add(const stripe::Intrinsic& add) {
  // Accepts two inputs, cast to operation type
  assert(2 == add.inputs.size());
  Scalar lhs = Cast(scalars_[add.inputs[0]], add.type);
  Scalar rhs = Cast(scalars_[add.inputs[1]], add.type);
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
  Scalar lhs = Cast(scalars_[sub.inputs[0]], sub.type);
  Scalar rhs = Cast(scalars_[sub.inputs[1]], sub.type);
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
  Scalar op = Cast(scalars_[neg.inputs[0]], neg.type);
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
  Scalar lhs = Cast(scalars_[mul.inputs[0]], mul.type);
  Scalar rhs = Cast(scalars_[mul.inputs[1]], mul.type);
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
  Scalar lhs = Cast(scalars_[div.inputs[0]], div.type);
  Scalar rhs = Cast(scalars_[div.inputs[1]], div.type);
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
  Scalar lhs = Cast(scalars_[mod.inputs[0]], mod.type);
  Scalar rhs = Cast(scalars_[mod.inputs[1]], mod.type);
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
  Scalar lhs = Cast(scalars_[lt.inputs[0]], lt.type);
  Scalar rhs = Cast(scalars_[lt.inputs[1]], lt.type);
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
    throw Error("Invalid comparison type (LT): " + to_string(lt.type));
  }
  OutputBool(ret, lt);
}

void Compiler::LessThanOrEqualTo(const stripe::Intrinsic& lte) {
  // Accepts two inputs
  assert(2 == lte.inputs.size());
  Scalar lhs = Cast(scalars_[lte.inputs[0]], lte.type);
  Scalar rhs = Cast(scalars_[lte.inputs[1]], lte.type);
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
    throw Error("Invalid comparison type (LE): " + to_string(lte.type));
  }
  OutputBool(ret, lte);
}

void Compiler::GreaterThan(const stripe::Intrinsic& gt) {
  // Accepts two inputs
  assert(2 == gt.inputs.size());
  Scalar lhs = Cast(scalars_[gt.inputs[0]], gt.type);
  Scalar rhs = Cast(scalars_[gt.inputs[1]], gt.type);
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
    throw Error("Invalid comparison type (GT): " + to_string(gt.type));
  }
  OutputBool(ret, gt);
}

void Compiler::GreaterThanOrEqualTo(const stripe::Intrinsic& gte) {
  // Accepts two inputs
  assert(2 == gte.inputs.size());
  Scalar lhs = Cast(scalars_[gte.inputs[0]], gte.type);
  Scalar rhs = Cast(scalars_[gte.inputs[1]], gte.type);
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
    throw Error("Invalid comparison type (GE): " + to_string(gte.type));
  }
  OutputBool(ret, gte);
}

void Compiler::Equal(const stripe::Intrinsic& eq) {
  // Accepts two inputs
  assert(2 == eq.inputs.size());
  Scalar lhs = Cast(scalars_[eq.inputs[0]], eq.type);
  Scalar rhs = Cast(scalars_[eq.inputs[1]], eq.type);
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
    throw Error("Invalid comparison type (EQ): " + to_string(eq.type));
  }
  OutputBool(ret, eq);
}

void Compiler::Unequal(const stripe::Intrinsic& neq) {
  // Accepts two inputs
  assert(2 == neq.inputs.size());
  Scalar lhs = Cast(scalars_[neq.inputs[0]], neq.type);
  Scalar rhs = Cast(scalars_[neq.inputs[1]], neq.type);
  // Inputs are cast to operation type
  // Equality placed into single output
  // Output type is boolean
  llvm::Value* ret = nullptr;
  if (is_float(neq.type)) {
    ret = builder_.CreateFCmpONE(lhs.value, rhs.value);
  } else if (is_int(neq.type) || is_uint(neq.type)) {
    ret = builder_.CreateICmpNE(lhs.value, rhs.value);
  } else if (DataType::BOOLEAN == neq.type) {
    ret = builder_.CreateICmpNE(lhs.value, rhs.value);
  } else {
    throw Error("Invalid comparison type (NE): " + to_string(neq.type));
  }
  OutputBool(ret, neq);
}

void Compiler::Conditional(const stripe::Intrinsic& cond) {
  // Three inputs: C, T, F; C is boolean, T and F are operation type
  assert(3 == cond.inputs.size());
  Scalar c = CheckBool(scalars_[cond.inputs[0]]);
  Scalar t = Cast(scalars_[cond.inputs[1]], cond.type);
  Scalar f = Cast(scalars_[cond.inputs[2]], cond.type);
  // Single output will be one of T or F
  // Output type is operation type
  llvm::Value* ret = builder_.CreateSelect(c.value, t.value, f.value);
  OutputType(ret, cond);
}

void Compiler::And(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  Scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  Scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateAnd(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Or(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  Scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  Scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateOr(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Not(const stripe::Intrinsic& stmt) {
  assert(1 == stmt.inputs.size());
  Scalar op = CheckBool(scalars_[stmt.inputs[0]]);
  llvm::Value* ret = builder_.CreateNot(op.value);
  OutputBool(ret, stmt);
}

void Compiler::Xor(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  Scalar lhs = CheckBool(scalars_[stmt.inputs[0]]);
  Scalar rhs = CheckBool(scalars_[stmt.inputs[1]]);
  llvm::Value* ret = builder_.CreateXor(lhs.value, rhs.value);
  OutputBool(ret, stmt);
}

void Compiler::Assign(const stripe::Intrinsic& stmt) {
  assert(1 == stmt.inputs.size());
  Scalar op = Cast(scalars_[stmt.inputs[0]], stmt.type);
  llvm::Value* ret = op.value;
  OutputType(ret, stmt);
}

void Compiler::BitLeft(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  Scalar lhs = Cast(scalars_[stmt.inputs[0]], stmt.type);
  Scalar rhs = Cast(scalars_[stmt.inputs[1]], stmt.type);
  llvm::Value* ret = builder_.CreateShl(lhs.value, rhs.value);
  OutputType(ret, stmt);
}

void Compiler::BitRight(const stripe::Intrinsic& stmt) {
  assert(2 == stmt.inputs.size());
  Scalar lhs = Cast(scalars_[stmt.inputs[0]], stmt.type);
  Scalar rhs = Cast(scalars_[stmt.inputs[1]], stmt.type);
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

void Compiler::Sqrt(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "sqrtf", "sqrt"); }

void Compiler::Exp(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "expf", "exp"); }

void Compiler::Log(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "logf", "log"); }

void Compiler::Pow(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "powf", "pow"); }

void Compiler::Tanh(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "tanhf", "tanh"); }

void Compiler::Cos(const stripe::Intrinsic& stmt) { CallIntrinsicFunc(stmt, "cosf", "cos"); }

void Compiler::Zero(const stripe::Special& zero) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation ZERO is not yet specified");
}

void Compiler::Copy(const stripe::Special& copy) {
  // present in stripe.proto but not defined in the specification
  throw Error("Special operation COPY is not yet specified");
}

void Compiler::Reshape(const stripe::Special& reshape) {
  assert(1 == reshape.inputs.size());
  Buffer src = buffers_[reshape.inputs[0]];
  assert(1 == reshape.outputs.size());
  Buffer dst = buffers_[reshape.outputs[0]];
  auto size = IndexConst(dst.refinement->interior_shape.byte_size());
  builder_.CreateMemCpy(dst.base, 0, src.base, 0, size);
}

void Compiler::PrngStep(const stripe::Special& prng_step) {
  // Input is a matrix of 3xN containing PRNG state.
  assert(1 == prng_step.inputs.size());
  Buffer in_state = buffers_[prng_step.inputs[0]];
  // Outputs are another matrix of PRNG state, and a buffer to be filled.
  // Output state shape must match input state shape.
  assert(2 == prng_step.outputs.size());
  Buffer out_state = buffers_[prng_step.outputs[0]];
  assert(out_state.refinement->interior_shape == in_state.refinement->interior_shape);
  Buffer dest = buffers_[prng_step.outputs[1]];
  llvm::Type* int32ptrType = builder_.getInt32Ty()->getPointerTo();
  llvm::Value* dest_arg = builder_.CreateBitCast(dest.base, int32ptrType);
  size_t dest_bytes = dest.refinement->interior_shape.byte_size();
  llvm::Value* count = IndexConst(dest_bytes / sizeof(uint32_t));
  std::vector<llvm::Value*> args{in_state.base, out_state.base, dest_arg, count};
  builder_.CreateCall(PrngStepFunction(), args, "");
}

Compiler::Scalar Compiler::Cast(Scalar v, DataType to_type) {
  if (v.type == to_type) {
    return v;
  }
  llvm::Type* to_llvmtype = CType(to_type);
  bool from_signed = is_int(v.type) || is_float(v.type);
  bool to_signed = is_int(to_type) || is_float(to_type);
  auto op = llvm::CastInst::getCastOpcode(v.value, from_signed, to_llvmtype, to_signed);
  llvm::Value* ret = builder_.CreateCast(op, v.value, to_llvmtype);
  return Scalar{ret, to_type};
}

Compiler::Scalar Compiler::CheckBool(Scalar v) {
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

llvm::Value* Compiler::ElementPtr(const Buffer& buf) {
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
  scalars_[intrinsic.outputs[0]] = Scalar{ret, intrinsic.type};
}

void Compiler::OutputBool(llvm::Value* ret, const stripe::Intrinsic& intrinsic) {
  assert(1 == intrinsic.outputs.size());
  scalars_[intrinsic.outputs[0]] = Scalar{ret, DataType::BOOLEAN};
}

void Compiler::CallIntrinsicFunc(const stripe::Intrinsic& stmt, const char* name_f32, const char* name_f64) {
  assert(1 == stmt.inputs.size());
  Scalar op = Cast(scalars_[stmt.inputs[0]], stmt.type);
  std::vector<llvm::Value*> argvals{op.value};
  // C intrinsics come in either f32 or f64 flavors. We'll use f32 for single
  // and half-precision float inputs, f64 for ints and doubles
  bool use_f32 = (stmt.type == DataType::FLOAT16 || stmt.type == DataType::FLOAT32);
  const char* name = use_f32 ? name_f32 : name_f64;
  llvm::Type* ctype = use_f32 ? builder_.getFloatTy() : builder_.getDoubleTy();
  std::vector<llvm::Type*> argtypes{ctype};
  auto functype = llvm::FunctionType::get(ctype, argtypes, false);
  auto func = module_->getOrInsertFunction(name, functype).getCallee();
  llvm::Value* ret = builder_.CreateCall(func, argvals, "");
  OutputType(ret, stmt);
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

llvm::Value* Compiler::XSMMDispatchFunction(llvm::Type* alphaBetaPrtrType, const std::string& functionName) {
  llvm::Type* iptr = llvm::Type::getInt32PtrTy(context_);
  std::vector<llvm::Type*> param_types{
      alphaBetaPrtrType,  // a
      alphaBetaPrtrType,  // b
      alphaBetaPrtrType,  // c
  };
  llvm::FunctionType* rftype = llvm::FunctionType::get(builder_.getVoidTy(), param_types, false);
  std::vector<llvm::Type*> argtypes{
      builder_.getInt32Ty(),  // m
      builder_.getInt32Ty(),  // n
      builder_.getInt32Ty(),  // k
      iptr,                   // lda
      iptr,                   // ldb
      iptr,                   // ldc
      alphaBetaPrtrType,      // alpha
      alphaBetaPrtrType,      // beta
      iptr,                   // flags
      iptr,                   // prefetch
  };
  auto functype = llvm::FunctionType::get(rftype->getPointerTo(), argtypes, false);
  return module_->getOrInsertFunction(functionName.c_str(), functype).getCallee();
}

llvm::Value* Compiler::MallocFunction(void) {
  std::vector<llvm::Type*> argtypes{IndexType()};
  llvm::Type* rettype = builder_.getInt8PtrTy();
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "malloc";
  return module_->getOrInsertFunction(funcname, functype).getCallee();
}

llvm::Value* Compiler::CallocFunction(void) {
  std::vector<llvm::Type*> argtypes{IndexType(), IndexType()};
  llvm::Type* rettype = builder_.getInt8PtrTy();
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "calloc";
  return module_->getOrInsertFunction(funcname, functype).getCallee();
}

llvm::Value* Compiler::FreeFunction(void) {
  llvm::Type* ptrtype = builder_.getInt8PtrTy();
  std::vector<llvm::Type*> argtypes{ptrtype};
  llvm::Type* rettype = llvm::Type::getVoidTy(context_);
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "free";
  return module_->getOrInsertFunction(funcname, functype).getCallee();
}

llvm::Value* Compiler::PrngStepFunction(void) {
  llvm::Type* int32ptrType = builder_.getInt32Ty()->getPointerTo();
  std::vector<llvm::Type*> argtypes{int32ptrType, int32ptrType, int32ptrType, IndexType()};
  llvm::Type* rettype = llvm::Type::getVoidTy(context_);
  auto functype = llvm::FunctionType::get(rettype, argtypes, false);
  const char* funcname = "prng_step";
  return module_->getOrInsertFunction(funcname, functype).getCallee();
}

Executable::Executable(const ProgramModule& module) : parameters_(module.parameters) {
  std::string errStr;
  std::unique_ptr<llvm::LegacyJITSymbolResolver> rez(new Runtime(module.externals));
  assert(module.module);
  std::unique_ptr<llvm::Module> clone(llvm::CloneModule(*module.module));
  auto ee = llvm::EngineBuilder(std::move(clone))
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
    args[i] = safe_at(buffers, parameters_[i]);
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
void prng_step(uint32_t* in_state, uint32_t* out_state, uint32_t* buf, size_t count) {
  // A reimplementation of the PRNG from tile/lang/gen_special.cc.
  // x_n = (s1_n ^ s2_n ^ s3_n)
  // s1_{n+1} = (((s1_n & 4294967294) <<12) ^ (((s1_n <<13) ^ s1_n) >>19))
  // s2_{n+1} = (((s2_n & 4294967288) << 4) ^ (((s2_n << 2) ^ s2_n) >>25))
  // s3_{n+1} = (((s3_n & 4294967280) <<17) ^ (((s3_n << 3) ^ s3_n) >>11))
  for (size_t i = 0; i < count; ++i) {
    buf[i] = in_state[0] ^ in_state[1] ^ in_state[2];
    out_state[0] = (((in_state[0] & 4294967294) << 12) ^ (((in_state[0] << 13) ^ in_state[0]) >> 19));
    out_state[1] = (((in_state[1] & 4294967288) << 4) ^ (((in_state[1] << 2) ^ in_state[1]) >> 25));
    out_state[2] = (((in_state[2] & 4294967280) << 17) ^ (((in_state[2] << 3) ^ in_state[2]) >> 11));
    in_state = out_state;
  }
}
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
      {"prng_step", symInfo(rt::prng_step)},
      {"_prng_step", symInfo(rt::prng_step)},
      {"libxsmm_smmdispatch", symInfo(libxsmm_smmdispatch)},
      {"_libxsmm_smmdispatch", symInfo(libxsmm_smmdispatch)},
      {"libxsmm_dmmdispatch", symInfo(libxsmm_dmmdispatch)},
      {"_libxsmm_dmmdispatch", symInfo(libxsmm_dmmdispatch)},
  };
  auto loc_rt = symbols.find(name);
  if (loc_rt != symbols.end()) {
    return loc_rt->second;
  }
  auto loc_extern = externals_.find(name);
  if (loc_extern != externals_.end()) {
    return symInfo(loc_extern->second);
  }
  if (name.size() > 1 && name[0] == '_') {
    loc_extern = externals_.find(name.substr(1));
    if (loc_extern != externals_.end()) {
      return symInfo(loc_extern->second);
    }
  }
  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name);
  // If we failed to resolve the symbol, and its first character is an
  // underscore, try again without the underscore. The code may have been
  // generated for a system whose loader expects every symbol to have an
  // underscore prefix, but the DynamicLibrary module expects no prefix.
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

struct Native::Impl {
  llvm::LLVMContext context;
  ProgramModule module;
  std::unique_ptr<Executable> executable;

  void compile(const stripe::Block& program, const std::map<std::string, External>& externals) {
    Compiler compiler(&context, externals);
    module = compiler.CompileProgram(program);
    assert(module.module);
    executable.reset(new Executable(module));
  }

  void run(const std::map<std::string, void*>& buffers) { executable->Run(buffers); }

  void save(const std::string& filename) {
    std::error_code ec;
    llvm::ToolOutputFile result(filename, ec, llvm::sys::fs::F_None);
    WriteBitcodeToFile(*module.module, result.os());
    result.keep();
  }
};

Native::Native() : m_impl(new Native::Impl) {}
Native::~Native() {}
void Native::compile(const stripe::Block& program, const std::map<std::string, External>& externals) {
  m_impl->compile(program, externals);
}
void Native::run(const std::map<std::string, void*>& buffers) { m_impl->run(buffers); }
void Native::save(const std::string& filename) { m_impl->save(filename); }

void JitExecute(const stripe::Block& program, const std::map<std::string, void*>& buffers) {
  llvm::LLVMContext context;
  std::map<std::string, External> externals;
  Compiler compiler(&context, externals);
  auto module = compiler.CompileProgram(program);
  Executable executable(std::move(module));
  executable.Run(buffers);
}

void JitExecute(const stripe::Block& program, const std::map<std::string, External>& externals,
                const std::map<std::string, void*>& buffers) {
  llvm::LLVMContext context;
  Compiler compiler(&context, externals);
  auto module = compiler.CompileProgram(program);
  Executable executable(std::move(module));
  executable.Run(buffers);
}

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
