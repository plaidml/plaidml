// Copyright 2018, Intel Corp.

#pragma once

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/config.h"
#include "tile/targets/cpu/programmodule.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {

enum class XSMMDispatch : int {
  NONE = 0,  // No XSMM dispatch function to call.
  SMM = 1,   // singe float
  DMM = 2,   // double float
  WIMM = 3,  // int8, uint8 ---> int
  BSMM = 4,  // TODO: Need Stripe support for bfloat16.
  BMMM = 5,  // TODO: Need Stripe support for bfloat16.
};

// What block we are compiling.
enum CompileFor {
  NORMAL_BLOCK,
  THREADED_BLOCK,
  XSMM_BLOCK,
};

class Compiler : private stripe::ConstStmtVisitor {
 public:
  Compiler(llvm::LLVMContext* context, const Config& config);
  ProgramModule CompileProgram(const stripe::Block& program);

  // Internal data type definitions.
 private:
  struct XSMMCallData {
    XSMMCallData() : in0(nullptr), in1(nullptr), out0(nullptr), lda_a_value(0), lda_b_value(0),
                     lda_c_value(0), offset_in0(0), offset_in1(0), offset_out0(0) {}

    const stripe::Refinement* in0;
    const stripe::Refinement* in1;
    const stripe::Refinement* out0;

    int32_t lda_a_value;
    int32_t lda_b_value;
    int32_t lda_c_value;

    int32_t offset_in0;
    int32_t offset_in1;
    int32_t offset_out0;
  };

 protected:
  explicit Compiler(llvm::LLVMContext* context, llvm::Module* module, const Config& config);
  void GenerateInvoker(const stripe::Block& program, llvm::Function* main);
  uint64_t MeasureArena(const stripe::Block& block);
  void GenerateArena(const stripe::Block& block);
  llvm::Function* CompileXSMMBlock(const stripe::Block& block, const XSMMDispatch xsmmDispatch,
                                   const XSMMCallData& xsmmCallData);
  llvm::Function* CompileThreadedBlock(const stripe::Block& block);
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
  void Floor(const stripe::Intrinsic&);
  void Ceil(const stripe::Intrinsic&);
  void Round(const stripe::Intrinsic&);
  void Abs(const stripe::Intrinsic&);
  void Acos(const stripe::Intrinsic&);
  void Asin(const stripe::Intrinsic&);
  void Atan(const stripe::Intrinsic&);
  void Acosh(const stripe::Intrinsic&);
  void Asinh(const stripe::Intrinsic&);
  void Atanh(const stripe::Intrinsic&);
  void Cosh(const stripe::Intrinsic&);
  void Sin(const stripe::Intrinsic&);
  void Sinh(const stripe::Intrinsic&);
  void Tan(const stripe::Intrinsic&);
  void Zero(const stripe::Special&);
  void Copy(const stripe::Special&);
  void Reshape(const stripe::Special&);
  void PrngStep(const stripe::Special&);
  void Shape(const stripe::Special&);
  void AggInitAdd(const stripe::Special&);
  void AggInitMul(const stripe::Special&);
  void AggInitMin(const stripe::Special&);
  void AggInitMax(const stripe::Special&);
  void Scatter(const stripe::Special&);
  void Gather(const stripe::Special&);
  void AsFloat(const stripe::Intrinsic&);
  void AsInt(const stripe::Intrinsic&);
  void AsUInt(const stripe::Intrinsic&);
  void AsBool(const stripe::Intrinsic&);

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
  void CreateLoop(Loop* loop, std::string name);
  void EnterLoop(Loop* loop, llvm::Value* variable, llvm::Value* init, llvm::Value* limit);
  void LeaveLoop(Loop* loop, llvm::Value* variable);
  Scalar Cast(Scalar, DataType);
  Scalar CheckBool(Scalar);
  llvm::Type* CType(DataType);
  llvm::Value* ElementPtr(const Buffer& buf);
  llvm::Value* Eval(const stripe::Affine& access);
  void OutputType(llvm::Value* ret, const stripe::Intrinsic&);
  void OutputBool(llvm::Value* ret, const stripe::Intrinsic&);
  void CallIntrinsicFunc(const stripe::Intrinsic&, const char* name_f32, const char* name_f64,
                         const size_t numParams = 1);
  llvm::Type* IndexType();
  llvm::Value* IndexConst(ssize_t val);
  llvm::FunctionType* BlockType(const stripe::Block&);
  llvm::Value* XSMMDispatchFunction(llvm::Type* alphaPtrType, llvm::Type* betaPtrType, llvm::Type* aPtrType,
                                    llvm::Type* bPtrType, llvm::Type* cPtrType, const std::string& funcionName);
  llvm::Value* Malloc(size_t size);
  void Free(llvm::Value* buffer);
  llvm::Value* PrngStepFunction();
  llvm::Value* ReadCycleCounter();
  void ProfileBlockEnter(const stripe::Block& block);
  void ProfileBlockLeave(const stripe::Block& block);
  void ProfileLoopEnter(const stripe::Block& block);
  void ProfileLoopLeave(const stripe::Block& block);
  std::string ProfileBlockID(const stripe::Block& block);
  const XSMMDispatch GetXSMMDispatch(const stripe::Block& block);
  llvm::Value* RunTimeLogEntry(void);
  void EmitRunTimeLogEntry(const std::string& str, const std::string& extra, llvm::Value* value = nullptr);
  void PrintOutputAssembly();
  void AggInit(const Buffer& dest, llvm::Value* init_val);
  void ParallelFor(llvm::Value* refs, llvm::Value* idxs, size_t range, llvm::Function* func);
  CompileFor getCompileFor(const stripe::Block& block);

  // Gets the leading dimensions and the buffers for an XSMM call if available.
  // @returns true if the XSMM call is applicable, otherwise false.
  bool GetXSMMCallData(XSMMCallData* xsmmCallData, const stripe::Block& block);

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  llvm::Module* module_ = nullptr;
  Config config_;
  std::map<std::string, void*> external_funcptrs_;

  std::map<std::string, Scalar> scalars_;
  std::map<std::string, Buffer> buffers_;
  std::map<std::string, Index> indexes_;
  uint64_t arenaSize_ = 0;
};

}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai
