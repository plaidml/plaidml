// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Emit : public sem::Visitor {
 public:
  Emit();
  void Visit(const sem::IntConst&) override;
  void Visit(const sem::FloatConst&) override;
  void Visit(const sem::LookupLVal&) override;
  void Visit(const sem::LoadExpr&) override;
  void Visit(const sem::StoreStmt&) override;
  void Visit(const sem::SubscriptLVal&) override;
  void Visit(const sem::DeclareStmt&) override;
  void Visit(const sem::UnaryExpr&) override;
  void Visit(const sem::BinaryExpr&) override;
  void Visit(const sem::CondExpr&) override;
  void Visit(const sem::SelectExpr&) override;
  void Visit(const sem::ClampExpr&) override;
  void Visit(const sem::CastExpr&) override;
  void Visit(const sem::CallExpr&) override;
  void Visit(const sem::LimitConst&) override;
  void Visit(const sem::IndexExpr&) override;
  void Visit(const sem::Block&) override;
  void Visit(const sem::IfStmt&) override;
  void Visit(const sem::ForStmt&) override;
  void Visit(const sem::WhileStmt&) override;
  void Visit(const sem::BarrierStmt&) override;
  void Visit(const sem::ReturnStmt&) override;
  void Visit(const sem::Function&) override;
  std::string str() const;
  std::unique_ptr<llvm::Module>&& result() { return std::move(module_); }

 private:
  struct value {
    explicit value(llvm::Value* v_in = nullptr, sem::Type t_in = sem::Type{sem::Type::TVOID, DataType::INVALID})
        : v{v_in}, t{t_in} {}
    llvm::Value* v;
    sem::Type t;
  };

  struct block {
    std::map<std::string, value> symbols;
    llvm::BasicBlock* doneblock = nullptr;  // optional
    llvm::BasicBlock* testblock = nullptr;  // optional
  };

  typedef std::deque<block> block_stack;

  value Process(const sem::Node&);
  value Eval(sem::ExprPtr e);
  value LVal(sem::LValPtr l);
  void Resolve(value);
  llvm::Type* CType(const sem::Type&);
  llvm::Value* Define(const std::string& name, sem::Type type);
  value Lookup(const std::string& name);
  llvm::Value* CastTo(value, sem::Type);
  llvm::Value* ToBool(llvm::Value*);
  void Enter();
  void EnterLoop(llvm::BasicBlock* done, llvm::BasicBlock* check);
  void Leave();
  void LimitConstSInt(unsigned bits, sem::LimitConst::Which);
  void LimitConstUInt(unsigned bits, sem::LimitConst::Which);
  void LimitConstFP(const llvm::fltSemantics&, sem::LimitConst::Which);
  bool CurrentBlockIsTerminated();
  static bool IsUnsignedIntegerType(const sem::Type&);
  static bool IsFloatingPointType(const sem::Type&);
  bool PointerAddition(value left, value right);
  sem::Type CommonType(const value& left, const value& right);
  sem::Type ConvergeOperands(value* left, value* right);
  static std::string print(const llvm::Type*);

  llvm::LLVMContext& context_;
  llvm::IRBuilder<> builder_;
  std::unique_ptr<llvm::Module> module_;
  std::unique_ptr<llvm::legacy::FunctionPassManager> funcopt_;
  llvm::IntegerType* int32type_ = nullptr;
  llvm::IntegerType* booltype_ = nullptr;
  llvm::IntegerType* ssizetype_ = nullptr;
  llvm::ArrayType* gridSizeType_ = nullptr;
  llvm::Function* function_ = nullptr;
  llvm::Value* workIndex_ = nullptr;
  llvm::Function* barrier_func_ = nullptr;
  std::map<std::string, llvm::Function*> builtins_;
  block_stack blocks_;
  value result_;
  // What is the current function's return value type? This will be populated
  // on entry to the Function visitor and cleared on exit, so it can be used
  // during nested evaluations.
  sem::Type returntype_{sem::Type::TVOID};
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
