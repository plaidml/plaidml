// Copyright 2019, Intel Corporation

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/enums.h"

namespace mlir {
class Value;
class Operation;
}  // namespace mlir

namespace pmlc::dialect::tile {

using DataType = util::DataType;

struct Shape {
  DataType elementType;
  llvm::ArrayRef<int64_t> dims;
};

struct ProgramUpdate {
  mlir::Value source;
  mlir::Value target;

  bool operator<(const ProgramUpdate& rhs) const {
    auto lhs_src_ptr = source.getAsOpaquePointer();
    auto lhs_tgt_ptr = target.getAsOpaquePointer();
    auto rhs_src_ptr = rhs.source.getAsOpaquePointer();
    auto rhs_tgt_ptr = rhs.target.getAsOpaquePointer();
    return std::tie(lhs_src_ptr, lhs_tgt_ptr) < std::tie(rhs_src_ptr, rhs_tgt_ptr);
  }
};

struct ProgramMutations {
  std::vector<mlir::Value> outputs;
  std::set<ProgramUpdate> updates;
  DataType floatx = DataType::invalid;
  DataType intx = DataType::invalid;

  void set_floatx(const char* floatx_str) {
    floatx = util::from_string(floatx_str);
    if (!isFloat(floatx)) {
      throw std::runtime_error("Invalid floatx string " + std::string(floatx_str));
    }
  }
  void set_intx(const char* intx_str) {
    intx = util::from_string(intx_str);
    if (!isInteger(intx)) {
      throw std::runtime_error("Invalid intx string " + std::string(intx_str));
    }
  }
};

class TileBuilder {
  struct Impl;

 public:
  TileBuilder();
  ~TileBuilder();

  void Destroy(mlir::Value value);

  mlir::RankedTensorType MakeRankedTensorType(DataType dtype, llvm::ArrayRef<int64_t> dims);
  void BindTensorDims(mlir::Value from, llvm::ArrayRef<mlir::Value*> into);
  mlir::RankedTensorType ComputeShape(mlir::Value tensor);
  void BindShape(mlir::Value tensor, mlir::RankedTensorType type);
  void BindBuffer(mlir::Value tensor, pmlc::util::BufferPtr buffer);

  mlir::MemRefType MakeMemRefType(DataType dtype, llvm::ArrayRef<int64_t> sizes, llvm::ArrayRef<int64_t> strides);
  mlir::MemRefType IntoMemRefType(mlir::RankedTensorType type);

  llvm::StringRef GetStringValue(mlir::Value value);
  int64_t GetIntegerValue(mlir::Value value);
  double GetFloatValue(mlir::Value value);
  std::vector<mlir::Value> GetTupleElements(mlir::Value value);

  std::vector<mlir::Value> ComputeGradients(llvm::ArrayRef<mlir::Value> wrt, mlir::Value loss);
  mlir::Value Clone(mlir::Value value);

  mlir::Value MakeNoneOp();
  mlir::Value MakeStringOp(llvm::StringRef value);
  mlir::Value MakeTupleOp(llvm::ArrayRef<mlir::Value> elts);

  mlir::Value MakeScalarConstantOp(int64_t value);
  mlir::Value MakeScalarConstantOp(double value);
  mlir::Value MakePrimitiveOp(llvm::StringRef fn, llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeCastOp(mlir::Value tensor, DataType dtype);
  mlir::Value MakeTraceOp(mlir::Value tensor, const char* msg);
  mlir::Value MakeDimOp(mlir::Value tensor, unsigned dim);
  mlir::Value MakePlaceholderOp(mlir::RankedTensorType type, pmlc::util::BufferPtr buffer, llvm::StringRef name);
  mlir::Value MakeAffineConstantOp(int64_t value);
  mlir::Value MakeAffineIndexOp(llvm::StringRef name = "");
  mlir::Value MakeAffineAddOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineSubOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineMulOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineDivOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineNegOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineMaxOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineMinOp(llvm::ArrayRef<mlir::Value> args);
  mlir::Value MakeAffineTensorMapOp(mlir::Value tensor, llvm::ArrayRef<mlir::Value> idxs);
  mlir::Value MakeAffineMapOp(llvm::ArrayRef<mlir::Value> idxs);

  mlir::Value MakeContractionOp(         //
      util::AggregationKind agg,         //
      util::CombinationKind combo,       //
      llvm::ArrayRef<mlir::Value> srcs,  //
      mlir::Value sink,                  //
      mlir::Value sizes,                 //
      llvm::StringRef name);

  void AddConstraint(mlir::Value cion, mlir::Value lhs, mlir::Value rhs);
  void SetUseDefault(mlir::Value cion, mlir::Value defaultValue);
  void SetNoReduce(mlir::Value cion, bool no_reduce);

  std::shared_ptr<TileProgram> MakeProgram(llvm::StringRef name, const ProgramMutations& mutations);

  void Dump();

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace pmlc::dialect::tile
