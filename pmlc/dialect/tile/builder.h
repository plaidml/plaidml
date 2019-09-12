// Copyright 2019, Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "tile/base/shape.h"

namespace mlir {
class Value;
class Operation;
}  // namespace mlir

namespace pmlc {
namespace dialect {
namespace tile {

using DataType = vertexai::tile::DataType;

struct Shape {
  DataType elementType;
  llvm::ArrayRef<int64_t> dims;
};

struct TileProgram;

class TileBuilder {
  struct Impl;

 public:
  TileBuilder();
  ~TileBuilder();

  void Destroy(mlir::Value* value);

  void BindTensorDim(unsigned dim, mlir::Value* from, mlir::Value** into);
  Shape GetShape(mlir::Value* tensor);
  std::vector<mlir::Value*> GetTupleElements(mlir::Value* value);
  std::vector<mlir::Value*> ComputeGradients(llvm::ArrayRef<mlir::Value*> wrt, mlir::Value* loss);
  mlir::Value* Clone(mlir::Value* value);
  mlir::Value* MakeNoneOp();
  mlir::Value* MakeStringOp(llvm::StringRef value);
  mlir::Value* MakeTupleOp(llvm::ArrayRef<mlir::Value*> elts);
  mlir::Value* MakeScalarConstantOp(int64_t value);
  mlir::Value* MakeScalarConstantOp(double value);
  mlir::Value* MakePrimitiveOp(llvm::StringRef fn, llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeDimOp(mlir::Value* tensor, unsigned dim);
  mlir::Value* MakePlaceholderOp(DataType dtype, llvm::ArrayRef<int64_t> dims);
  mlir::Value* MakeAffineConstantOp(int64_t value);
  mlir::Value* MakeAffineIndexOp(llvm::StringRef name = "");
  mlir::Value* MakeAffineAddOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineSubOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineMulOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineDivOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineNegOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineSourceIndexMapOp(mlir::Value* tensor, llvm::ArrayRef<mlir::Value*> idxs);
  mlir::Value* MakeAffineSinkIndexMapOp(llvm::ArrayRef<mlir::Value*> idxs);
  mlir::Value* MakeAffineSizeMapOp(llvm::ArrayRef<mlir::Value*> sizes);

  mlir::Value* MakeConAssignOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConAssignAddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConAssignCondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConAssignEqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConAssignMulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);

  mlir::Value* MakeConMaxOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMaxAddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMaxCondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMaxEqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMaxMulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);

  mlir::Value* MakeConMinOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMinAddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMinCondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMinEqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConMinMulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);

  mlir::Value* MakeConProdOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConProdAddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConProdCondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConProdEqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConProdMulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);

  mlir::Value* MakeConSumOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConSumAddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConSumCondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConSumEqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);
  mlir::Value* MakeConSumMulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes);

  std::shared_ptr<TileProgram> MakeProgram(  //
      llvm::StringRef name,                  //
      llvm::ArrayRef<mlir::Value*> outputs,  //
      llvm::MutableArrayRef<mlir::Value*> new_outputs);

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
