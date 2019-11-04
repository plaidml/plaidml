// Copyright 2019, Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/enums.h"
#include "tile/base/shape.h"

namespace mlir {
class Value;
class Operation;
}  // namespace mlir

namespace pmlc::dialect::tile {

using DataType = vertexai::tile::DataType;

struct Shape {
  DataType elementType;
  llvm::ArrayRef<int64_t> dims;
};

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
  mlir::Value* MakePlaceholderOp(DataType dtype, llvm::ArrayRef<int64_t> dims, vertexai::tile::BufferPtr buffer,
                                 llvm::StringRef name);
  mlir::Value* MakeAffineConstantOp(int64_t value);
  mlir::Value* MakeAffineIndexOp(llvm::StringRef name = "");
  mlir::Value* MakeAffineAddOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineSubOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineMulOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineDivOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineNegOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineMaxOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineMinOp(llvm::ArrayRef<mlir::Value*> args);
  mlir::Value* MakeAffineSourceIndexMapOp(mlir::Value* tensor, llvm::ArrayRef<mlir::Value*> idxs);
  mlir::Value* MakeAffineSinkIndexMapOp(llvm::ArrayRef<mlir::Value*> idxs);
  mlir::Value* MakeAffineSizeMapOp(llvm::ArrayRef<mlir::Value*> sizes);

  mlir::Value* MakeContractionOp(         //
      util::AggregationKind agg,          //
      util::CombinationKind combo,        //
      llvm::ArrayRef<mlir::Value*> srcs,  //
      mlir::Value* sink,                  //
      mlir::Value* sizes,                 //
      llvm::StringRef name);

  void AddConstraint(mlir::Value* cion, mlir::Value* lhs, mlir::Value* rhs);
  void SetUseDefault(mlir::Value* cion, mlir::Value* defaultValue);
  void SetNoReduce(mlir::Value* cion, bool no_reduce);

  std::shared_ptr<TileProgram> MakeProgram(  //
      llvm::StringRef name,                  //
      llvm::ArrayRef<mlir::Value*> outputs,  //
      llvm::MutableArrayRef<mlir::Value*> new_outputs);

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace pmlc::dialect::tile
