// Copyright 2020, Intel Corporation

namespace mlir {
class OpPassManager;
}  // namespace mlir

namespace pmlc::compiler {

// Target encapsulates the compiler's support for a particular
// compilation target.
class Target {
 public:
  virtual void addPassesToPipeline(mlir::OpPassManager* mgr) = 0;
};

}  // namespace pmlc::compiler
