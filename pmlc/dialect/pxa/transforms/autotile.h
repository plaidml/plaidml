#pragma once

#include <memory>
#include <vector>

#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/tile.h"

namespace pmlc::dialect::pxa {

// Abstract tile generator class
class TileSizeGenerator {
 public:
  virtual bool initialize(int64_t newRange) = 0;
  virtual int64_t current() const = 0;
  virtual bool next() = 0;
};

// A partially concrete base class that fits most cases, and only requires implementing
// the next function
class BaseTileSizeGenerator : public TileSizeGenerator {
 public:
  bool initialize(int64_t newRange) override {
    range = newRange;
    cur = 1;
    return true;
  }
  int64_t current() const override { return cur; }

 protected:
  int64_t range;
  int64_t cur;
};

class AllTilesGenerator : public BaseTileSizeGenerator {
 public:
  bool next() override {
    cur++;
    return cur <= range;
  }
};

class PowerOfTwoGenerator : public BaseTileSizeGenerator {
 public:
  bool next() override {
    cur *= 2;
    return cur <= range;
  }
};

class EvenTilingGenerator : public BaseTileSizeGenerator {
 public:
  bool next() override {
    if (cur == range) {
      return false;
    }
    cur++;
    while (range % cur != 0) cur++;
    return true;
  }
};

class ExactRangeGenerator : public TileSizeGenerator {
 public:
  bool initialize(int64_t newRange) override {
    range = newRange;
    return true;
  }
  int64_t current() const override { return range; }
  bool next() override { return false; }

 private:
  int64_t range;
};

class FixedTileSizeGenerator : public TileSizeGenerator {
 public:
  explicit FixedTileSizeGenerator(int64_t size) : size(size) {}
  bool initialize(int64_t newRange) override { return true; }
  int64_t current() const override { return size; }
  bool next() override { return false; }

 private:
  int64_t size;
};

class TileCostModel {
  // Return false to skip optimizing this PF, otherwise prepare
  virtual bool prepare(AffineParallelOp op) = 0;
  // For a given tile size, return a cost, or +inf if not feasible
  virtual double computeCost(ArrayRef<int64_t> tile) const = 0;
};

// Always returns a cost of 1
class DummyCostModel : public TileCostModel {
  // Return false to skip optimizing this PF, otherwise prepare
  bool prepare(AffineParallelOp op) override { return true; }
  // For a given tile size, return a cost, or +inf if not feasible
  double computeCost(ArrayRef<int64_t> tile) const override { return 1.0; }
};

struct AutotilePass : public mlir::FunctionPass<AutotilePass> {
  AutotilePass() {}
  AutotilePass(std::shared_ptr<TileSizeGenerator> generator, std::shared_ptr<TileCostModel> costModel)
      : generator(generator), costModel(costModel) {}
  void runOnFunction() override {
    auto func = this->getFunction();
    func.walk([](AffineParallelOp op) { Tile(op, {10, 10, 10}); });
  }
  std::shared_ptr<TileSizeGenerator> generator;
  std::shared_ptr<TileCostModel> costModel;
};

}  // namespace pmlc::dialect::pxa
