// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::dialect::pxa {

namespace {

struct LocalizePass : public LocalizeBase<LocalizePass> {
  void runOnFunction() final {
    //
  }
};

} // namespace

std::unique_ptr<Pass> createLocalizePass() {
  return std::make_unique<LocalizePass>();
}

} // namespace pmlc::dialect::pxa
