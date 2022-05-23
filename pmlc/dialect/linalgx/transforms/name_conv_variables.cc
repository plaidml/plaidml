// Copyright 2022 Intel Corporation
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/linalgx/analysis/convolution.h"
#include "pmlc/dialect/linalgx/transforms/name_conv_variables.h"

#include "pmlc/dialect/linalgx/transforms/pass_detail.h"
#include "pmlc/util/logging.h"
using namespace mlir; // NOLINT

static constexpr StringLiteral nameAttribute = "convolution_indices";
namespace pmlc::dialect::linalgx {

void setNameAttr(Operation *op, std::vector<std::string> tags) {
  if (tags.empty())
    return;

  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(nameAttribute);
  for (int i = 0; i < tags.size(); i++) {
    dict.set(tags[i], UnitAttr::get(op->getContext()));
  }
  op->setAttr(nameAttribute, dict.getDictionary(op->getContext()));
}

struct NameConvVariablesPass
    : public NameConvVariablesBase<NameConvVariablesPass> {
  NameConvVariablesPass() = default;
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](linalg::GenericOp op) { nameConvVariables(op); });
  }

  void maybeCaptureOp(linalg::GenericOp op) {
    auto capturedConv = detectConv(op);
    if (!capturedConv) {
      IVLOG(3, "Cannot label non conv operator " << debugString(op));
      return;
    }
    if (capturedConv->input.idxMap.getNumResults() != 4 ||
        capturedConv->filter.idxMap.getNumResults() != 4 ||
        capturedConv->output.idxMap.getNumResults() != 4) {
      IVLOG(3, "Cannot label non 2d conv operator " << debugString(op));
      return;
    }
    std::vector<std::string> tagsList;
    std::string prefixes[7] = {"N:", "H:", "W:", "C:", "K:", "R:", "S:"};
    auto shapesToLoopsMap = op.getShapesToLoopsMap();
    for (int i = 0; i < shapesToLoopsMap.getNumResults(); i++) {
      AffineExpr term = shapesToLoopsMap.getResult(i);
      AffineDimExpr termDimExpr = term.dyn_cast<AffineDimExpr>();
      unsigned pos = termDimExpr.getPosition();
      std::stringstream stream(prefixes[i]);
      stream << "d" << pos;
      tagsList.push_back(stream.str());
    }
    setNameAttr(op, tagsList);
  }
  void nameConvVariables(linalg::GenericOp op) { maybeCaptureOp(op); }
};

std::unique_ptr<mlir::Pass> createNameConvVariablesPass() {
  return std::make_unique<NameConvVariablesPass>();
}

} // namespace pmlc::dialect::linalgx
