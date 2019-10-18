// Copyright 2019 Intel Corporation.

#include "pmlc/dialect/tile/gradient.h"

// TODO: Clean includes
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/dialect/tile/builder.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/util/slice.h"

namespace pmlc::dialect::tile {

Gradient::Gradient(mlir::Value* loss, TileBuilder* builder) : builder_(builder) {
  IVLOG(3, "Gradient::Gradient> loss: " << mlir::debugString(*loss));
  grads_[loss] = builder_->MakeScalarConstantOp(1.0);
  llvm::SetVector<mlir::Value*> loss_setvec;
  loss_setvec.insert(loss);
  auto defs = util::getBackwardSlice(loss_setvec, false, std::function<bool(mlir::Value*)>{[](mlir::Value* val) {
                                       auto op = val->getDefiningOp();
                                       // TODO: This is an ad hoc list of what to filter out; make it principled
                                       return !mlir::isa<AffineConstraintsOp>(op) && !mlir::isa<AffineMapOp>(op) &&
                                              !mlir::isa<AffineIndexOp>(op) && !mlir::isa<DimOp>(op) &&
                                              !mlir::isa<eltwise::ScalarConstantOp>(op);
                                     }});
  for (auto def = defs.rbegin(); def != defs.rend(); def++) {
    ComputeOperandDerivs(*def);
  }
  if (VLOG_IS_ON(5)) {
    IVLOG(5, "Gradient::Gradient> Computed the following gradients: ");
    for (auto [key, value] : grads_) {
      IVLOG(5, "  key is " << mlir::debugString(*key) << "\n    for val " << mlir::debugString(*value));
    }
  }
}

void Gradient::AddToGradient(Value* source_op, Value* deriv) {
  // Adds the gradient `deriv` computed for one use of `source_op` to the overall gradient of `source_op`
  if (!grads_.count(source_op)) {
    grads_[source_op] = deriv;
  } else {
    grads_[source_op] = builder_->MakePrimitiveOp("add", {grads_[source_op], deriv});
  }
}

void Gradient::ComputeOperandDerivs(mlir::Value* val) {
  IVLOG(4, "Gradient::ComputeDerivative> " << mlir::debugString(*val));
  // TODO: Throw on ops with multiple results?
  auto op = val->getDefiningOp();
  if (mlir::isa<AffineConstraintsOp>(op) || mlir::isa<AffineMapOp>(op) || mlir::isa<AffineIndexOp>(op)) {
    // TODO: Make the list of which ops these are more principled. Also, should these all be caught in the backwards
    // slice filter? If so, probably throw here.
    IVLOG(6, "Gradient::ComputeDerivative> Skipping computing derivatives for op "
                 << mlir::debugString(*op) << ", as it is not a type of op for which gradients apply");
    return;
  }
  if (!grads_.count(val)) {
    IVLOG(1, "Gradient::ComputeOperandDerivs> Called on Value which has not itself been differentiated: "
                 << mlir::debugString(*val));
    throw std::runtime_error("Unexpected missing derivative in ComputeOperandDerivs");
  }
  if (mlir::isa<eltwise::EltwiseOp>(op)) {
    size_t idx = 0;  // Need to track which operand we're at
    for (const auto& operand : op->getOperands()) {
      auto dop = DeriveEltwise(grads_[val], val, idx);
      AddToGradient(operand, dop);
      idx++;
    }
    // TODO: if grads_[operand] has rank > 0, call a simple_reduce
  } else if (auto cion_op = mlir::dyn_cast<SymbolicContractionOp>(op)) {
    {
      // TODO: Do `init` deriv right: we should passthrough on plus, conditional against result on max/min, throw on
      // product
      // TODO: This is a temporary hack!
      AddToGradient(cion_op.init(), grads_[val]);
    }
    size_t idx = 0;  // Need to track which operand we're at
    for (auto src : cion_op.srcs()) {
      auto dop = DeriveContraction(grads_[val], val, idx);
      AddToGradient(src, dop);
      idx++;
    }
  } else if (mlir::isa<SpecialOp>(op)) {
    // TODO: Possibly can merge w/ EltwiseOp case?
    throw std::runtime_error("TODO: Derivs of Specials not yet implemented");
  } else if (mlir::isa<DimOp>(op) || mlir::isa<eltwise::ScalarConstantOp>(op)) {
    // TODO: If it turns out one of these matters, be sure to split it off from the ones that don't matter
    // TODO: Or otherwise skip? These shouldn't matter...
    // TODO: See if int/float matters...
    auto dop = builder_->MakeScalarConstantOp(0.);
    for (const auto& operand : op->getOperands()) {
      AddToGradient(operand, dop);
    }
  } else if (auto tmap_op = mlir::dyn_cast<AffineTensorMapOp>(op)) {
    // Just forward the gradient in a tmap to the tensor
    AddToGradient(tmap_op.tensor(), grads_[val]);
  } else {
    if (op->getNumOperands()) {
      throw std::runtime_error("Unexpected Operation type in ComputeOperandDerivs! Operation is " +
                               mlir::debugString(*op));
    }
    // If it has no operands, it doesn't matter whether we can differentiate it, so we do nothing
  }
}

mlir::Value* Gradient::GetDerivative(mlir::Value* val) {
  IVLOG(5, "Gradient::GetDerivative> " << mlir::debugString(*val));
  auto it = grads_.find(val);
  if (it != grads_.end()) {
    IVLOG(6, "  Gradient::GetDerivative> Derivative retrieved: " << mlir::debugString(*it->second));
    return it->second;
  }
  // TODO:
  // In the long run, this should probably just return 0 or otherwise indicate a continuation
  // For now, I want to know if we hit this (as we shouldn't in most cases)
  IVLOG(1, "Gradient::GetDerivative> The requested derivative of " << mlir::debugString(*val) << " was not computed!");
  throw std::runtime_error("TODO: requested derivative not from getBackwardSlice");
}

mlir::Value* Gradient::DeriveEltwise(mlir::Value* dout, mlir::Value* out, size_t idx) {
  auto op = out->getDefiningOp();
  IVLOG(5, "Gradient::DeriveEltwise> dout=" << mlir::debugString(*dout) << ", op=" << mlir::debugString(*op)
                                            << ", idx=" << idx);
  // TODO: Handle reshape specially. AST code for that follows
  // if (op->fn == "reshape") {
  //   std::vector<ExprPtr> args = {dout};
  //   auto in = op->args[0];
  //   auto dim_exprs = in->shape.dims_as_exprs();
  //   for (size_t i = 0; i < in->shape.dims.size(); ++i) {
  //     args.push_back(std::make_shared<DimExprExpr>(dim_exprs[i]));
  //   }
  //   return MakeCall("reshape", args);
  // }
  auto deriv = DerivRegistry::Instance()->Resolve(op->getName().getStringRef());
  llvm::SmallVector<mlir::Value*, 3> operands{op->getOperands()};  // TODO: Size
  return deriv.fn(out, dout, operands, deriv.user_fn, deriv.user_ctx)[idx];
}

mlir::Value* Gradient::DeriveContraction(mlir::Value* dout, mlir::Value* out, size_t idx) {
  IVLOG(5, "Gradient::DeriveContraction> dout=" << mlir::debugString(*dout) << ", out=" << mlir::debugString(*out)
                                                << ", idx=" << idx);
  auto op = llvm::dyn_cast_or_null<SymbolicContractionOp>(out->getDefiningOp());
  if (!op) {
    throw std::runtime_error("DeriveContraction called on non-contraction");
  }

  auto combo_kind = util::CombinationKind::none;  // This may be reset later if necessary
  size_t i = 0;
  std::vector<Value*> new_srcs;
  Value* target_src = nullptr;
  for (auto src : op.srcs()) {
    if (i == idx) {
      // This is the differentiated input; so swap in dout here to create the new op
      std::vector<Value*> dout_idxs;
      for (const auto& dim : llvm::cast<AffineMapOp>(op.sink()->getDefiningOp()).dims()) {
        dout_idxs.push_back(dim);
      }
      new_srcs.push_back(builder_->MakeAffineSourceIndexMapOp(dout, dout_idxs));
      // Also track that this is the differentiated source for later use
      target_src = src;
    } else {
      // This is the non-differentiated input; behavior depends on combo op
      switch (op.combo()) {
        case util::CombinationKind::none:
          IVLOG(1, "About to fail on a NONE combo, with idx==" << idx << ", and i==" << i);
          throw std::runtime_error(
              "Unexpected multiple inputs found when differentiating contraction with NONE combination op");
          break;
        case util::CombinationKind::add:
          // For +, we ignore the non-differentiated input
          combo_kind = util::CombinationKind::none;
          break;
        case util::CombinationKind::cond:
          throw std::runtime_error("Gradient of sum of conditionals not supported");
          break;
        case util::CombinationKind::eq:
          throw std::runtime_error("Gradient of sum of equalities not supported");
          break;
        case util::CombinationKind::mul:
          // For *, we multiply by the non-differentiated input
          new_srcs.push_back(src);
          combo_kind = util::CombinationKind::mul;
          break;
        default:
          throw std::runtime_error("Failed to recognize combination op during differentiation");
      }
    }
    i++;
  }
  if (!target_src) {
    throw std::runtime_error(
        llvm::formatv("Trying to derive contraction at out of range index (requested source operand {0})", idx).str());
  }
  auto target_src_op = llvm::cast<AffineTensorMapOp>(target_src->getDefiningOp());
  std::vector<mlir::Value*> sizes;
  for (size_t i = 0; i < target_src_op.tensor()->getType().dyn_cast<RankedTensorType>().getRank(); ++i) {
    sizes.push_back(builder_->MakeDimOp(target_src_op.tensor(), i));
  }
  std::vector<mlir::Value*> dsrc_idxs;
  for (const auto& dim : target_src_op.dims()) {
    dsrc_idxs.push_back(dim);
  }

  // TODO: Need to copy the constraints!
  auto dop = builder_->MakeContractionOp(             //
      util::AggregationKind::add,                     //
      combo_kind,                                     //
      new_srcs,                                       //
      builder_->MakeAffineSinkIndexMapOp(dsrc_idxs),  //
      builder_->MakeAffineSizeMapOp(sizes),           //
      llvm::formatv("d{0}", op.name()).str());
  return dop;
}

mlir::Value* Gradient::DeriveSpecial(const mlir::Value* dout, SpecialOp* op, size_t idx) {
  throw std::runtime_error("Made it to DeriveSpecial!");
}

}  // namespace pmlc::dialect::tile
