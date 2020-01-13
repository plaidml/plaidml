// Copyright 2019 Intel Corporation.

#include "pmlc/dialect/tile/gradient.h"

#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/tile/builder.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/slice.h"

namespace pmlc::dialect::tile {

Gradient::Gradient(mlir::Value loss, TileBuilder* builder) : builder_(builder) {
  IVLOG(3, "Gradient::Gradient> loss: " << mlir::debugString(loss));
  grads_[loss] = builder_->MakeScalarConstantOp(1.0);
  llvm::SetVector<mlir::Value> loss_setvec;
  loss_setvec.insert(loss);
  auto defs = util::getBackwardSlice(  //
      loss_setvec, false, std::function<bool(mlir::Value)>{[](mlir::Value val) {
        auto op = val->getDefiningOp();
        // TODO: This is an ad hoc list of what to filter out; make it principled
        return !mlir::isa<AffineConstraintsOp>(op) &&  //
               !mlir::isa<AffineMapOp>(op) &&          //
               !mlir::isa<AffineIndexOp>(op) &&        //
               !mlir::isa<DimOp>(op) &&                //
               !mlir::isa<AffineConstantOp>(op) &&     //
               !mlir::isa<AffineAddOp>(op) &&          //
               !mlir::isa<AffineDivOp>(op) &&          //
               !mlir::isa<AffineMulOp>(op) &&          //
               !mlir::isa<AffineNegOp>(op) &&          //
               !mlir::isa<AffineSubOp>(op) &&          //
               !mlir::isa<AffineMaxOp>(op) &&          //
               !mlir::isa<AffineMinOp>(op) &&          //
               !mlir::isa<eltwise::ScalarConstantOp>(op);
      }});
  for (auto def = defs.rbegin(); def != defs.rend(); def++) {
    ComputeOperandDerivs(*def);
  }
  if (VLOG_IS_ON(5)) {
    IVLOG(5, "Gradient::Gradient> Computed the following gradients: ");
    for (auto [key, value] : grads_) {
      IVLOG(5, "  key is " << mlir::debugString(key) << "\n    for val " << mlir::debugString(value));
    }
  }
}

void Gradient::AddToGradient(Value source_op, Value deriv) {
  // Adds the gradient `deriv` computed for one use of `source_op` to the overall gradient of `source_op`
  if (!grads_.count(source_op)) {
    grads_[source_op] = deriv;
  } else {
    grads_[source_op] = builder_->MakePrimitiveOp("add", {grads_[source_op], deriv});
  }
}

void Gradient::ComputeOperandDerivs(mlir::Value val) {
  IVLOG(4, "Gradient::ComputeDerivative> " << mlir::debugString(val));
  // TODO: Throw on ops with multiple results?
  auto op = val->getDefiningOp();
  if (mlir::isa<AffineConstraintsOp>(op) ||  //
      mlir::isa<AffineMapOp>(op) ||          //
      mlir::isa<AffineIndexOp>(op) ||        //
      mlir::isa<PrngOp>(op) ||               //
      mlir::isa<ShapeOp>(op)) {
    // TODO: Make the list of which ops these are more principled. Also, should these all be caught in the backwards
    // slice filter? If so, probably throw here.
    IVLOG(6, "Gradient::ComputeDerivative> Skipping computing derivatives for op "
                 << mlir::debugString(*op) << ", as it is not a type of op for which gradients apply");
    return;
  }
  if (!grads_.count(val)) {
    IVLOG(1, "Gradient::ComputeOperandDerivs> Called on Value which has not itself been differentiated: "
                 << mlir::debugString(val));
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
      // TODO: I discussed this with Jeremy and I think the below notes are wrong, but we need another discussion.
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
  } else if (auto gather_op = mlir::dyn_cast<GatherOp>(op)) {
    auto tensor_input = gather_op.tensor();
    auto dims_input = gather_op.dims();
    auto dop = builder_->MakePrimitiveOp("scatter", {grads_[val], dims_input, tensor_input});
    auto zero_op = builder_->MakeScalarConstantOp(0.);
    AddToGradient(tensor_input, dop);
    AddToGradient(dims_input, zero_op);
  } else if (mlir::isa<ScatterOp>(op)) {
    // TODO
    throw std::runtime_error("TODO: Derivs of Scatter not yet implemented");
  } else if (auto reshape_op = mlir::dyn_cast<ReshapeOp>(op)) {
    auto tensor_input = reshape_op.tensor();
    std::vector<mlir::Value> args{grads_[val]};
    for (int i = 0; i < tensor_input->getType().dyn_cast<RankedTensorType>().getRank(); ++i) {
      args.push_back(builder_->MakeDimOp(tensor_input, i));
    }
    auto dop = builder_->MakePrimitiveOp("reshape", args);
    AddToGradient(tensor_input, dop);
  } else if (mlir::isa<SpecialOp>(op)) {
    throw std::runtime_error("Unrecognized special operation, unable to differentiate");
  } else if (mlir::isa<DimOp>(op) || mlir::isa<eltwise::ScalarConstantOp>(op)) {
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

mlir::Value Gradient::GetDerivative(mlir::Value val) {
  IVLOG(5, "Gradient::GetDerivative> " << mlir::debugString(val));
  auto it = grads_.find(val);
  if (it != grads_.end()) {
    IVLOG(6, "  Gradient::GetDerivative> Derivative retrieved: " << mlir::debugString(it->second));
    return it->second;
  }
  // TODO:
  // In the long run, this should probably just return 0 or otherwise indicate a continuation
  // For now, I want to know if we hit this (as we shouldn't in most cases)
  IVLOG(1, "Gradient::GetDerivative> The requested derivative of " << mlir::debugString(val) << " was not computed!");
  throw std::runtime_error("TODO: requested derivative not from getBackwardSlice");
}

mlir::Value Gradient::DeriveEltwise(mlir::Value dout, mlir::Value out, size_t idx) {
  auto op = out->getDefiningOp();
  IVLOG(5, "Gradient::DeriveEltwise> dout=" << mlir::debugString(dout) << ", op=" << mlir::debugString(*op)
                                            << ", idx=" << idx);
  auto deriv = DerivRegistry::Instance()->Resolve(op->getName().getStringRef());
  llvm::SmallVector<mlir::Value, 3> operands{op->getOperands()};  // TODO: Size
  // TODO: Need to add simple_reduce here, unless done in the "if isa EltwiseOp" block above
  return deriv.fn(out, dout, operands, deriv.user_fn, deriv.user_ctx)[idx];
}

mlir::Value Gradient::DeriveContraction(mlir::Value dout, mlir::Value out, size_t idx) {
  IVLOG(5, "Gradient::DeriveContraction> dout=" << mlir::debugString(dout) << ", out=" << mlir::debugString(out)
                                                << ", idx=" << idx);
  auto op = llvm::dyn_cast_or_null<SymbolicContractionOp>(out->getDefiningOp());
  if (!op) {
    throw std::runtime_error("DeriveContraction called on non-contraction");
  }
  if (op.combo() == util::CombinationKind::eq) {
    // TODO: What type should this 0 be?
    return builder_->MakeScalarConstantOp(0.);
  }

  auto combo_kind = util::CombinationKind::none;  // This may be reset later if necessary
  std::vector<Value> new_srcs;
  Value target_src = nullptr;
  switch (op.agg()) {
    case util::AggregationKind::max:
    case util::AggregationKind::min: {
      size_t i = 0;
      // TODO: Is there a better option to do "Get the unique src (or throw if not unique)"?
      for (auto src : op.srcs()) {
        if (i == idx) {
          combo_kind = util::CombinationKind::cond;
          auto src_op = mlir::dyn_cast_or_null<AffineTensorMapOp>(src->getDefiningOp());
          if (!src_op) {
            throw std::runtime_error("src_op as cast is null");
          }
          new_srcs.push_back(src_op);
          std::vector<Value> dout_idxs;
          for (const auto& dim : llvm::cast<AffineMapOp>(op.sink()->getDefiningOp()).dims()) {
            dout_idxs.push_back(dim);
          }
          new_srcs.push_back(builder_->MakeAffineSourceIndexMapOp(op, dout_idxs));
          new_srcs.push_back(builder_->MakeAffineSourceIndexMapOp(dout, dout_idxs));
          // Also track that this is the differentiated source for later use
          target_src = src;
        } else {
          throw std::runtime_error("Cannot differentiate max/min contractions with multiple input tensors");
        }
        i++;
      }
    } break;
    case util::AggregationKind::add:
    case util::AggregationKind::assign: {
      size_t i = 0;
      for (auto src : op.srcs()) {
        if (i == idx) {
          // This is the differentiated input; so swap in dout here to create the new op
          std::vector<Value> dout_idxs;
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
              throw std::logic_error("Gradient unexpectedly failed to detect combo op as equality");
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
    } break;
    case util::AggregationKind::mul:
      throw std::runtime_error("Cannot differentiate multiplication aggregations");
      break;
    default:
      throw std::runtime_error("Did not recognize aggregation operation when differentiating " +
                               mlir::debugString(out));
  }
  if (!target_src) {
    throw std::runtime_error(
        llvm::formatv("Trying to derive contraction at out of range index (requested source operand {0})", idx).str());
  }
  auto target_src_op = llvm::cast<AffineTensorMapOp>(target_src->getDefiningOp());
  std::vector<mlir::Value> sizes;
  for (int i = 0; i < target_src_op.tensor()->getType().dyn_cast<RankedTensorType>().getRank(); ++i) {
    sizes.push_back(builder_->MakeDimOp(target_src_op.tensor(), i));
  }
  std::vector<mlir::Value> dsrc_idxs;
  for (const auto& dim : target_src_op.dims()) {
    dsrc_idxs.push_back(dim);
  }

  std::string new_name;
  // TODO: Currently names are broken
  // auto src_name = op.name();
  // if (src_name) {
  //   new_name = llvm::formatv("d{0}", op.name()).str();
  // } // else use the empty string (already initialized by default ctor)
  auto dop_val = builder_->MakeContractionOp(         //
      util::AggregationKind::add,                     //
      combo_kind,                                     //
      new_srcs,                                       //
      builder_->MakeAffineSinkIndexMapOp(dsrc_idxs),  //
      builder_->MakeAffineSizeMapOp(sizes),           //
      new_name);
  // Copy the constraints from the forward pass contraction
  auto src_cons = llvm::dyn_cast<AffineConstraintsOp>(op.cons()->getDefiningOp());
  auto dop = llvm::dyn_cast<SymbolicContractionOp>(dop_val->getDefiningOp());
  auto dst_cons = llvm::dyn_cast<AffineConstraintsOp>(dop.cons()->getDefiningOp());
  mlir::SmallVector<mlir::Value, 6> pairs{src_cons.pairs()};
  for (const auto& pair : src_cons.pairs()) {
    pairs.emplace_back(pair);
  }
  dst_cons.getOperation()->setOperands(pairs);
  return dop_val;
}

mlir::Value Gradient::DeriveSpecial(const mlir::Value dout, SpecialOp* op, size_t idx) {
  throw std::runtime_error("Made it to DeriveSpecial!");
}

}  // namespace pmlc::dialect::tile
