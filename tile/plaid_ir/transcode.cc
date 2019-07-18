// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/transcode.h"
#include "base/util/lookup.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

static Type ToPlaidIR(MLIRContext* ctx, DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT16:
      return FloatType::get(mlir::StandardTypes::F16, ctx);
    case DataType::FLOAT32:
      return FloatType::get(mlir::StandardTypes::F32, ctx);
    case DataType::FLOAT64:
      return FloatType::get(mlir::StandardTypes::F64, ctx);
    default:
      throw std::runtime_error("Unimplemented");
  }
}

static Type ToPlaidIR(MLIRContext* ctx, const TensorShape& shape) {
  Type dtype = ToPlaidIR(ctx, shape.type);
  std::vector<TensorDim> dims(shape.dims.size());
  for (size_t i = 0; i < shape.dims.size(); i++) {
    const auto& src = shape.dims[i];
    auto& dst = dims[i];
    dst.size = src.size;
    dst.stride = src.stride;
  }
  return TensorType::get(ctx, dtype, dims);
}

struct SymbolTable {
  std::map<std::string, Value*> refs;
  std::map<std::string, Value*> idxs;
  std::map<std::string, Value*> scalars;
};

static Value* ToPlaidIR(OpBuilder* builder, const SymbolTable& syms, const stripe::Affine& affine) {
  std::vector<Value*> add_inputs;
  for (const auto& kvp : affine.getMap()) {
    Value* term;
    if (kvp.first.empty()) {
      term = builder->create<AffineConstOp>(builder->getUnknownLoc(), builder->getType<AffineType>(),
                                            builder->getI64IntegerAttr(kvp.second));
    } else {
      Value* orig = safe_at(syms.idxs, kvp.first);
      term = builder->create<AffineMulOp>(builder->getUnknownLoc(), builder->getType<AffineType>(), orig,
                                          builder->getI64IntegerAttr(kvp.second));
    }
    add_inputs.push_back(term);
  }
  if (add_inputs.size() == 0) {
    return builder->create<AffineConstOp>(builder->getUnknownLoc(), builder->getType<AffineType>(),
                                          builder->getI64IntegerAttr(0));
  }
  if (add_inputs.size() == 1) {
    return add_inputs[0];
  }
  return builder->create<AffineAddOp>(builder->getUnknownLoc(), builder->getType<AffineType>(), add_inputs);
}

static void ToPlaidIR(OpBuilder* builder, const SymbolTable& outer, const stripe::Block& block) {
  // Make room for local symbols
  SymbolTable locals;

  // Make the actual inner block + terminate it
  auto orig_insert = builder->saveInsertionPoint();
  Block* body = new Block();
  builder->setInsertionPointToStart(body);
  builder->create<TerminateOp>(builder->getUnknownLoc());
  builder->setInsertionPointToStart(body);

  // Process the indexes
  std::vector<int64_t> ranges;
  for (const auto& idx : block.idxs) {
    if (idx.affine == stripe::Affine()) {
      // Handle the normal index case by adding a param to the body and the
      // range to the list to be used in the eventual attributes
      auto arg = body->addArgument(AffineType::get(builder->getContext()));
      locals.idxs.emplace(idx.name, arg);
      ranges.push_back(static_cast<int64_t>(idx.range));
    } else {
      // Handle the 'passthru' case by computing the appropriate affine and
      // adding into the symbol table
      if (idx.range != 1) {
        throw std::runtime_error("Invalid Stripe: range and affine both set on index");
      }
      locals.idxs.emplace(idx.name, ToPlaidIR(builder, outer, idx.affine));
    }
  }

  // Process the refinements
  for (const auto& ref : block.refs) {
    Value* from;
    if (ref.from == "") {
      Type atype = ToPlaidIR(builder->getContext(), ref.interior_shape);
      from = builder->create<AllocateOp>(builder->getUnknownLoc(), atype);
    } else {
      from = safe_at(outer.refs, ref.from);
    }
    std::vector<Value*> offsets;
    for (const auto& aff : ref.access) {
      offsets.push_back(ToPlaidIR(builder, locals, aff));
    }
    Value* nref = builder->create<RefineOp>(builder->getUnknownLoc(), from->getType(), from, offsets).result();
    locals.refs.emplace(ref.into(), nref);
  }

  // Process the constraints
  for (const auto& con : block.constraints) {
    // Make the actual constraint value
    auto aif = builder->create<AffineIfOp>(builder->getUnknownLoc(), ToPlaidIR(builder, locals, con));
    // Make the block + attach to the region
    Block* if_body = new Block();
    aif.getOperation()->getRegion(0).push_back(if_body);
    // Move to the interior
    builder->setInsertionPointToStart(if_body);
    builder->create<TerminateOp>(builder->getUnknownLoc());
    builder->setInsertionPointToStart(if_body);
  }

  // Process the statements
  for (const auto& stmt : block.stmts) {
    switch (stmt->kind()) {
      case stripe::StmtKind::Load: {
        const auto& load = stripe::Load::Downcast(stmt);
        Value* from = safe_at(locals.refs, load->from);
        auto tt = from->getType().cast<TensorType>();
        auto inst = builder->create<LoadOp>(builder->getUnknownLoc(), tt.base(), from);
        locals.scalars.emplace(load->into, inst);
      } break;
      case stripe::StmtKind::Store: {
        const auto& store = stripe::Store::Downcast(stmt);
        Value* into = safe_at(locals.refs, store->into);
        Value* from = safe_at(locals.scalars, store->from);
        builder->create<StoreOp>(builder->getUnknownLoc(), into, from);
      } break;
      case stripe::StmtKind::Constant:
        throw std::runtime_error("Constant Unimplemented");
        break;
      case stripe::StmtKind::LoadIndex:
        throw std::runtime_error("LoadIndex Unimplemented");
        break;
      case stripe::StmtKind::Special:
        throw std::runtime_error("Special Unimplemented");
        break;
      case stripe::StmtKind::Intrinsic: {
        const auto& intrinsic = stripe::Intrinsic::Downcast(stmt);
        std::vector<Value*> inputs;
        for (const auto& in : intrinsic->inputs) {
          inputs.push_back(safe_at(locals.scalars, in));
        }
        std::vector<Type> out_types;
        if (intrinsic->outputs.size() != 1) {
          throw std::runtime_error("Multi-return intrinsics not supported");
        }
        out_types.push_back(ToPlaidIR(builder->getContext(), intrinsic->type));
        auto name_attr = builder->getStringAttr(intrinsic->name);
        auto inst = builder->create<IntrinsicOp>(builder->getUnknownLoc(), out_types, inputs, name_attr);
        auto rr = inst.getResults().begin();
        for (const auto& out : intrinsic->outputs) {
          Value* val = *rr++;
          locals.scalars.emplace(out, val);
        }
      } break;
      case stripe::StmtKind::Block:
        ToPlaidIR(builder, locals, *stripe::Block::Downcast(stmt));
        break;
    }
  }

  // Build the loop itself
  builder->restoreInsertionPoint(orig_insert);
  auto loop_op = builder->create<ParallelForOp>(builder->getUnknownLoc(), builder->getI64ArrayAttr(ranges));
  loop_op.getOperation()->getRegion(0).push_back(body);
}

mlir::FuncOp StripeToPlaidIR(MLIRContext* ctx, const stripe::Program& prog) {
  std::cout << *prog.entry;
  // Build the function prototype from program entry
  std::vector<mlir::Type> arg_types;
  for (const auto& ref : prog.entry->refs) {
    arg_types.push_back(ToPlaidIR(ctx, ref.interior_shape));
  }
  auto func_type = mlir::FunctionType::get(arg_types, {}, ctx);
  // Make function
  mlir::Location loc = mlir::UnknownLoc::get(ctx);
  mlir::FuncOp func = mlir::FuncOp::create(loc, "test", func_type, {});
  func.addEntryBlock();
  auto& region = func.getBody();
  auto& block = region.front();
  auto builder = llvm::make_unique<mlir::OpBuilder>(region);
  // Fill initial symbol table with parameters
  SymbolTable initial;
  size_t i = 0;
  for (const auto& ref : prog.entry->refs) {
    initial.refs.emplace(ref.into(), block.getArgument(i++));
  }
  ToPlaidIR(builder.get(), initial, *prog.entry->SubBlock(0));
  builder->create<TerminateOp>(builder->getUnknownLoc());
  return func;
}

stripe::Program PlaidIRToStripe(const mlir::FuncOp& func) {
  stripe::Program r;
  auto prog = std::make_shared<stripe::Block>();
  r.entry = prog;
  return r;
}

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
