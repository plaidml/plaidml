// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/builder.h"

#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/env.h"
#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/tile/dialect.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/slice.h"
#include "tile/base/shape.h"

namespace pmlc {
namespace dialect {
namespace tile {

using eltwise::ScalarConstantOp;
using eltwise::ScalarType;
using mlir::Block;
using mlir::BlockAndValueMapping;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::UnknownLoc;

struct TileBuilder::Impl {
  MLIRContext context;
  ModuleOp module;
  OpBuilder builder;

  Impl()
      : module(ModuleOp::create(UnknownLoc::get(&context))),  //
        builder(module.getBody()) {}

  mlir::Type ComputeElementType(llvm::ArrayRef<mlir::Type> types) {
    DataType ret = DataType::INVALID;
    for (auto type : types) {
      auto tensorType = type.cast<ShapedType>();
      auto dtype = tensorType.getElementType().cast<ScalarType>().type();
      ret = CommonSupertype(ret, dtype);
    }
    return builder.getType<ScalarType>(ret);
  }

  const mlir::AbstractOperation* lookupOperation(StringRef op) {
    auto opName = eltwise::Dialect::getCanonicalOpName(op);
    auto abstractOp = mlir::AbstractOperation::lookup(opName, &context);
    if (!abstractOp) {
      opName = tile::Dialect::getCanonicalOpName(op);
      abstractOp = mlir::AbstractOperation::lookup(opName, &context);
      if (!abstractOp) {
        throw std::runtime_error("Unknown op: " + op.str());
      }
    }
    return abstractOp;
  }

  using CreateOpFunc = std::function<void(OpBuilder, BlockAndValueMapping*)>;

  Operation* MakeContraction(             //
      llvm::ArrayRef<mlir::Value*> srcs,  //
      mlir::Value* sink,                  //
      mlir::Value* sizes,                 //
      CreateOpFunc fn) {
    IVLOG(5, "TileBuilder::Impl::MakeContraction>");
    IVLOG(5, mlir::debugString(module));
    // Compute the sink shape of the contraction
    llvm::SmallVector<mlir::Type, 3> types;
    for (auto src : srcs) {
      IVLOG(6, "  src: " << mlir::debugString(*src));
      auto map_op = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      types.push_back(map_op.tensor()->getType());
    }
    IVLOG(6, "  sink: " << mlir::debugString(*sink));
    IVLOG(6, "  sizes: " << mlir::debugString(*sizes));
    auto elementType = ComputeElementType(types);
    auto size_map_op = llvm::cast<AffineSizeMapOp>(sizes->getDefiningOp());
    llvm::SmallVector<Value*, 4> size_map_sizes(size_map_op.sizes());
    auto shape = eltwise::ComputeShape(size_map_sizes);
    auto tensorType = builder.getTensorType(shape, elementType);
    auto domain = builder.create<AffineDomainOp>(builder.getUnknownLoc(), tensorType);
    auto body = new Block();
    domain.body().push_back(body);
    llvm::SetVector<mlir::Value*> values;
    values.insert(srcs.begin(), srcs.end());
    values.insert(sink);
    values.insert(sizes);
    auto slice = util::getBackwardSlice(values, false, [](Value* value) {  //
      return value->getType().isa<IndexType>();
    });
    // Find and replace each AffineIndexOp with a BlockArgument of the domain op
    BlockAndValueMapping mapper;
    std::queue<mlir::Value*> worklist;
    for (auto value : slice) {
      auto op = value->getDefiningOp();
      if (auto idx_op = llvm::dyn_cast<AffineIndexOp>(op)) {
        auto arg = body->addArgument(idx_op.getType());
        mapper.map(value, arg);
        worklist.push(value);
      }
    }
    // Move across only the values/ops that depend on AffineIndexOps
    // First determine the transitive users of AffineIndexOps
    std::set<Value*> belong;
    while (worklist.size()) {
      auto value = worklist.front();
      worklist.pop();
      for (auto user : value->getUsers()) {
        auto user_value = user->getResult(0);
        if (!belong.count(user_value)) {
          belong.insert(user_value);
          worklist.push(user_value);
        }
      }
    }
    // Now move across ops but do so in topologically sorted order
    OpBuilder domain_builder(body);
    for (auto value : slice) {
      auto op = value->getDefiningOp();
      if (belong.count(value) ||                    //
          llvm::isa<AffineSourceIndexMapOp>(op) ||  //
          llvm::isa<AffineSinkIndexMapOp>(op) ||    //
          llvm::isa<AffineSizeMapOp>(op)) {
        auto new_value = domain_builder.clone(*op, mapper)->getResult(0);
        mapper.map(value, new_value);
      }
    }
    fn(domain_builder, &mapper);
    IVLOG(5, mlir::debugString(domain));
    return domain.getOperation();
  }

  template <typename ConOp>
  mlir::Value* MakeUnaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 1) {
      throw std::runtime_error("Unary contraction op requires 1 operand");
    }
    auto domain = MakeContraction({srcs[0]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapper) {
      auto new_src = mapper->lookup(srcs[0]);
      auto new_sink = mapper->lookup(sink);
      auto new_sizes = mapper->lookup(sizes);
      domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src, new_sink);
    });
    return domain->getResult(0);
  }

  template <typename ConOp>
  mlir::Value* MakeBinaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 2) {
      throw std::runtime_error("Binary contraction op requires 2 operands");
    }
    auto domain =
        MakeContraction({srcs[0], srcs[1]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapper) {
          auto new_src1 = mapper->lookup(srcs[0]);
          auto new_src2 = mapper->lookup(srcs[1]);
          auto new_sink = mapper->lookup(sink);
          auto new_sizes = mapper->lookup(sizes);
          domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src1, new_src2, new_sink);
        });
    return domain->getResult(0);
  }

  template <typename ConOp>
  mlir::Value* MakeTernaryContraction(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, mlir::Value* sizes) {
    if (srcs.size() != 3) {
      throw std::runtime_error("Ternary contraction op requires 3 operands");
    }
    auto domain = MakeContraction(
        {srcs[0], srcs[1], srcs[2]}, sink, sizes, [&](OpBuilder domain_builder, BlockAndValueMapping* mapper) {
          auto new_src1 = mapper->lookup(srcs[0]);
          auto new_src2 = mapper->lookup(srcs[1]);
          auto new_src3 = mapper->lookup(srcs[2]);
          auto new_sink = mapper->lookup(sink);
          auto new_sizes = mapper->lookup(sizes);
          domain_builder.create<ConOp>(builder.getUnknownLoc(), new_sizes, new_src1, new_src2, new_src3, new_sink);
        });
    return domain->getResult(0);
  }
};

TileBuilder::TileBuilder() : impl(new Impl) {}

TileBuilder::~TileBuilder() = default;

void TileBuilder::Destroy(mlir::Value* value) {
  IVLOG(5, "TileBuilder::Destroy> value");
  // TODO: fix memory mgmt issues, once purely MLIR path is complete
  // if (value && value->use_empty()) {
  //   auto op = value->getDefiningOp();
  //   if (op && op->use_empty()) {
  //     op->erase();
  //   }
  // }
}

void TileBuilder::BindTensorDim(unsigned dim, mlir::Value* from, mlir::Value** into) {
  if (!from) {
    throw std::runtime_error("BindTensorDim: from == nullptr");
  }
  if (from == reinterpret_cast<mlir::Value*>(0x1)) {  // TODO: Special testing code REMOVE IT
    throw std::runtime_error("FOUND THE SPECIAL 0x1!!");
  }
  IVLOG(5, "TileBuilder::BindTensorDim> from: " << mlir::debugString(*from));
  if (!into) {
    throw std::runtime_error("BindTensorDim: into == nullptr");
  }
  if (*into) {
    IVLOG(6, "into: " << mlir::debugString(**into));
    auto fromType = from->getType().dyn_cast<mlir::RankedTensorType>();
    if (!fromType) {
      throw std::runtime_error("Unexpected type");
    }
    auto fromSize = fromType.getDimSize(dim);
    if (!mlir::ShapedType::isDynamic(fromSize)) {
      auto op = (*into)->getDefiningOp();
      if (!op) {
        throw std::runtime_error("No defining op");
      }
      if (auto const_op = llvm::dyn_cast<AffineConstantOp>(op)) {
        auto attr = const_op.getValue().dyn_cast<IntegerAttr>();
        if (!attr) {
          throw std::runtime_error("Expected IntegerAttr for value of AffineConstantOp");
        }
        IVLOG(6, "dim: " << dim << ", from: " << fromSize << ", into: " << attr.getInt());
        if (fromSize != attr.getInt()) {
          std::string str;
          llvm::raw_string_ostream os(str);
          os << llvm::formatv("bind_dims() mismatch on dim {0}. from: {1}, into: {2}", dim, fromSize, attr.getInt());
          throw std::runtime_error(os.str());
        }
      }
    }
  }
  *into = MakeDimOp(from, dim);
}

Shape TileBuilder::GetShape(mlir::Value* tensor) {
  IVLOG(5, "TileBuilder::GetShape>");
  auto type = tensor->getType().dyn_cast<mlir::RankedTensorType>();
  if (!type) {
    throw std::runtime_error("Only tensor types are supported");
  }
  auto elementType = type.getElementType().dyn_cast<ScalarType>();
  if (!elementType) {
    throw std::runtime_error("Only scalar element types are supported");
  }
  return Shape{elementType.type(), type.getShape()};
}

mlir::Value* TileBuilder::MakePrimitiveOp(llvm::StringRef fn, llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakePrimitiveOp> " << fn.str());
  for (auto arg : args) {
    IVLOG(6, "  arg: " << mlir::debugString(*arg));
  }
  auto abstractOp = impl->lookupOperation(fn);
  auto genericBuilder = abstractOp->getInterface<util::GenericBuilder>();
  if (!genericBuilder) {
    throw std::runtime_error("Unknown intrinsic: " + fn.str());
  }
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);  // TODO
  auto op = genericBuilder->create(&impl->builder, impl->builder.getUnknownLoc(), type, args);
  return op->getResult(0);
}

mlir::Value* TileBuilder::Clone(mlir::Value* value) {
  IVLOG(5, "TileBuilder::Clone> " << mlir::debugString(*value));
  return impl->builder.clone(*value->getDefiningOp())->getResult(0);
}

mlir::Value* TileBuilder::MakeNoneOp() {
  IVLOG(5, "TileBuilder::MakeNoneOp>");
  auto type = impl->builder.getNoneType();
  return impl->builder.create<NoneOp>(impl->builder.getUnknownLoc(), type).result();
}

mlir::Value* TileBuilder::MakeStringOp(llvm::StringRef value) {
  IVLOG(5, "TileBuilder::MakeStringOp> " << value.str());
  auto type = StringType::get(&impl->context);
  auto attr = impl->builder.getStringAttr(value);
  return impl->builder.create<StringOp>(impl->builder.getUnknownLoc(), type, attr).result();
}

mlir::Value* TileBuilder::MakeTupleOp(llvm::ArrayRef<mlir::Value*> elts) {
  IVLOG(5, "TileBuilder::MakeTupleOp> elts: " << elts.size());
  std::vector<Type> types;
  for (auto elt : elts) {
    types.push_back(elt->getType());
  }
  auto tupleType = impl->builder.getTupleType(types);
  return impl->builder.create<TupleOp>(impl->builder.getUnknownLoc(), tupleType, elts).result();
}

std::vector<mlir::Value*> TileBuilder::GetTupleElements(mlir::Value* value) {
  IVLOG(5, "TileBuilder::GetTupleElements> " << mlir::debugString(*value));
  if (auto op = llvm::dyn_cast<TupleOp>(value->getDefiningOp())) {
    return std::vector<mlir::Value*>(op.elts().begin(), op.elts().end());
  }
  throw std::runtime_error("Expected TupleOp");
}

mlir::Value* TileBuilder::MakeScalarConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::INT32);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

mlir::Value* TileBuilder::MakeScalarConstantOp(double value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

mlir::Value* TileBuilder::MakeDimOp(mlir::Value* tensor, unsigned dim) {
  IVLOG(5, "TileBuilder::MakeDimOp> tensor: " << mlir::debugString(*tensor) << ", dim: " << dim);
  return impl->builder.create<DimOp>(impl->builder.getUnknownLoc(), tensor, dim).result();
}

mlir::Value* TileBuilder::MakePlaceholderOp(DataType dtype, llvm::ArrayRef<int64_t> dims) {
  IVLOG(5, "TileBuilder::MakePlaceholderOp> " << to_string(dtype));
  auto elt_type = impl->builder.getType<ScalarType>(dtype);
  // Convert dims: PlaidML semantics use 0 for unknown size, MLIR uses -1.
  llvm::SmallVector<int64_t, 4> mlir_dims(dims.begin(), dims.end());
  for (unsigned i = 0; i < mlir_dims.size(); i++) {
    if (mlir_dims[i] == 0) {
      mlir_dims[i] = -1;
    }
  }
  auto shape = RankedTensorType::get(mlir_dims, elt_type);
  return impl->builder.create<PlaceholderOp>(impl->builder.getUnknownLoc(), shape).result();
}

mlir::Value* TileBuilder::MakeAffineConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeAffineConstantOp> " << value);
  return impl->builder.create<AffineConstantOp>(impl->builder.getUnknownLoc(), value).result();
}

mlir::Value* TileBuilder::MakeAffineIndexOp(llvm::StringRef name) {
  IVLOG(5, "TileBuilder::MakeAffineIndexOp> " << name.str());
  auto op = impl->builder.create<AffineIndexOp>(impl->builder.getUnknownLoc());
  if (!name.empty()) {
    // op.setAttr(SymbolTable::getSymbolAttrName(), impl->builder.getStringAttr(name));
  }
  return op.result();
}

mlir::Value* TileBuilder::MakeAffineAddOp(llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineAddOp>");
  return impl->builder.create<AffineAddOp>(impl->builder.getUnknownLoc(), args).result();
}

mlir::Value* TileBuilder::MakeAffineSubOp(llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineSubOp>");
  return impl->builder.create<AffineSubOp>(impl->builder.getUnknownLoc(), args).result();
}

mlir::Value* TileBuilder::MakeAffineMulOp(llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMulOp>");
  return impl->builder.create<AffineMulOp>(impl->builder.getUnknownLoc(), args).result();
}

mlir::Value* TileBuilder::MakeAffineDivOp(llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineDivOp>");
  return impl->builder.create<AffineDivOp>(impl->builder.getUnknownLoc(), args).result();
}

mlir::Value* TileBuilder::MakeAffineNegOp(llvm::ArrayRef<mlir::Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineNegOp>");
  return impl->builder.create<AffineNegOp>(impl->builder.getUnknownLoc(), args).result();
}

mlir::Value* TileBuilder::MakeAffineSourceIndexMapOp(mlir::Value* tensor, llvm::ArrayRef<mlir::Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSourceIndexMapOp>");
  return impl->builder.create<AffineSourceIndexMapOp>(impl->builder.getUnknownLoc(), tensor, idxs).result();
}

mlir::Value* TileBuilder::MakeAffineSinkIndexMapOp(llvm::ArrayRef<mlir::Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSinkIndexMapOp>");
  return impl->builder.create<AffineSinkIndexMapOp>(impl->builder.getUnknownLoc(), idxs).result();
}

mlir::Value* TileBuilder::MakeAffineSizeMapOp(llvm::ArrayRef<mlir::Value*> sizes) {
  IVLOG(5, "TileBuilder::MakeAffineSizeMapOp>");
  return impl->builder.create<AffineSizeMapOp>(impl->builder.getUnknownLoc(), sizes).result();
}

#define DEFINE_CONTRACTION_OPS(_agg_op_)                                                                    \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##Op(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,     \
                                                  mlir::Value* sizes) {                                     \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##Op>");                                                        \
    return impl->MakeUnaryContraction<Con##_agg_op_##Op>(srcs, sink, sizes);                                \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##AddOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,  \
                                                     mlir::Value* sizes) {                                  \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##AddOp>");                                                     \
    return impl->MakeBinaryContraction<Con##_agg_op_##AddOp>(srcs, sink, sizes);                            \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##CondOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink, \
                                                      mlir::Value* sizes) {                                 \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##CondOp>");                                                    \
    return impl->MakeTernaryContraction<Con##_agg_op_##CondOp>(srcs, sink, sizes);                          \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##EqOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,   \
                                                    mlir::Value* sizes) {                                   \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##EqOp>");                                                      \
    return impl->MakeBinaryContraction<Con##_agg_op_##EqOp>(srcs, sink, sizes);                             \
  }                                                                                                         \
                                                                                                            \
  mlir::Value* TileBuilder::MakeCon##_agg_op_##MulOp(llvm::ArrayRef<mlir::Value*> srcs, mlir::Value* sink,  \
                                                     mlir::Value* sizes) {                                  \
    IVLOG(5, "TileBuilder::MakeCon##_agg_op_##MulOp>");                                                     \
    return impl->MakeBinaryContraction<Con##_agg_op_##MulOp>(srcs, sink, sizes);                            \
  }

DEFINE_CONTRACTION_OPS(Assign);
DEFINE_CONTRACTION_OPS(Max);
DEFINE_CONTRACTION_OPS(Min);
DEFINE_CONTRACTION_OPS(Prod);
DEFINE_CONTRACTION_OPS(Sum);

std::shared_ptr<TileProgram> TileBuilder::MakeProgram(  //
    llvm::StringRef name,                               //
    llvm::ArrayRef<mlir::Value*> outputs,               //
    llvm::MutableArrayRef<mlir::Value*> new_outputs) {
  IVLOG(5, "TileBuilder::MakeProgram> " << name.str());
  IVLOG(6, mlir::debugString(impl->module));
  // Compute the result types
  std::vector<mlir::Type> resultTypes(outputs.size());
  llvm::SetVector<mlir::Value*> values;
  for (unsigned i = 0; i < outputs.size(); i++) {
    if (!outputs[i]) {
      throw std::runtime_error("Invalid output");
    }
    resultTypes[i] = outputs[i]->getType();
    if (values.count(outputs[i])) {
      values.insert(MakePrimitiveOp("ident", {outputs[i]}));
    } else {
      values.insert(outputs[i]);
    }
  }
  auto slice = util::getBackwardSlice(values, true);
  // Compute the input types
  std::vector<mlir::Type> inputTypes;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    if (!op) {
      continue;
    }
    if (auto var_op = llvm::dyn_cast<PlaceholderOp>(op)) {
      auto value = var_op.getResult();
      inputTypes.push_back(value->getType());
    }
  }
  // Construct a module
  auto loc = mlir::UnknownLoc::get(&impl->context);
  auto module = ModuleOp::create(loc);
  auto program = std::make_shared<TileProgram>(module);
  // Construct a function to represent the entire program
  auto funcType = mlir::FunctionType::get(inputTypes, resultTypes, &impl->context);
  auto funcOp = mlir::FuncOp::create(loc, name, funcType, {});
  funcOp.addEntryBlock();
  OpBuilder builder(funcOp.getBody());
  unsigned argcnt = 0;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    // Only copy over top-level ops (those owned by the workspace module)
    if (op && op->getBlock() == impl->module.getBody()) {
      if (auto var_op = llvm::dyn_cast<PlaceholderOp>(op)) {
        // Replace placeholders with block arguments
        auto new_value = funcOp.getArgument(argcnt++);
        program->mapper.map(value, new_value);
      } else {
        auto new_value = builder.clone(*op, program->mapper)->getResult(0);
        program->mapper.map(value, new_value);
      }
    }
  }
  // Add a final ReturnOp
  std::vector<mlir::Value*> rets;
  for (unsigned i = 0; i < values.size(); i++) {
    auto new_value = program->mapper.lookup(values[i]);
    new_outputs[i] = new_value;
    rets.push_back(new_value);
  }
  builder.create<mlir::ReturnOp>(loc, rets);
  // Attach the function to the module
  module.push_back(funcOp);
  IVLOG(5, module);
  if (failed(mlir::verify(module))) {
    emitError(loc, "Module verification error");
  }
  // Do some optimization passes
  mlir::PassManager pm(&impl->context);
  if (vertexai::env::Get("PLAIDML_MLIR") == "1") {
    // TODO: Debug & re-enable the canonicalization pass
    pm.addPass(mlir::createCanonicalizerPass());
  }
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(module);
  if (failed(result)) {
    emitError(loc, "Optimization passes failure");
  }
  IVLOG(2, module);
  return program;
}

std::vector<mlir::Value*> TileBuilder::ComputeGradients(llvm::ArrayRef<mlir::Value*> wrt, mlir::Value* loss) {
  // TODO
  return wrt;
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
