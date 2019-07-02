#include "tile/lang/ast_ops.h"

#include "tile/lang/gen_special.h"

namespace vertexai {
namespace tile {
namespace lang {

struct ReshapeOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    IVLOG(1, "ReshapeOp");
    if (args.size() < 1) {
      throw std::runtime_error("'reshape' requires at least one argument.");
    }
    std::vector<std::shared_ptr<DimExpr>> dims;
    for (size_t i = 1; i < args.size(); i++) {
      auto int_expr = std::dynamic_pointer_cast<IntConst>(args[i]);
      if (int_expr) {
        dims.push_back(std::make_shared<DimIntExpr>(int_expr->value));
      } else {
        auto dim_expr = std::dynamic_pointer_cast<DimExprExpr>(args[i]);
        if (!dim_expr) {
          throw std::runtime_error("Additional parameters to 'reshape' must be an integer or TensorDims.");
        }
        dims.push_back(dim_expr->expr);
      }
    }
    return LogicalShape(args[0]->shape.dtype, dims);
  }
};

struct BooleanOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    auto ret = ComputeOutputShape(args);
    ret.dtype = DataType::BOOLEAN;
    return ret;
  }
};

struct FloatCastOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 2) {
      throw std::runtime_error("'as_float' requires 2 arguments.");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[1]);
    if (!int_expr) {
      throw std::runtime_error("'as_float' requires the second argument to be an integer.");
    }
    LogicalShape ret = args[0]->shape;
    switch (int_expr->value) {
      case 16:
        ret.dtype = DataType::FLOAT16;
        break;
      case 32:
        ret.dtype = DataType::FLOAT32;
        break;
      case 64:
        ret.dtype = DataType::FLOAT64;
        break;
      default:
        throw std::runtime_error("'as_float' requires the width to be one of: (16, 32, 64)");
    }
    return ret;
  }
};

struct IntCastOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 2) {
      throw std::runtime_error("'as_int' requires 2 arguments.");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[1]);
    if (!int_expr) {
      throw std::runtime_error("'as_int' requires the second argument to be an integer.");
    }
    LogicalShape ret = args[0]->shape;
    switch (int_expr->value) {
      case 16:
        ret.dtype = DataType::INT16;
        break;
      case 32:
        ret.dtype = DataType::INT32;
        break;
      case 64:
        ret.dtype = DataType::INT64;
        break;
      default:
        throw std::runtime_error("'as_int' requires the width to be one of: (16, 32, 64)");
    }
    return ret;
  }
};

struct UintCastOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 2) {
      throw std::runtime_error("'as_uint' requires 2 arguments.");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[1]);
    if (!int_expr) {
      throw std::runtime_error("'as_uint' requires the second argument to be an integer.");
    }
    LogicalShape ret = args[0]->shape;
    switch (int_expr->value) {
      case 16:
        ret.dtype = DataType::UINT16;
        break;
      case 32:
        ret.dtype = DataType::UINT32;
        break;
      case 64:
        ret.dtype = DataType::UINT64;
        break;
      default:
        throw std::runtime_error("'as_uint' requires the width to be one of: (16, 32, 64)");
    }
    return ret;
  }
};

struct IndexOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 2) {
      throw std::runtime_error("'index' requires 2 arguments.");
    }
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[1]);
    if (!int_expr) {
      throw std::runtime_error("'index' requires the second argument to be an integer.");
    }
    LogicalShape ret = args[0]->shape;
    ret.dtype = DataType::INT32;
    return ret;
  }
};

// struct ElementOp : PrimitiveOp {
//   LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
//     if (args.size() != 2) {
//       throw std::runtime_error("'element' requires 2 arguments.");
//     }
//     if (args[0].tag != Binding::TUPLE) {
//       throw std::runtime_error("'element' requires the first argument to be a tuple.");
//     }
//     auto int_expr = std::dynamic_pointer_cast<IntConst>(args[1]);
//     if (!int_expr) {
//       throw std::runtime_error("'element' requires the second arguement to be an integer.");
//     }
//     auto elt = int_expr->value;
//     if (elt < 0 || elt >= static_cast<int64_t>(args[0].tuple.size())) {
//       throw std::runtime_error(
//           "'element' requires the second argument to be within the bounds of the specified tuple.");
//     }
//     if (args[0].tuple[elt].tag != Binding::TENSOR) {
//       throw std::runtime_error("'element' requires the resulting binding to be a tensor.");
//     }
//     return args[0].tuple[elt].shape;
//   }
// };

struct GatherOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 2) {
      throw std::runtime_error("'gather' requires 2 arguments.");
    }
    auto data = args[0];
    auto index = args[1];
    if (data->shape.dims.empty()) {
      throw std::runtime_error("'gather' requires first argument to have at least one dimension.");
    }
    if (index->shape.dtype != DataType::INT32) {
      // TODO: Handle other integer types?  Floor floats?
      throw std::runtime_error("'gather' requires the data type for the second argument to be INT32.");
    }
    std::vector<std::shared_ptr<DimExpr>> dims;
    for (size_t i = 0; i < index->shape.dims.size(); i++) {
      dims.push_back(index->shape.dims[i].expr);
    }
    for (size_t i = 1; i < data->shape.dims.size(); i++) {
      dims.push_back(data->shape.dims[i].expr);
    }
    return LogicalShape(data->shape.dtype, dims);
  }
};

struct ScatterOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 3) {
      throw std::runtime_error("'scatter' requires 3 arguments.");
    }
    if (args[0]->shape.dims.empty()) {
      throw std::runtime_error("'scatter' requires first argument to have at least one dimension.");
    }
    if (args[1]->shape.dtype != DataType::INT32) {
      // TODO: Handle other integer types?  Floor floats?
      throw std::runtime_error("'scatter' requires the data type for the second argument to be INT32.");
    }
    std::vector<std::shared_ptr<DimExpr>> dims = {args[2]->shape.dims[0].expr};
    for (size_t i = args[1]->shape.dims.size(); i < args[0]->shape.dims.size(); i++) {
      dims.push_back(args[0]->shape.dims[i].expr);
    }
    return LogicalShape(args[0]->shape.dtype, dims);
  }
};

struct ShapeOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 1) {
      throw std::runtime_error("'shape' requires exactly one argument.");
    }
    auto ndims = args[0]->shape.dims.size();
    return LogicalShape(DataType::INT32, {std::make_shared<DimIntExpr>(ndims)});
  }
};

struct PrngStateOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 1) {
      throw std::runtime_error("'prng_state' requires exactly one argument.");
    }
    if (args[0]->shape.dtype != DataType::PRNG) {
      throw std::runtime_error("'prng_state' requires one argument that is the result of 'prng_step'");
    }
    std::vector<std::shared_ptr<DimExpr>> dims = {
        std::make_shared<DimIntExpr>(3),
        std::make_shared<DimIntExpr>(k_rng_size),
    };
    return LogicalShape(DataType::UINT32, dims);
  }
};

struct PrngValueOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() != 1) {
      throw std::runtime_error("'prng_value' requires exactly one argument.");
    }
    if (args[0]->shape.dtype != DataType::PRNG) {
      throw std::runtime_error("'prng_value' requires one argument that is the result of 'prng_step'");
    }
    auto ret = args[0]->shape;
    ret.dtype = DataType::FLOAT32;
    return ret;
  }
};

struct PrngStepOp : PrimitiveOp {
  LogicalShape ComputeShape(const std::vector<std::shared_ptr<Expr>>& args) const {
    if (args.size() < 1) {
      throw std::runtime_error("'prng_step' must have at least one argument.");
    }
    // Valididate PRNG state size
    // auto shape = args[0]->shape;
    // if (shape != LogicalShape(DataType::UINT32, {3, k_rng_size})) {
    //   throw std::runtime_error("'prng_step' requires a valid PRNG state tensor.");
    // }
    // Get the output shape sizes
    std::vector<std::shared_ptr<DimExpr>> dims;
    for (size_t i = 1; i < args.size(); i++) {
      auto int_expr = std::dynamic_pointer_cast<IntConst>(args[i]);
      if (!int_expr) {
        throw std::runtime_error("'prng_step' requires additional arguments to be integers.");
      }
      dims.push_back(std::make_shared<DimIntExpr>(int_expr->value));
    }
    return LogicalShape(DataType::PRNG, dims);
  }
};

[[gnu::unused]] auto init = []() {
  auto registry = PrimitiveOpRegistry::Instance();
  registry->Register("as_float", std::make_unique<FloatCastOp>());
  registry->Register("as_int", std::make_unique<IntCastOp>());
  registry->Register("as_uint", std::make_unique<UintCastOp>());
  registry->Register("cmp_eq", std::make_unique<BooleanOp>());
  registry->Register("cmp_ge", std::make_unique<BooleanOp>());
  registry->Register("cmp_gt", std::make_unique<BooleanOp>());
  registry->Register("cmp_le", std::make_unique<BooleanOp>());
  registry->Register("cmp_lt", std::make_unique<BooleanOp>());
  registry->Register("cmp_ne", std::make_unique<BooleanOp>());
  // registry->Register("element", std::make_unique<ElementOp>());
  registry->Register("gather", std::make_unique<GatherOp>());
  registry->Register("index", std::make_unique<IndexOp>());
  registry->Register("prng_state", std::make_unique<PrngStateOp>());
  registry->Register("prng_step", std::make_unique<PrngStepOp>());
  registry->Register("prng_value", std::make_unique<PrngValueOp>());
  registry->Register("reshape", std::make_unique<ReshapeOp>());
  registry->Register("scatter", std::make_unique<ScatterOp>());
  registry->Register("shape", std::make_unique<ShapeOp>());
  return 0;
}();

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
