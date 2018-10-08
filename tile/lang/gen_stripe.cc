#include "tile/lang/gen_stripe.h"

#include "tile/lang/bound.h"
#include "tile/lang/defract.h"
#include "tile/lang/reduce.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;    // NOLINT
using namespace stripe;  // NOLINT

namespace {

class StripeGenerator {
 public:
  StripeGenerator(const Program& parsed,  //
                  const Bindings& vars,   //
                  const ShapeMap& inputs,
                  const ShapeMap& outputs)
      : parsed_(parsed),   //
        vars_(vars),       //
        inputs_(inputs),   //
        outputs_(outputs)  //
  {}

  std::shared_ptr<Block> Run(const std::string& name) {
    LOG(INFO) << "Compiling " << parsed_.ops.size() << " ops";
    // The top level block is a 'main' function.
    // In/Out/InOut refinements made on main relate to user supplied inputs and outputs.
    // None refinements made on main relate to temporaries needed for communication between kernels.
    // The list of kernels to execute are the list of blocks defined within main.
    auto main = std::make_shared<Block>();
    main->name = name;
    // Add decls for external inputs/outputs
    AddDecls(main.get(), inputs_, true);
    AddDecls(main.get(), outputs_, false);
    // Add kernels to main
    for (const auto& op : parsed_.ops) {
      IVLOG(2, "Processing: " << op);
      switch (op.tag) {
        case Op::CONTRACTION:
          ProcessContraction(main.get(), op);
          break;
        case Op::FUNCTION:
          if (op.f.is_special()) {
            ProcessSpecial(main.get(), op);
          } else if (op.f.fn == "reshape") {
            ProcessReshape(main.get(), op);
          } else {
            ProcessElementwise(main.get(), op);
          }
          break;
        case Op::CONSTANT:
          // LOG(INFO) << "CONSTANT";
          break;
      }
    }
    // Add decls for temporaries
    for (const auto& item : vars_) {
      if (externals_.count(item.first) == 0) {
        const auto& binding = item.second;
        if (binding.tag == Binding::TENSOR) {
          std::vector<Affine> access(binding.shape.dims.size());
          main->refs.emplace_back(Refinement{
              RefDir::None,  // dir
              "",            // from
              item.first,    // into
              access,        // access
              binding.shape  // shape
          });
        }
      }
    }
    IVLOG(2, "Done");
    return main;
  }

 private:
  void AddDecls(Block* main, const ShapeMap& shapes, bool is_input) {
    for (const auto& item : shapes) {
      externals_.insert(item.first);
      std::vector<Affine> access(item.second.dims.size());
      if (is_input) {
        main->refs.emplace_back(Refinement{
            RefDir::In,  // dir
            item.first,  // from
            item.first,  // into
            access,      // access
            item.second  // shape
        });
      } else {
        main->refs.emplace_back(Refinement{
            RefDir::Out,       // dir
            item.first,        // from
            item.first,        // into
            access,            // access
            item.second,       // shape
            Intrinsic::ASSIGN  // agg_op
        });
      }
    }
  }

  void ProcessReshape(Block* main, const Op& op) {
    auto stmt = std::make_shared<Special>();
    stmt->name = op.f.fn;
    stmt->params = op.f.params;
    stmt->inputs = op.inputs;
    stmt->outputs = {op.output};
    main->stmts.push_back(stmt);
  }

  void ProcessContraction(Block* main, const Op& op) {
    if (GetShape(op.output).byte_size() == 0) {
      IVLOG(3, "Contraction output " << op.output << " size==0; skipping");
      return;
    }
    Contraction cion;
    std::vector<math::RangeConstraint> range_cons;
    auto shapes = MakeShapes(op.c);
    std::tie(cion, range_cons) = CompileContraction(op.c, shapes);

    // Compute bounds
    IndexBounds bounds;
    std::vector<SimpleConstraint> simple_cons;
    try {
      std::tie(bounds, simple_cons) = ComputeBounds(range_cons);
    } catch (const std::runtime_error& ex) {
      LOG(WARNING) << "Unable to compute bounds for contraction: " << to_string(cion);
      throw;
    }

    auto kernel = AddKernel(main);
    kernel->comments = to_string(op);

    std::vector<std::string> scalar_inputs;
    for (size_t i = 0; i < cion.specs.size(); i++) {
      const auto& spec = cion.specs[i];
      auto shape = ScalarShape(cion.specs[i].id);
      std::vector<Affine> access;
      for (const auto& poly : spec.spec) {
        access.emplace_back(Integerize(poly, bounds));
      }
      if (i == 0) {
        kernel->refs.emplace_back(Refinement{
            RefDir::Out,           // dir
            spec.id,               // from
            spec.id,               // into
            access,                // access
            shape,                 // shape
            GetAggOp(cion.agg_op)  // agg_op
        });
      } else {
        auto scalar_name = ScalarName(spec.id);
        scalar_inputs.push_back(scalar_name);
        kernel->refs.emplace_back(Refinement{
            RefDir::In,  // dir
            spec.id,     // from
            spec.id,     // into
            access,      // access
            shape        // shape
                         // agg_op
        });
        // LOAD
        kernel->stmts.push_back(std::make_shared<Load>(spec.id, scalar_name));
      }
    }

    for (const auto& kvp : bounds) {
      uint64_t range = kvp.second.max - kvp.second.min + 1;
      if (range == 1) {
        continue;
      }
      kernel->idxs.emplace_back(Index{kvp.first, range, 0});
    }
    for (const auto& constraint : simple_cons) {
      auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
      lhs -= constraint.rhs;                           // lhs <= 0;
      lhs = -lhs;                                      // lhs >= 0
      kernel->constraints.emplace_back(lhs);
    }

    if (NeedsInitialize(*kernel, shapes[0])) {
      auto stmt = std::make_shared<Special>();
      stmt->outputs = {op.output};
      if (op.c.use_default.empty()) {
        stmt->name = Intrinsic::ZERO;
      } else {
        stmt->name = Intrinsic::COPY;
        stmt->inputs.push_back(op.c.use_default);
      }
      main->stmts.insert(std::prev(main->stmts.end()), stmt);
    }

    // Combination Op
    if (scalar_inputs.size() > 1) {
      auto combo_op = GetComboOp(cion.comb_op);
      if (!combo_op.empty()) {
        AddIntrinsic(kernel.get(), combo_op, scalar_inputs, {ScalarName(op.output)});
      }
    }

    // STORE
    kernel->stmts.push_back(std::make_shared<Store>(ScalarName(op.output), op.output));
  }

  bool NeedsInitialize(const Block& block, const TensorShape& out_shape) {
    // Check if have a simple output: 1 unique index per dimension, each full range
    // If not, presume we need initialization for safety
    // We assume here that the 0'th refinement is the output refinement
    std::set<std::string> out_idxs;
    for (size_t i = 0; i < out_shape.dims.size(); i++) {
      Affine a = block.refs[0].access[i];
      if (a == 0 && out_shape.dims[i].size == 1) {
        continue;
      }
      if (a.constant() != 0 || a.getMap().size() != 1 || a.getMap().begin()->second != 1) {
        return true;  // If it's not a single index with a multiplier of 1, bail
      }
      std::string idx = a.getMap().begin()->first;
      if (out_idxs.count(idx)) {
        return true;  // If the index isn't unique, bail
      }
      out_idxs.insert(idx);
      if (block.idx_by_name(idx)->range != out_shape.dims[i].size) {
        return true;  // Index range doesn't match out_shape size
      }
    }
    // Now we check if we have any constraints that are 'output only'
    // Output only indexes actually reduce the range we write to, whereas constraints
    // that use both input + output make writes but only process some of the input
    for (const auto& con : block.constraints) {
      bool any_inputs = false;
      for (const auto& kvp : con.getMap()) {
        if (!kvp.first.empty() && out_idxs.count(kvp.first) == 0) {
          any_inputs = true;
        }
      }
      if (!any_inputs) {
        return true;  // Found at least one output only constraint
      }
    }
    return false;  // Looks good!
  }

  void ProcessElementwise(Block* main, const Op& op) {
    auto kernel = AddKernel(main);
    kernel->comments = to_string(op);

    auto out_shape = GetShape(op.output);
    std::vector<Affine> out_access;
    for (std::size_t i = 0; i < out_shape.dims.size(); ++i) {
      auto idx = Index{
          printstring("i%zu", i + 1),  // name
          out_shape.dims[i].size,      // range
          0                            // factor
      };
      out_access.emplace_back(Affine{idx.name});
      kernel->idxs.emplace_back(idx);
    }

    for (const auto& input : op.inputs) {
      const auto& binding = vars_.at(input);
      IVLOG(2, "  " << input << ": " << binding);
      switch (binding.tag) {
        case Binding::TENSOR: {
          // Be careful to handle broadcasts
          std::vector<Affine> access;
          int diff = out_shape.dims.size() - binding.shape.dims.size();
          for (int i = 0; i < out_shape.dims.size(); i++) {
            if (i >= diff) {
              const auto& dim = binding.shape.dims[i - diff];
              if (dim.size > 1) {
                access.emplace_back(Affine{kernel->idxs[i].name});
              } else {
                access.emplace_back(Affine{});
              }
            }
          }
          kernel->refs.emplace_back(Refinement{RefDir::In, input, input, access, GetShape(input)});
          // LOAD
          kernel->stmts.push_back(std::make_shared<Load>(input, ScalarName(input)));
        } break;
        case Binding::ICONST:
          kernel->stmts.push_back(std::make_shared<Constant>(ScalarName(input), binding.iconst));
          break;
        case Binding::FCONST:
          kernel->stmts.push_back(std::make_shared<Constant>(ScalarName(input), binding.fconst));
          break;
        case Binding::TUPLE:
          throw std::runtime_error("Not implemented!");
          break;
      }
    }

    kernel->refs.emplace_back(Refinement{RefDir::Out, op.output, op.output, out_access, out_shape});

    // INTRINSIC
    std::vector<std::string> scalar_inputs;
    for (const auto& input : op.inputs) {
      scalar_inputs.push_back(ScalarName(input));
    }
    AddIntrinsic(kernel.get(), op.f.fn, scalar_inputs, {ScalarName(op.output)});

    // STORE
    kernel->stmts.push_back(std::make_shared<Store>(ScalarName(op.output), op.output));
  }

  void ProcessSpecial(Block* main, const Op& op) {
    auto stmt = std::make_shared<Special>();
    stmt->name = op.f.fn;
    stmt->params = op.f.params;
    stmt->inputs = op.inputs;
    stmt->outputs = {op.output};
    main->stmts.push_back(stmt);
  }

  std::shared_ptr<Block> AddKernel(Block* parent, const char* prefix = "") {
    auto block = std::make_shared<Block>();
    block->name = printstring("%skernel_%zu", prefix, parent->stmts.size());
    parent->stmts.push_back(block);
    return block;
  }

  std::vector<TensorShape> MakeShapes(const Contraction& con) {
    std::vector<TensorShape> shapes;
    for (const TensorSpec& spec : con.specs) {
      shapes.push_back(GetShape(spec.id));
    }
    return shapes;
  }

  // bool NeedsZero(const FlatContraction& flat) {
  //   std::vector<std::pair<size_t, size_t>> out_pattern;
  //   if (flat.access[0].offset != 0) {
  //     return true;
  //   }
  //   for (size_t i = 0; i < flat.names.size(); i++) {
  //     if (flat.access[0].strides[i] == 0) {
  //       continue;
  //     }
  //     if (flat.access[0].strides[i] < 0) {
  //       return true;
  //     }  // Don't try to be fancy, fallback
  //     out_pattern.emplace_back(flat.access[0].strides[i], flat.ranges[i]);
  //   }
  //   for (const auto& fc : flat.constraints) {
  //     bool output_only = true;
  //     for (size_t i = 0; i < flat.names.size(); i++) {
  //       if (fc.lhs[i] != 0 && flat.access[0].strides[i] == 0) {
  //         output_only = false;
  //         break;
  //       }
  //     }
  //     if (output_only) {
  //       return true;
  //     }
  //   }
  //   std::sort(out_pattern.begin(), out_pattern.end());
  //   size_t curskip = 1;
  //   for (const auto& p : out_pattern) {
  //     if (curskip != p.first) {
  //       return true;
  //     }
  //     curskip *= p.second;
  //   }
  //   return curskip != flat.access[0].global_index_limit;
  // }

  void AddIntrinsic(Block* block,                            //
                    const std::string& name,                 //
                    const std::vector<std::string>& inputs,  //
                    const std::vector<std::string>& outputs) {
    auto stmt = std::make_shared<Intrinsic>();
    stmt->name = name;
    stmt->inputs = inputs;
    stmt->outputs = outputs;
    block->stmts.push_back(stmt);
  }

  inline std::string ScalarName(const std::string& name) {  //
    return printstring("$%s", name.c_str());
  }

  TensorShape GetShape(const std::string& name) const {
    auto it = vars_.find(name);
    if (it == vars_.end()) {
      throw std::runtime_error(printstring("Unknown shape: %s", name.c_str()));
    }
    return it->second.shape;
  }

  TensorShape ScalarShape(const std::string& name) const {
    auto it = vars_.find(name);
    if (it == vars_.end()) {
      throw std::runtime_error(printstring("Unknown shape: %s", name.c_str()));
    }
    TensorShape shape(it->second.shape.type, {});
    for (const auto& dim : it->second.shape.dims) {
      shape.dims.push_back(TensorDimension(dim.stride, 1));
    }
    return shape;
  }

  Affine Integerize(const Polynomial<Rational>& poly, const IndexBounds& bounds) {
    Affine result;
    for (const auto& term : poly.getMap()) {
      if (denominator(term.second) != 1) {
        throw std::runtime_error("Non-integer polynomial in Integerize");
      }
      auto int_value = static_cast<int64_t>(numerator(term.second));
      if (term.first.empty()) {
        result += int_value;
      } else {
        const auto& bound = bounds.at(term.first);
        result += int_value * bound.min;
        if (bound.min != bound.max) {
          result += Polynomial<int64_t>(term.first, int_value);
        }
      }
    }
    return result;
  }

  std::string GetAggOp(AggregationOp op) {
    switch (op) {
      case AggregationOp::SUM:
        return Intrinsic::SUM;
      case AggregationOp::MAX:
        return Intrinsic::MAX;
      case AggregationOp::MIN:
        return Intrinsic::MIN;
      case AggregationOp::PROD:
        return Intrinsic::PROD;
      case AggregationOp::ASSIGN:
        return Intrinsic::ASSIGN;
      default:
        return "";
    }
  }

  std::string GetComboOp(CombinationOp op) {
    switch (op) {
      case CombinationOp::MULTIPLY:
        return Intrinsic::MUL;
      case CombinationOp::PLUS:
        return Intrinsic::ADD;
      case CombinationOp::EQ:
        return Intrinsic::EQ;
      case CombinationOp::COND:
        return Intrinsic::COND;
      default:
        return "";
    }
  }

  std::pair<Contraction, std::vector<math::RangeConstraint>>  //
  CompileContraction(const Contraction& cion, const std::vector<TensorShape>& shapes) {
    if (cion.specs.size() != 2 && cion.specs.size() != 3 && cion.specs.size() != 4) {
      throw std::runtime_error("Currently, we only support 1, 2, or 3 element Contractions");
    }
    std::ostringstream cs;
    SVLOG(cs, 3, "Original:\n" << to_string(cion).c_str());
    auto integral_cion = ConstrainIndexVarsToInts(cion);
    SVLOG(cs, 3, "With Index Variables Made Integral:\n" << to_string(integral_cion).c_str());
    // Check if we can skip reduce
    bool fancy = false;
    for (const auto& poly : cion.specs[0].spec) {
      if (poly.getMap().size() > 2 || (poly.getMap().size() == 2 && poly.constant() == 0)) {
        fancy = true;
        break;
      }
    }
    auto cons = GatherConstraints(integral_cion, shapes);
    SVLOG(cs, 3, "Constraints:" << to_string(cons));
    // Reduce if needed
    Contraction reduced;
    if (fancy && !cion.no_defract) {
      reduced = ReduceOutputPolynomials(integral_cion, cons);
      SVLOG(cs, 3, "Reduced:\n" << to_string(reduced));
      cons = GatherConstraints(reduced, shapes);
      SVLOG(cs, 3, "Reduced Constraints:" << to_string(cons));
    } else {
      reduced = integral_cion;
    }
    MergeParallelConstraints(&cons);
    SVLOG(cs, 3, "Merged Parallel Constraints:" << to_string(cons));
    // Defract if needed (defract does early return if not required)
    auto defracted = Defract(reduced, cons);
    SVLOG(cs, 3, "Defracted:\n" << to_string(defracted));
    // Gather the constraints from index bounds
    cons = GatherConstraints(defracted, shapes);
    // New parallel constraints might have been introduced by defract; re-merge them
    MergeParallelConstraints(&cons);
    return std::make_pair(defracted, cons);
  }

 private:
  Program parsed_;
  Bindings vars_;
  ShapeMap inputs_;
  ShapeMap outputs_;
  std::set<std::string> externals_;
};

}  // namespace

std::shared_ptr<Block> GenerateStripe(const RunInfo& runinfo) {
  Parser parser;
  auto parsed = parser.Parse(runinfo.code);
  // process reshape
  auto vars = BindProgram(&parsed, runinfo.input_shapes, runinfo.output_shapes);
  StripeGenerator gen(parsed, vars, runinfo.input_shapes, runinfo.output_shapes);
  return gen.Run(runinfo.program_name);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
