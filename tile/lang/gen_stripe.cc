#include "tile/lang/gen_stripe.h"

#include <set>
#include <utility>
#include <vector>

#include <boost/format.hpp>

#include "tile/lang/bound.h"
#include "tile/lang/defract.h"
#include "tile/lang/parser.h"
#include "tile/lang/reduce.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;    // NOLINT
using namespace stripe;  // NOLINT

namespace {

class StripeGenerator {
 public:
  explicit StripeGenerator(const RunInfo& runinfo, bool i8_mode)
      : runinfo_(runinfo),  //
        i8_mode_(i8_mode) {
    if (!runinfo.from_edsl) {
      if (runinfo_.program.ops.empty()) {
        Parser parser;
        runinfo_.program = parser.Parse(runinfo_.code);
      }
      runinfo_.vars = BindProgram(&runinfo_.program, runinfo_.input_shapes, runinfo_.output_shapes);
      EnforceSpecifiedShapes();
    }
  }

  void EnforceSpecifiedShapes() {
    for (auto& out_it : runinfo_.output_shapes) {
      auto var_it = runinfo_.vars.find(out_it.first);
      if (var_it != runinfo_.vars.end()) {
        var_it->second.shape = out_it.second;
      }
    }
  }

  std::shared_ptr<stripe::Program> Run() {
    auto program = std::make_shared<stripe::Program>();
    program->input_shapes = runinfo_.input_shapes;
    program->output_shapes = runinfo_.output_shapes;
    auto entry = program->entry = std::make_shared<stripe::Block>();
    entry->set_tag("program");
    entry->name = runinfo_.program_name;
    IVLOG(1, "Compiling " << runinfo_.program.ops.size() << " ops");
    // The top level block is a 'main' function.
    // In/Out/InOut refinements made on main relate to user supplied inputs and outputs.
    // None refinements made on main relate to temporaries needed for communication between kernels.
    // The list of kernels to execute are the list of blocks defined within main.
    auto main = std::make_shared<Block>();
    main->set_tag("main");
    entry->stmts.push_back(main);
    main->name = "main";
    // Add decls for external inputs/outputs
    AddDecls(entry.get(), main.get(), runinfo_.input_shapes, true);
    AddDecls(entry.get(), main.get(), runinfo_.output_shapes, false);
    // Add decls for temporaries
    for (const auto& item : runinfo_.vars) {
      if (externals_.count(item.first) == 0) {
        const auto& binding = item.second;
        auto shape = AdjustShape(binding.shape);
        if (binding.tag == Binding::TENSOR) {
          std::vector<Affine> access(binding.shape.dims.size());
          Refinement new_ref{
              RefDir::None,  // dir
              "",            // from
              item.first,    // into
              access,        // access
              shape,         // interior_shape
          };
          new_ref.set_tag("tmp");
          entry->refs.emplace(std::move(new_ref));
          Refinement tmp_ref{
              RefDir::InOut,  // dir
              item.first,     // from
              item.first,     // into
              access,         // access
              shape,          // interior_shape
          };
          tmp_ref.set_tag("tmp");
          main->refs.emplace(std::move(tmp_ref));
        }
      }
    }
    // Add kernels to main
    for (size_t op_idx = 0; op_idx < runinfo_.program.ops.size(); op_idx++) {
      const auto& op = runinfo_.program.ops[op_idx];
      if (to_skip_.count(op_idx)) {
        IVLOG(2, "Skipping as already handled: " << op);
        continue;
      }
      IVLOG(2, "Processing: " << op);
      switch (op.tag) {
        case Op::CONTRACTION:
          ProcessContraction(main.get(), op);
          break;
        case Op::FUNCTION:
          if (op.f.is_special()) {
            ProcessSpecial(main.get(), op_idx);
          } else if (op.f.fn == "reshape") {
            ProcessReshape(main.get(), op);
          } else if (op.f.fn == "index") {
            ProcessIndex(entry.get(), main.get(), op);
          } else {
            ProcessElementwise(entry.get(), main.get(), op);
          }
          break;
        case Op::CONSTANT:
          // Do nothing -- these are handed by constant propagation
          break;
      }
    }
    IVLOG(2, "Done");
    entry->set_attr("total_macs", total_macs_);
    return program;
  }

 private:
  void AddDecls(Block* program, Block* main, const ShapeMap& shapes, bool is_input) {
    for (const auto& item : shapes) {
      externals_.insert(item.first);
      std::vector<Affine> access(item.second.dims.size());
      auto shape = AdjustShape(item.second);
      shape.is_const = IsConst(item.first);
      Refinement new_ref{
          RefDir::None,  // dir
          "",            // from
          item.first,    // into
          access,        // access
          shape,         // interior_shape
      };
      new_ref.set_tag("user");
      program->refs.emplace(std::move(new_ref));
      if (is_input) {
        main->refs.emplace(Refinement{
            RefDir::In,  // dir
            item.first,  // from
            item.first,  // into
            access,      // access
            shape,       // interior_shape
        });
      } else {
        main->refs.emplace(Refinement{
            RefDir::Out,  // dir
            item.first,   // from
            item.first,   // into
            access,       // access
            shape,        // interior_shape
            "",           // agg_op
        });
      }
    }
  }

  std::shared_ptr<Block> InitBuffer(Block* main, const Op& op, const TensorShape& shape) {
    auto stmt = std::make_shared<Block>();
    stmt->set_tag("kernel");
    TensorShape interior_shape{shape.type, std::vector<TensorDimension>(shape.dims.size(), TensorDimension{1, 1})};
    std::vector<Affine> dst_access;
    for (std::size_t idx = 0; idx < shape.dims.size(); ++idx) {
      if (shape.dims[idx].size == 1) {
        dst_access.push_back(0);
      } else {
        auto var = "d" + std::to_string(idx);
        dst_access.push_back(Affine{var});
        stmt->idxs.push_back(Index{var, shape.dims[idx].size});
      }
    }

    stmt->refs.emplace(Refinement{
        RefDir::Out,     // dir
        op.output,       // from
        "dst",           // into
        dst_access,      // access
        interior_shape,  // interior_shape
    });

    if (op.c.use_default.empty()) {
      stmt->set_tag("zero");
      stmt->name = op.output + " = Zero()";
      stmt->comments = "Zero " + op.output;
      stmt->stmts.emplace_back(std::make_shared<Constant>("$ZERO", INT64_C(0)));
      stmt->stmts.emplace_back(std::make_shared<Store>("$ZERO", "dst"));
    } else {
      stmt->set_tag("copy");
      stmt->name = op.output + " = " + op.c.use_default;
      stmt->comments = "Pre-Initialize " + op.output;
      stmt->refs.emplace(Refinement{
          RefDir::In,        // dir
          op.c.use_default,  // from
          "src",             // into
          dst_access,        // access
          interior_shape,    // interior_shape
          "",                // agg_op
          {},                // location
          true               // is_const
      });
      stmt->stmts.emplace_back(std::make_shared<Load>("src", "$X"));
      stmt->stmts.emplace_back(std::make_shared<Store>("$X", "dst"));
    }
    return stmt;
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

    auto kernel = AddKernel(main, op);
    auto agg_op = GetAggOp(cion.agg_op);
    kernel->set_tag("contraction");
    kernel->set_tag("agg_op_" + agg_op);
    std::vector<std::string> scalar_inputs;
    kernel->name += "(";
    std::string out_ref_name = "";
    for (size_t i = 0; i < cion.specs.size(); i++) {
      const auto& spec = cion.specs[i];
      auto interior_shape = ScalarShape(spec.id);
      std::vector<Affine> access;
      for (const auto& poly : spec.spec) {
        access.emplace_back(Integerize(poly, bounds));
      }
      if (i == 0) {
        kernel->refs.emplace(Refinement{
            RefDir::Out,     // dir
            spec.id,         // from
            spec.id,         // into
            access,          // access
            interior_shape,  // interior_shape
            agg_op,          // agg_op
        });
        out_ref_name = spec.id;
      } else {
        if (i != 1) {
          kernel->name += ",";
        }
        kernel->name += spec.id;
        std::string sname = kernel->unique_ref_name(spec.id);
        auto scalar_name = ScalarName(sname);
        scalar_inputs.push_back(scalar_name);
        // if this is a constant, propagate it into the load statement
        const auto& src = runinfo_.vars.find(spec.id);  // TODO: Better name
        if (src != runinfo_.vars.end()) {
          if (src->second.tag == Binding::FCONST) {
            kernel->stmts.push_back(std::make_shared<Constant>(scalar_name, src->second.fconst));
            continue;
          } else if (src->second.tag == Binding::ICONST) {
            kernel->stmts.push_back(std::make_shared<Constant>(scalar_name, src->second.iconst));
            continue;
          }
        }
        // otherwise fall through and do a normal load
        Refinement ref{
            RefDir::In,      // dir
            spec.id,         // from
            sname,           // into
            access,          // access
            interior_shape,  // interior_shape
        };
        ref.set_tag("contraction");
        kernel->refs.emplace(std::move(ref));
        // LOAD
        kernel->stmts.push_back(std::make_shared<Load>(sname, scalar_name));
      }
    }
    kernel->name += ")";

    for (const auto& kvp : bounds) {
      uint64_t range = kvp.second.max - kvp.second.min + 1;
      kernel->idxs.emplace_back(Index{kvp.first, range});
    }
    for (const auto& constraint : simple_cons) {
      auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
      lhs -= constraint.rhs;                           // lhs <= 0;
      lhs = -lhs;                                      // lhs >= 0
      kernel->constraints.emplace_back(lhs);
    }

    if (NeedsInitialize(*kernel, out_ref_name, shapes[0])) {
      auto stmt = InitBuffer(main, op, shapes[0]);
      main->stmts.insert(std::prev(main->stmts.end()), stmt);
      auto ref_it = kernel->ref_by_into(out_ref_name);
      ref_it->mut().dir = RefDir::InOut;
      ref_it->mut().set_tag("initialized");
    }

    // Combination Op
    auto output_type = GetShape(op.output).type;
    tile::DataType input_based_type = tile::DataType::INVALID;
    for (const auto& input : op.inputs) {
      auto input_type = GetShape(input).type;
      input_based_type = CommonSupertype(input_type, input_based_type);
    }
    if (scalar_inputs.size() > 1) {
      if (cion.comb_op == CombinationOp::COND) {
        kernel.get()->stmts.push_back(std::make_shared<Constant>("$ZERO", 0.0));
        AddIntrinsic(kernel.get(), Intrinsic::EQ, input_based_type, {scalar_inputs[0], scalar_inputs[1]}, {"$IS_EQ"});
        AddIntrinsic(kernel.get(), Intrinsic::COND, output_type, {"$IS_EQ", scalar_inputs[2], "$ZERO"},
                     {ScalarName(op.output)});
        kernel->set_tag("comb_op_cond");
      } else {
        auto combo_op = GetComboOp(cion.comb_op);
        if (!combo_op.empty()) {
          AddIntrinsic(kernel.get(), combo_op, input_based_type, scalar_inputs, {ScalarName(op.output)});
          kernel->set_tag("comb_op_" + combo_op);
          if (agg_op == Intrinsic::SUM && combo_op == Intrinsic::MUL) {
            total_macs_ += kernel->idxs_product();
          }
        }
      }
    } else {
      AddIntrinsic(kernel.get(), "assign", input_based_type, scalar_inputs, {ScalarName(op.output)});
      // Mark including the agg_op
      kernel->set_tag("agg_op_" + agg_op + "_no_comb_op");
    }

    // STORE
    kernel->stmts.push_back(std::make_shared<Store>(ScalarName(op.output), op.output));
  }

  bool NeedsInitialize(const Block& block, const std::string& out_ref_name, const TensorShape& out_shape) {
    // Check if have a simple output: 1 unique index per dimension, each full range
    // If not, presume we need initialization for safety
    // We assume here that the 0'th refinement is the output refinement
    std::set<std::string> out_idxs;
    for (size_t i = 0; i < out_shape.dims.size(); i++) {
      Affine affine = block.refs.find(out_ref_name)->access[i];
      if (affine == 0 && out_shape.dims[i].size == 1) {
        continue;
      }
      if (affine.constant() != 0 || affine.getMap().size() != 1 || affine.getMap().begin()->second != 1) {
        return true;  // If it's not a single index with a multiplier of 1, bail
      }
      std::string idx = affine.getMap().begin()->first;
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

  void ProcessElementwise(Block* program, Block* main, const Op& op) {
    auto kernel = AddKernel(main, op);
    kernel->set_tag("eltwise");
    kernel->set_tag("eltwise_" + op.f.fn);

    auto out_shape = GetShape(op.output);
    std::vector<Affine> out_access;
    for (std::size_t i = 0; i < out_shape.dims.size(); ++i) {
      Index idx{
          str(boost::format("i%zu") % (i + 1)),  // name
          out_shape.dims[i].size,                // range
      };
      if (out_shape.dims[i].size > 1) {
        out_access.emplace_back(Affine{idx.name});
      } else {
        out_access.emplace_back(Affine{0});
      }
      kernel->idxs.emplace_back(idx);
    }

    kernel->name += "(";
    std::set<std::string> loaded;
    for (size_t i = 0; i < op.inputs.size(); i++) {
      const auto& input = op.inputs[i];
      const auto& binding = runinfo_.vars.at(input);
      IVLOG(2, "  " << input << ": " << binding);
      if (i != 0) {
        kernel->name += ",";
      }
      kernel->name += input;
      if (loaded.count(input)) {
        continue;
      }
      loaded.emplace(input);
      auto shape = AdjustShape(binding.shape);
      switch (binding.tag) {
        case Binding::TENSOR: {
          // Be careful to handle broadcasts
          std::vector<Affine> access;
          int start = (out_shape.dims.size() >= shape.dims.size()) ? 0 : (shape.dims.size() - out_shape.dims.size());
          int idx_offset = out_shape.dims.size() - shape.dims.size();  // can be negative
          for (int i = 0; i < shape.dims.size(); i++) {
            // TODO: Confirm whether i < start case ever exists
            if (i < start) {
              access.emplace_back(Affine{});
            } else {
              const auto& dim = shape.dims[i];
              if (dim.size > 1) {
                access.emplace_back(Affine{kernel->idxs[i + idx_offset].name});
              } else {
                // TODO: confirm that this should be empty Affine rather than affine{0}
                access.emplace_back(Affine{});
              }
            }
          }
          Refinement ref{
              RefDir::In,          // dir
              input,               // from
              input,               // into
              access,              // access
              ScalarShape(input),  // interior_shape
          };
          ref.set_tag("eltwise_" + op.f.fn);
          kernel->refs.emplace(std::move(ref));
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
    kernel->name += ")";

    // Remove unused indexes
    kernel->idxs.erase(
        remove_if(kernel->idxs.begin(), kernel->idxs.end(), [](const Index& idx) { return idx.range == 1; }),
        kernel->idxs.end());

    kernel->refs.emplace(Refinement{
        RefDir::Out,             // dir
        op.output,               // from
        op.output,               // into
        out_access,              // access
        ScalarShape(op.output),  // interior_shape
    });

    // INTRINSIC
    std::vector<std::string> scalar_inputs;
    tile::DataType output_type = tile::DataType::INVALID;
    for (const auto& input : op.inputs) {
      scalar_inputs.push_back(ScalarName(input));
      auto input_type = GetShape(input).type;
      output_type = CommonSupertype(input_type, output_type);
    }

    // Clean up the semantics of cond to be more strict at the Stripe level
    if (op.f.fn == "cond" && GetShape(op.inputs[0]).type != tile::DataType::BOOLEAN) {
      std::string oname = ScalarName(op.inputs[0] + "_cast");
      AddIntrinsic(kernel.get(), "as_bool", tile::DataType::BOOLEAN, {scalar_inputs[0]}, {oname});
      scalar_inputs[0] = oname;
    }

    AddIntrinsic(  //
        kernel.get(), op.f.fn, output_type, scalar_inputs, {ScalarName(op.output)});

    // STORE
    kernel->stmts.push_back(std::make_shared<Store>(ScalarName(op.output), op.output));
  }

  void ProcessSpecial(Block* main, size_t op_idx) {
    const auto& op = runinfo_.program.ops[op_idx];
    if (op.f.fn == "prng_state" || op.f.fn == "prng_value") {
      throw std::runtime_error("prng functions must come in threes");
    }
    if (op.f.fn == "prng_step") {
      ProcessPrng(main, op_idx);
      return;
    }
    if (op.f.fn == "scatter") {
      if (op.inputs.size() != 3) {
        throw std::runtime_error(str(boost::format("scatter needs 3 parameters, actually gets %d") % op.inputs.size()));
      }
      // Initialize the output buffer of scatter
      auto stmt = InitBuffer(main, op, GetShape(op.inputs[2]));
      main->stmts.push_back(stmt);
    }

    auto stmt = std::make_shared<Special>();
    stmt->name = op.f.fn;
    stmt->inputs = op.inputs;
    stmt->outputs = {op.output};
    main->stmts.push_back(stmt);
  }

  void ProcessIndex(Block* program, Block* main, const Op& op) {
    auto kernel = AddKernel(main, op);
    kernel->set_tag("eltwise");
    kernel->set_tag("eltwise_" + op.f.fn);

    auto out_shape = GetShape(op.output);
    std::vector<Affine> out_access;
    for (std::size_t i = 0; i < out_shape.dims.size(); ++i) {
      Index idx{
          str(boost::format("i%zu") % (i + 1)),  // name
          out_shape.dims[i].size,                // range
      };
      if (out_shape.dims[i].size > 1) {
        out_access.emplace_back(Affine{idx.name});
      } else {
        out_access.emplace_back(Affine{0});
      }
      kernel->idxs.emplace_back(idx);
    }

    kernel->name += "(";
    const auto& input = op.inputs[0];
    const auto& input_binding = runinfo_.vars.at(input);
    assert(input_binding.tag == Binding::TENSOR);
    kernel->name += input;

    const auto& dim_binding = runinfo_.vars.at(op.inputs[1]);
    assert(dim_binding.tag == Binding::ICONST);
    const std::string& load_idx_name = kernel->idxs[dim_binding.iconst].name;
    kernel->stmts.push_back(std::make_shared<LoadIndex>(Affine(load_idx_name), ScalarName(op.output)));
    kernel->name += ")";

    // Remove unused indexes
    kernel->idxs.erase(
        remove_if(kernel->idxs.begin(), kernel->idxs.end(), [](const Index& idx) { return idx.range == 1; }),
        kernel->idxs.end());

    // Add the output refinement
    kernel->refs.emplace(Refinement{
        RefDir::Out,             // dir
        op.output,               // from
        op.output,               // into
        out_access,              // access
        ScalarShape(op.output),  // interior shape
        "",                      // agg_op
        {},                      // location
    });

    // STORE
    kernel->stmts.push_back(std::make_shared<Store>(ScalarName(op.output), op.output));
  }

  void ProcessPrng(Block* main, size_t op_idx) {
    const auto& op = runinfo_.program.ops[op_idx];
    auto stmt = std::make_shared<Special>();
    stmt->name = "prng_step";
    stmt->inputs = {op.inputs[0]};
    stmt->outputs = {};
    std::string tup = op.output;
    std::string sout;
    std::string vout;
    size_t sout_pos = 0;
    // Find the other parts
    for (size_t j = op_idx + 1; j < runinfo_.program.ops.size(); j++) {
      const Op& nop = runinfo_.program.ops[j];
      if (nop.f.fn == "prng_state" && nop.inputs.size() == 1 && nop.inputs[0] == tup) {
        sout = nop.output;
        sout_pos = j;
        to_skip_.emplace(j);
      } else if (nop.f.fn == "prng_value" && nop.inputs.size() == 1 && nop.inputs[0] == tup) {
        vout = nop.output;
        to_skip_.emplace(j);
      }
    }
    if (vout == "" && sout == "") {
      return;  // Skip the whole thing
    }
    if (vout == "") {
      // Convert state output to identity
      Op& xop = runinfo_.program.ops[sout_pos];
      xop.f.fn = "ident";
      xop.inputs[0] = op.inputs[0];
      to_skip_.erase(sout_pos);
      return;
    }
    if (sout == "") {
      throw std::runtime_error("prng_step function missing its companions");
    }
    stmt->outputs.push_back(sout);
    stmt->outputs.push_back(vout);
    main->stmts.push_back(stmt);
  }

  void ProcessReshape(Block* main, const Op& op) {
    auto stmt = std::make_shared<Special>();
    stmt->name = op.f.fn;
    stmt->inputs = std::vector<std::string>{op.inputs[0]};
    stmt->outputs = {op.output};
    main->stmts.push_back(stmt);
  }

  std::shared_ptr<Block> AddKernel(Block* parent, const Op& op, const char* prefix = "") {
    auto block = std::make_shared<Block>();
    block->name = str(boost::format("%skernel_%zu") % prefix % parent->stmts.size());
    block->comments = to_string(op);
    block->set_tag("kernel");
    for (const auto& attr : op.attributes) {
      if (attr.name() == "pid" && attr.params_size()) {
        block->name = attr.params(0);
      }
    }
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

  void AddIntrinsic(Block* block, const std::string& name, const DataType& type,
                    const std::vector<std::string>& inputs,  //
                    const std::vector<std::string>& outputs) {
    auto stmt = std::make_shared<Intrinsic>();
    stmt->name = name;
    stmt->inputs = inputs;
    stmt->outputs = outputs;
    stmt->type = type;
    block->stmts.push_back(stmt);
  }

  inline std::string ScalarName(const std::string& name) {  //
    return str(boost::format("$%s") % name);
  }

  TensorShape GetShape(const std::string& name) const {
    auto it = runinfo_.vars.find(name);
    if (it == runinfo_.vars.end()) {
      throw std::runtime_error(str(boost::format("Unknown shape: %s") % name));
    }
    return AdjustShape(it->second.shape);
  }

  TensorShape ScalarShape(const std::string& name) const {
    auto it = runinfo_.vars.find(name);
    if (it == runinfo_.vars.end()) {
      throw std::runtime_error(str(boost::format("Unknown shape: %s") % name));
    }
    TensorShape shape(it->second.shape.type, {});
    for (const auto& dim : it->second.shape.dims) {
      shape.dims.push_back(TensorDimension(dim.stride, 1));
    }
    shape.is_const = IsConst(name);
    shape.layout = it->second.shape.layout;
    return AdjustShape(shape);
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
        result += Affine(term.first, int_value);
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

  bool IsConst(const std::string& name) const {
    // Returns whether the specified tensor input is constant
    return runinfo_.const_inputs.count(name);
  }

  TensorShape AdjustShape(TensorShape shape) const {
    if (i8_mode_) {
      shape.type = DataType::INT8;
    }
    return shape;
  }

 private:
  RunInfo runinfo_;
  std::set<std::string> externals_;
  std::set<size_t> to_skip_;
  bool i8_mode_;
  int64_t total_macs_ = 0;
};

}  // namespace

std::shared_ptr<stripe::Program> GenerateStripe(const RunInfo& runinfo, bool i8_mode) {  //
  return StripeGenerator(runinfo, i8_mode).Run();
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
