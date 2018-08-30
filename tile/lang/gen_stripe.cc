#include "tile/lang/gen_stripe.h"

#include "tile/lang/compile.h"
#include "tile/lang/primitives.h"

namespace vertexai {
namespace tile {
namespace lang {

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

  stripe::proto::Block Run(const std::string& name) {
    program_.set_name(name);
    LOG(INFO) << "Compiling " << parsed_.ops.size() << " ops";
    AddDecls(inputs_);
    AddDecls(outputs_);
    for (size_t i = 0; i < parsed_.ops.size(); i++) {
      const Op& op = parsed_.ops[i];
      switch (op.tag) {
        case Op::CONTRACTION:
          ProcessContraction(op);
          break;
        case Op::FUNCTION:
          ProcessElementwise(op);
          break;
        case Op::CONSTANT:
          break;
      }
    }
    return program_;
  }

 private:
  void AddDecls(const ShapeMap& shapes) {
    for (const auto& item : shapes) {
      auto decl = program_.add_decls();
      decl->set_name(item.first);
      auto shape = decl->mutable_shape();
      shape->set_type(static_cast<shape::proto::TensorShape::DataType>(item.second.type));
      for (const auto& src : item.second.dims) {
        auto dst = shape->add_dimensions();
        dst->set_size(src.size);
        dst->set_stride(src.stride);
      }
    }
  }

  void ProcessContraction(const Op& op) {
    IVLOG(3, "Compiling contraction: " << op << ", vars = " << vars_);
    if (vars_.at(op.output).shape.byte_size() == 0) {
      IVLOG(3, "Contraction output " << op.output << " size==0; skipping");
      return;
    }
    std::vector<Polynomial> out_poly;
    FlatContraction flat = Compile(op.c, MakeTShapes(op.c), &out_poly);
    flat.output = op.output;

    auto kernel = AddKernel();
    kernel->set_comments(flat.comments);
    for (const auto& name : flat.names) {
      kernel->add_index_names(name);
    }
    for (const auto& range : flat.ranges) {
      kernel->add_index_ranges(range);
    }
    for (const auto& constraint : flat.constraints) {
      auto out = kernel->add_constraints();
      for (const auto& lhs : constraint.lhs) {
        out->add_lhs(lhs);
      }
      out->set_rhs(constraint.rhs);
    }

    if (NeedsZero(flat)) {
      if (op.c.use_default.empty()) {
        AddPrimitive(kernel, primitive::ZERO, {op.output}, {op.output});
      } else {
        AddPrimitive(kernel, primitive::COPY, {op.c.use_default}, {op.output});
      }
    }

    std::vector<std::string> scalar_inputs;
    for (size_t i = 1; i < flat.inputs.size(); i++) {
      auto scalar_name = ScalarName(flat.inputs[i]);
      scalar_inputs.push_back(scalar_name);

      auto input = kernel->add_ref_ins();
      input->set_name(flat.inputs[i]);
      IntoAccess(flat.access[i], input->mutable_access());

      // LOAD
      auto stmt = kernel->add_stmts()->mutable_load();
      stmt->set_from(flat.inputs[i]);
      stmt->set_into(scalar_name);
    }

    // Combination Op
    switch (flat.comb_op) {
      case CombinationOp::NONE:
        break;
      case CombinationOp::MULTIPLY:
        AddPrimitive(kernel, primitive::MUL, scalar_inputs, {ScalarName(flat.output)});
        break;
      case CombinationOp::PLUS:
        AddPrimitive(kernel, primitive::ADD, scalar_inputs, {ScalarName(flat.output)});
        break;
      case CombinationOp::EQ:
        AddPrimitive(kernel, primitive::EQ, scalar_inputs, {ScalarName(flat.output)});
        break;
      case CombinationOp::COND:
        AddPrimitive(kernel, primitive::COND, scalar_inputs, {ScalarName(flat.output)});
        break;
    }

    auto output = kernel->add_ref_outs();
    output->set_name(flat.output);
    switch (flat.agg_op) {
      case AggregationOp::SUM:
        output->set_agg_op(primitive::SUM);
        break;
      case AggregationOp::MAX:
        output->set_agg_op(primitive::MAX);
        break;
      case AggregationOp::MIN:
        output->set_agg_op(primitive::MIN);
        break;
      case AggregationOp::PROD:
        output->set_agg_op(primitive::PROD);
        break;
      case AggregationOp::ASSIGN:
        output->set_agg_op(primitive::ASSIGN);
        break;
      case AggregationOp::NONE:
        break;
    }
    IntoAccess(flat.access[0], output->mutable_access());

    // STORE
    auto stmt = kernel->add_stmts()->mutable_store();
    stmt->set_from(ScalarName(flat.output));
    stmt->set_into(flat.output);
  }

  void ProcessElementwise(const Op& op) {
    auto kernel = AddKernel();
    // kernel->set_comments(src.comments);

    const TensorShape& out_shape = vars_.at(op.output).shape;
    for (std::size_t i = 0; i < out_shape.dims.size(); ++i) {
      kernel->add_index_names(printstring("i%zu", i + 1));
      kernel->add_index_ranges(out_shape.dims[i].size);
    }

    for (const auto& input : op.inputs) {
      const TensorShape& shape = vars_.at(input).shape;
      auto ref_in = kernel->add_ref_ins();
      ref_in->set_name(input);
      auto access = ref_in->mutable_access();
      access->set_offset(0);
      for (const auto& dim : shape.dims) {
        access->add_strides(dim.stride);
      }
    }

    auto ref_out = kernel->add_ref_outs();
    ref_out->set_name(op.output);
    auto access = ref_out->mutable_access();
    access->set_offset(0);
    for (const auto& dim : out_shape.dims) {
      access->add_strides(dim.stride);
    }

    // LOAD
    for (const auto& input : op.inputs) {
      auto stmt_load = kernel->add_stmts()->mutable_load();
      stmt_load->set_from(input);
      stmt_load->set_into(ScalarName(input));
    }

    // PRIMITIVE
    auto stmt_core = kernel->add_stmts()->mutable_primitive();
    stmt_core->set_name(op.f.fn);
    for (const auto& input : op.inputs) {
      stmt_core->add_inputs(ScalarName(input));
    }
    stmt_core->add_outputs(ScalarName(op.output));

    // STORE
    auto stmt_store = kernel->add_stmts()->mutable_store();
    stmt_store->set_from(ScalarName(op.output));
    stmt_store->set_into(op.output);
  }

  stripe::proto::Block* AddKernel(const char* prefix = "") {
    auto name = printstring("%skernel_%zu", prefix, program_.stmts_size());
    auto stmt = program_.add_stmts();
    auto block = stmt->mutable_block();
    block->set_name(name);
    return block;
  }

  std::vector<TensorShape> MakeTShapes(const Contraction& con) {
    std::vector<TensorShape> tshapes;
    for (const TensorSpec& spec : con.specs) {
      auto it = vars_.find(spec.id);
      if (it == vars_.end()) {
        IVLOG(1, "Something went wrong: " << vars_);
        throw std::runtime_error(printstring("Unable to find tensor shape for id %s, ug", spec.id.c_str()));
      }
      tshapes.push_back(it->second.shape);
    }
    return tshapes;
  }

  bool NeedsZero(const FlatContraction& flat) {
    std::vector<std::pair<size_t, size_t>> out_pattern;
    if (flat.access[0].offset != 0) {
      return true;
    }
    for (size_t i = 0; i < flat.names.size(); i++) {
      if (flat.access[0].strides[i] == 0) {
        continue;
      }
      if (flat.access[0].strides[i] < 0) {
        return true;
      }  // Don't try to be fancy, fallback
      out_pattern.emplace_back(flat.access[0].strides[i], flat.ranges[i]);
    }
    for (const FlatConstraint& fc : flat.constraints) {
      bool output_only = true;
      for (size_t i = 0; i < flat.names.size(); i++) {
        if (fc.lhs[i] != 0 && flat.access[0].strides[i] == 0) {
          output_only = false;
          break;
        }
      }
      if (output_only) {
        return true;
      }
    }
    std::sort(out_pattern.begin(), out_pattern.end());
    size_t curskip = 1;
    for (const auto& p : out_pattern) {
      if (curskip != p.first) {
        return true;
      }
      curskip *= p.second;
    }
    return curskip != flat.access[0].global_index_limit;
  }

  void IntoAccess(const FlatTensorAccess& src, stripe::proto::BufferAccess* dst) {
    dst->set_offset(src.offset);
    for (const auto& stride : src.strides) {
      dst->add_strides(stride);
    }
  }

  void AddPrimitive(stripe::proto::Block* block,             //
                    const std::string& name,                 //
                    const std::vector<std::string>& inputs,  //
                    const std::vector<std::string>& outputs) {
    auto stmt = block->add_stmts()->mutable_primitive();
    stmt->set_name(name);
    for (const auto& input : inputs) {
      stmt->add_inputs(input);
    }
    for (const auto& output : outputs) {
      stmt->add_outputs(output);
    }
  }

  inline std::string ScalarName(const std::string& name) { return printstring("$%s", name.c_str()); }

 private:
  stripe::proto::Block program_;
  Program parsed_;
  Bindings vars_;
  ShapeMap inputs_;
  ShapeMap outputs_;
};

}  // namespace

stripe::proto::Block GenerateStripe(const std::string& name, const RunInfo& runinfo) {
  Parser parser;
  auto parsed = parser.Parse(runinfo.code);
  auto vars = BindProgram(&parsed, runinfo.input_shapes, runinfo.output_shapes);
  StripeGenerator gen(parsed, vars, runinfo.input_shapes, runinfo.output_shapes);
  return gen.Run(name);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
