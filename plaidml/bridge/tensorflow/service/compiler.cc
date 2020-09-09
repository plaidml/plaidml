/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "plaidml/bridge/tensorflow/service/compiler.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

namespace edsl = plaidml::edsl;
namespace plaidml_op = plaidml::op;

plaidml::Buffer makeBuffer(const plaidml::TensorShape& shape, const void* data) {
  std::string curDevice = ::plaidml::Settings::get("PLAIDML_DEVICE");
  plaidml::Buffer buffer(curDevice, shape);
  buffer.copy_from(data);
  return buffer;
}

static std::unordered_map<xla::PrimitiveType, plaidml::DType> pml_dtype_map = {
    {xla::PRED, plaidml::DType::BOOLEAN},  //
    {xla::S8, plaidml::DType::INT8},       //
    {xla::S16, plaidml::DType::INT16},     //
    {xla::S32, plaidml::DType::INT32},     //
    {xla::S64, plaidml::DType::INT64},     //
    {xla::U8, plaidml::DType::UINT8},      //
    {xla::U16, plaidml::DType::UINT16},    //
    {xla::U32, plaidml::DType::UINT32},    //
    {xla::U64, plaidml::DType::UINT64},    //
    {xla::F16, plaidml::DType::FLOAT16},   //
    {xla::F32, plaidml::DType::FLOAT32},   //
    {xla::F64, plaidml::DType::FLOAT32},   //
};

namespace xla {
namespace plaidml {

static std::unique_ptr<edsl::Program> makeProgram(const std::string& name, const std::vector<edsl::Tensor>& outputs) {
  VLOG(1) << "makeProgram begin";
  auto program = absl::make_unique<edsl::Program>(edsl::ProgramBuilder(name, outputs));
  VLOG(1) << "Generated program:\n" << (*program).str();
  return std::move(program);
}

static std::string legalize_computation_name(const std::string& cname) {
  std::string result;
  for (int i = 0; i < cname.size(); i++) {
    if (cname[i] == '.') {
      result += "_";
    } else {
      result += cname[i];
    }
  }
  return result;
}

PlaidMLCompiler::PlaidMLCompiler() {
  VLOG(1) << "Initializing PlaidMLCompiler";
  ::plaidml::init();
  ::plaidml::edsl::init();
  ::plaidml::op::init();
}

std::string PlaidMLCompiler::HumanString(const Shape& shape) {
  if (shape.IsTuple()) {
    string text = "{";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      absl::StrAppend(&text, prefix, HumanString(elem_shape));
      prefix = ", ";
    }
    text += "}";
    return text;
  }
  std::vector<std::string> dim_elements;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      dim_elements.push_back(absl::StrCat("<=", shape.dimensions(i)));
    } else {
      dim_elements.push_back(absl::StrCat(shape.dimensions(i)));
    }
  }
  return absl::StrCat("{", absl::StrJoin(dim_elements, ","), "}");
}

// Translate HLO Module to EDSL
StatusOr<std::unique_ptr<edsl::Program>> PlaidMLCompiler::ProgramFromHloModule(std::unique_ptr<HloModule> hlo_module) {
  VLOG(2) << "ORIGINAL HLO MODULE:\n" << hlo_module->ToString();

  std::unordered_map<int, edsl::Tensor> instr_map;
  std::unordered_map<int, edsl::Value> tuple_instr_map;
  std::unordered_map<std::string, edsl::Value> fn_returns;

  auto computation_is_addition = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), match::Add(match::Parameter(), match::Parameter()));
  };

  auto computation_is_multiplication = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), match::Multiply(match::Parameter(), match::Parameter()));
  };

  auto computation_is_maximum = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), match::Maximum(match::Parameter(), match::Parameter()));
  };

  auto computation_is_minimum = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), match::Minimum(match::Parameter(), match::Parameter()));
  };

  // TODO: may be unnecessary because TF has the kParameter opcode which instantiates Placeholder creation.
  // std::vector<Tensor> inputs;

  VLOG(1) << "ProgramFromHloModule begin";

  VLOG(1) << "result_shape" << hlo_module->result_shape().ToString();

  if (hlo_module->has_entry_computation()) {
    auto entry_computation = hlo_module->entry_computation();
    VLOG(1) << "PlaidML Entry Computation " + entry_computation->name() + ": num parameters "
            << entry_computation->num_parameters() << " returns "
            << entry_computation->root_instruction()->shape().ToString();
    /*
    for (int i = 0; i < entry_computation->num_parameters(); i++) {
      auto param = entry_computation->parameter_instruction(i);
      auto pshape = param->shape();
      std::vector<int64_t> param_shape;
      auto param_dtype = pml_dtype_map[pshape.element_type()];
      for (int j = 0; j < pshape.dimensions_size(); j++) {
        param_shape.push_back(pshape.dimensions(j));
      }
      inputs.push_back(Placeholder(param_dtype, param_shape));
    }
    */
  }

  for (auto* computation : hlo_module->computations()) {
    VLOG(2) << "Computation name" << computation->name() << " num_parameters " << computation->num_parameters()
            << " returns " << computation->root_instruction()->shape().ToString();
    // TODO: Verify that the computation return type should actuually be Tensor, or Value, or something else...
    // TODO: Add parameters
    // TODO: Replace periods in computation name with underscores to make it a valid function name
    auto function_name = legalize_computation_name(computation->name());
    auto root_instr = computation->root_instruction();
    auto root_instr_id = root_instr->unique_id();
    auto root_instr_shape = root_instr->shape();
    for (auto* instruction : computation->instructions()) {
      VLOG(2) << xla::HloOpcodeString(instruction->opcode()) << " name " << instruction->name() << " id "
              << instruction->unique_id() << " num_operands " << instruction->operand_count();
      VLOG(2) << instruction->OperandsToString(HloPrintOptions());
      VLOG(2) << "instruction returns " << instruction->shape().ToString();
      auto cur_instr_name = instruction->name();
      auto cur_instr_id = instruction->unique_id();
      auto num_operands = instruction->operand_count();
      std::vector<int> operand_ids;
      for (auto i = 0; i < num_operands; i++) {
        int operand_id = instruction->operand(i)->unique_id();
        VLOG(2) << "visiting operand " << operand_id;
        operand_ids.push_back(operand_id);
      }
      // result shape
      auto shape = instruction->shape();
      std::vector<int64_t> dims;
      for (int j = 0; j < shape.dimensions_size(); j++) {
        dims.push_back(shape.dimensions(j));
      }
      auto type = pml_dtype_map[shape.element_type()];
      // metadata
      auto instr_metadata = instruction->metadata();
      auto meta_name = cur_instr_name;
      if (!instr_metadata.op_name().empty()) {
        meta_name = instr_metadata.op_name();
      }
      // TODO: validate that all these general parameters are correct before constructing them into a larger program.
      switch (instruction->opcode()) {
        case HloOpcode::kConstant: {
          auto tshape = ::plaidml::TensorShape(type, dims);
          auto lshape = edsl::LogicalShape(type, dims);
          const Literal& literal = instruction->literal();
          auto buf = makeBuffer(tshape, literal.untyped_data());
          auto op = Constant(lshape, buf, meta_name);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        // Unary eltwise ops, in alphabetical order
        case HloOpcode::kAbs: {
          // Tensor elementwise absolute value
          auto op = plaidml_op::abs(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kCeil: {
          // Tensor elementwise ceiling
          auto op = edsl::ceil(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kCos: {
          // Tensor elementwise cosine
          auto op = edsl::cos(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kExp: {
          // Tensor elementwise exp
          auto op = edsl::exp(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kFloor: {
          // Tensor elementwise floor
          auto op = edsl::floor(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kLog: {
          // Tensor elementwise natural logarithm
          auto op = edsl::log(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kNegate: {
          // Tensor elementwise negate
          auto op = -(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kNot: {
          // Tensor elementwise bitwise NOT
          auto op = ~(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kRsqrt: {
          // Tensor elementwise reciprocal square root
          auto op = 1 / edsl::sqrt(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kSin: {
          // Tensor elementwise sine
          auto op = edsl::sin(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kSqrt: {
          // Tensor elementwise square root
          auto op = edsl::sqrt(instr_map[operand_ids[0]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        // Binary eltwise ops, in alphabetical order
        case HloOpcode::kAnd: {
          // Tensor elementwise bitwise AND
          auto op = instr_map[operand_ids[0]] & instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kAdd: {
          // Tensor elementwise addition
          auto op = instr_map[operand_ids[0]] + instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kCompare: {
          // Tensor elementwise compare
          edsl::Tensor op;
          auto direction = instruction->comparison_direction();
          switch (direction) {
            case ComparisonDirection::kEq: {
              op = instr_map[operand_ids[0]] == instr_map[operand_ids[1]];
              break;
            }
            case ComparisonDirection::kNe: {
              op = instr_map[operand_ids[0]] != instr_map[operand_ids[1]];
              break;
            }
            case ComparisonDirection::kGe: {
              op = instr_map[operand_ids[0]] >= instr_map[operand_ids[1]];
              break;
            }
            case ComparisonDirection::kGt: {
              op = instr_map[operand_ids[0]] > instr_map[operand_ids[1]];
              break;
            }
            case ComparisonDirection::kLe: {
              op = instr_map[operand_ids[0]] <= instr_map[operand_ids[1]];
              break;
            }
            case ComparisonDirection::kLt: {
              op = instr_map[operand_ids[0]] < instr_map[operand_ids[1]];
              break;
            }
            default: {
              VLOG(2) << "Unknown comparison direction";
            }
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kDivide: {
          // Tensor elementwise division
          auto op = instr_map[operand_ids[0]] / instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kMaximum: {
          // Tensor elementwise maximum
          auto op = plaidml_op::maximum(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kMinimum: {
          // Tensor elementwise minimum
          auto op = plaidml_op::minimum(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kMultiply: {
          // Tensor elementwise multiplication
          auto op = instr_map[operand_ids[0]] * instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kOr: {
          // Tensor elementwise bitwise OR
          auto op = instr_map[operand_ids[0]] | instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kPower: {
          // Tensor elementwise pow
          auto op = edsl::pow(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kRemainder: {
          // Tensor elementwise remainder/modulo op
          auto op = instr_map[operand_ids[0]] % instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kSubtract: {
          // Tensor elementwse subtraction
          auto op = instr_map[operand_ids[0]] - instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kXor: {
          // Tensor elementwise bitwise XOR
          auto op = instr_map[operand_ids[0]] ^ instr_map[operand_ids[1]];
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        // Contraction ops
        case HloOpcode::kBatchNormTraining: {
          // Batch normalization for training. Returns batch norm, batch mean, and batch variance
          // Operand 0: input tensor
          // Operand 1: gamma
          // Operand 2: beta
          // Epsilon and Feature Index operands must be accessed by calling the epsilon() and feature_index() functions
          int64_t feature_index = instruction->feature_index();
          VLOG(2) << "Feature index " << feature_index;
          edsl::Value findex = edsl::Value{feature_index};
          float e = instruction->epsilon();
          VLOG(2) << "Epsilon " << e;
          auto batch_mean = plaidml_op::mean(instr_map[operand_ids[0]], findex, true);
          auto batch_variance = plaidml_op::variance(instr_map[operand_ids[0]], findex, true);
          auto batch_norm_denom = edsl::sqrt(batch_variance + e);
          auto batch_norm = ((instr_map[operand_ids[0]] - batch_mean) * instr_map[operand_ids[1]] / batch_norm_denom) +
                            instr_map[operand_ids[2]];
          auto tup = edsl::make_tuple(batch_norm, batch_mean, batch_variance);
          tuple_instr_map.insert(std::make_pair(cur_instr_id, tup));
          break;
        }
        case HloOpcode::kConcatenate: {
          // Tensor concatenation
          std::vector<edsl::Tensor> tensors;
          for (auto i = 0; i < num_operands; i++) {
            tensors.push_back(instr_map[operand_ids[i]]);
          }
          int concat_axis = instruction->concatenate_dimension();
          VLOG(2) << "Concat axis " << concat_axis;
          auto op = plaidml_op::concatenate(tensors, concat_axis);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kConvolution: {
          // Tensor convolution operation
          // TODO: make sure optional operands are passed correctly
          auto op = plaidml_op::convolution(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          auto conv_dnums = instruction->convolution_dimension_numbers();
          auto input_spatial_dims = conv_dnums.input_spatial_dimensions();
          auto kernel_spatial_dims = conv_dnums.kernel_spatial_dimensions();
          // This window, unlike pooling, only takes spatial dimensions into account
          // TODO: smarter layout inference using convolution dnums
          auto raw_window = instruction->window();
          auto spatial_rank = raw_window.dimensions_size();
          std::vector<int> window_size;
          std::vector<int> strides;
          std::vector<int> dilations;
          std::vector<int> pads(2 * spatial_rank);
          for (int i = 0; i < spatial_rank; i++) {
            auto d = raw_window.dimensions()[i];
            VLOG(2) << "Spatial dimension: size " << d.size() << ", stride " << d.stride() << ", low pads "
                    << d.padding_low() << ", high pads " << d.padding_high() << ", base_dilation " << d.base_dilation()
                    << ", window_dilation " << d.window_dilation();
            window_size.push_back(d.size());
            strides.push_back(d.stride());
            dilations.push_back(d.window_dilation());
            pads[i] = d.padding_low();
            pads[i + spatial_rank] = d.padding_high();
          }
          op.filter_shape(window_size)
              .strides(strides)
              .dilations(dilations)
              .autopad_mode(plaidml_op::AutoPadMode::EXPLICIT)
              .manual_padding(pads)
              .name(meta_name);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kDot: {
          // Tensor dot operation
          auto op = plaidml_op::dot(instr_map[operand_ids[0]], instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kIota: {
          int total_dims = 1;
          for (auto i : dims) {
            total_dims *= i;
          }
          std::vector<int> x(total_dims);
          std::iota(x.begin(), x.end(), 0);
          auto tshape = ::plaidml::TensorShape(type, dims);
          auto lshape = edsl::LogicalShape(type, dims);
          auto buf = makeBuffer(tshape, x.data());
          auto op = Constant(lshape, buf);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kPad: {
          // Tensor pad operation
          std::vector<int> low_pads;
          std::vector<int> high_pads;
          auto padding_config = instruction->padding_config();
          // NXC layout, check only the X
          for (int64 i = 0; i < padding_config.dimensions_size(); i++) {
            auto padding_dimension = padding_config.dimensions(i);
            low_pads.push_back(padding_dimension.edge_padding_low());
            high_pads.push_back(padding_dimension.edge_padding_high());
          }
          auto op = plaidml_op::explicit_padding(instr_map[operand_ids[0]], low_pads, high_pads);
          op.mode(plaidml_op::PadMode::CONSTANT);
          op.padval(instr_map[operand_ids[1]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReduce: {
          // with reduce, an axis is specified, so figure out which computation and pass that in to the op lib
          VLOG(2) << "begin processing reduce";
          HloReduceInstruction* reduce = Cast<HloReduceInstruction>(instruction);
          auto reduction_dims = reduce->dimensions();
          std::vector<int> axes;
          for (auto d : reduction_dims) {
            axes.push_back(d);
            VLOG(2) << "Adding axis " << d << " to reduce";
          }
          edsl::Tensor op;
          auto applied_computation = instruction->to_apply();
          if (computation_is_maximum(applied_computation)) {
            VLOG(2) << "Reached condition: max with reduce";
            op = plaidml_op::max(instr_map[operand_ids[0]], edsl::make_tuple(axes));
          } else if (computation_is_minimum(applied_computation)) {
            VLOG(2) << "Reached condition: min with reduce";
            op = plaidml_op::min(instr_map[operand_ids[0]], edsl::make_tuple(axes));
          } else if (computation_is_addition(applied_computation)) {
            VLOG(2) << "Reached condition: add with reduce";
            op = plaidml_op::sum(instr_map[operand_ids[0]], edsl::make_tuple(axes));
          } else if (computation_is_multiplication(applied_computation)) {
            VLOG(2) << "Reached condition: prod with reduce";
            op = plaidml_op::prod(instr_map[operand_ids[0]], edsl::make_tuple(axes));
          } else {
            VLOG(2) << "Unknown reduction type recieved";
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReduceWindow: {
          // This is some sort of aggregation. If the aggregation op is max, then it's a max pool
          // TODO: Find a better way to calculate spatial dimensions
          VLOG(2) << "begin processing reduce-window";
          edsl::Tensor op;
          // Default pool mode and pad mode
          plaidml_op::PoolMode pm = plaidml_op::PoolMode::AVG;
          plaidml_op::AutoPadMode am = plaidml_op::AutoPadMode::EXPLICIT;
          // This window is in NXC format: for example, a window of 1x2x2x1 translates to {2, 2} in the pooling op
          // Spatial rank in this case has to take NXC into account, so subtract 2
          auto raw_window = instruction->window();
          auto spatial_rank = raw_window.dimensions_size() - 2;
          std::vector<int> window_size;
          std::vector<int> strides;
          std::vector<int> pads(2 * spatial_rank);
          for (int i = 0; i < spatial_rank; i++) {
            auto d = raw_window.dimensions()[i + 1];
            VLOG(2) << "Spatial dimension: size " << d.size() << ", stride " << d.stride() << ", low pads "
                    << d.padding_low() << ", high pads " << d.padding_high();
            window_size.push_back(d.size());
            strides.push_back(d.stride());
            pads[i] = d.padding_low();
            pads[i + spatial_rank] = d.padding_high();
          }
          auto applied_computation = instruction->to_apply();
          if (computation_is_maximum(applied_computation)) {
            VLOG(2) << "Reached condition: max pooling";
            pm = plaidml_op::PoolMode::MAX;
          } else if (computation_is_addition(applied_computation)) {
            VLOG(2) << "Reached condition: sum pooling";
            pm = plaidml_op::PoolMode::SUM;
          }
          // TODO: Add conditions for avg pooling, sum pooling, etc.
          op = plaidml_op::pool(instr_map[operand_ids[0]], pm, window_size, strides, am, pads,
                                plaidml_op::TensorLayout::NXC);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kBroadcast: {
          auto op = instr_map[operand_ids[0]];
          std::vector<int> result_shape(begin(dims), end(dims));
          std::vector<int> bcast_dims(begin(instruction->dimensions()), end(instruction->dimensions()));
          op = plaidml_op::broadcast(instr_map[operand_ids[0]], result_shape, bcast_dims);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kConvert: {
          // Equivalent to cast
          auto op = edsl::cast(instr_map[operand_ids[0]], type);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kParameter: {
          // Tensor inputs, create a placeholder
          auto op = edsl::Placeholder(type, dims);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReshape: {
          // Tensor reshape operation
          auto op = edsl::reshape(instr_map[operand_ids[0]], dims);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kReverse: {
          // Reverse in XLA is Flip in Numpy/Op Lib
          auto op = instr_map[operand_ids[0]];
          auto instr_dims = instruction->dimensions();
          for (auto dim : instr_dims) {
            // Can only flip on one axis at a time
            VLOG(2) << "Performing reverse on dimension " << dim;
            op = plaidml_op::flip(op, dim);
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kSelect: {
          // Tensor select conditional operation
          auto op = edsl::select(instr_map[operand_ids[0]], instr_map[operand_ids[1]], instr_map[operand_ids[2]]);
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kSlice: {
          // Tensor slice operation
          auto op = plaidml_op::slice(instr_map[operand_ids[0]]);
          // Grab start points, end points, and strides
          for (auto i = 0; i < dims.size(); i++) {
            VLOG(2) << "Slicing dimension " << i << " starting at index " << instruction->slice_starts(i)
                    << " ending at index " << instruction->slice_limits(i) << " with stride "
                    << instruction->slice_strides(i);
            op.add_dim(instruction->slice_starts(i), instruction->slice_limits(i), instruction->slice_strides(i));
          }
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kTranspose: {
          // Tensor transpose operation
          auto dims = instruction->dimensions();
          std::vector<edsl::Value> pattern;
          for (int64_t dim : dims) {
            pattern.push_back(edsl::Value(dim));
          }
          auto op = plaidml_op::transpose(instr_map[operand_ids[0]], edsl::Value(pattern));
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kTuple: {
          // a kTuple operation is kind of like make_tuple in EDSL
          std::vector<edsl::Value> tup_operands;
          for (int i = 0; i < num_operands; i++) {
            tup_operands.push_back(edsl::Value(instr_map[operand_ids[i]]));
          }
          auto op = edsl::make_tuple(tup_operands);
          tuple_instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kGetTupleElement: {
          // a kGetTupleElement operation is like taking an element in a Value and interpreting it as a tensor, int,
          // etc.
          // TODO: Handle return type
          auto tindex = instruction->tuple_index();
          auto op = tuple_instr_map[operand_ids[0]].as_tuple()[tindex].as_tensor();
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kGetDimensionSize: {
          auto dim_operand = instruction->dimension();
          VLOG(2) << "Getting dimension size at axis " << dim_operand;
          int64_t operand_shape = instruction->operand(0)->shape().dimensions(dim_operand);
          VLOG(2) << "DImension size at axis " << dim_operand << " is " << operand_shape;
          auto op = edsl::Tensor{operand_shape};
          instr_map.insert(std::make_pair(cur_instr_id, op));
          break;
        }
        case HloOpcode::kCall: {
          // Call another EDSL function
          auto computation_to_apply = instruction->to_apply();
          auto computation_name = legalize_computation_name(computation_to_apply->name());
          auto op = fn_returns[computation_name];
          instr_map.insert(std::make_pair(cur_instr_id, op.as_tensor()));
          break;
        }
        // TODO: Unary ops.
        case HloOpcode::kRoundNearestAfz:
        case HloOpcode::kBitcast:
        case HloOpcode::kClz:
        case HloOpcode::kCopy:
        case HloOpcode::kCopyStart:
        case HloOpcode::kCopyDone:
        case HloOpcode::kExpm1:
        case HloOpcode::kImag:
        case HloOpcode::kIsFinite:
        case HloOpcode::kLog1p:
        case HloOpcode::kPopulationCount:
        case HloOpcode::kReal:
        case HloOpcode::kSign:
        case HloOpcode::kTanh: {
          // Parse operands.
          VLOG(2) << "Unimplemented unary op " << cur_instr_name << " has been called here\n";
          break;
        }
        // Binary ops.
        case HloOpcode::kAtan2:
        case HloOpcode::kComplex:
        case HloOpcode::kShiftLeft:
        case HloOpcode::kShiftRightArithmetic:
        case HloOpcode::kShiftRightLogical: {
          // Parse operands.
          VLOG(2) << "Unimplemented binary op " << cur_instr_name << " has been called here\n";
          break;
        }
        // TODO: special instructions
        default:
          VLOG(2) << "Unknown op " << cur_instr_name << " (opcode " << HloOpcodeString(instruction->opcode())
                  << ") has been called here\n";
          break;
      }
    }
    if (root_instr_shape.IsArray()) {
      fn_returns.insert(std::make_pair(function_name, instr_map[root_instr_id]));
    } else {
      fn_returns.insert(std::make_pair(function_name, tuple_instr_map[root_instr_id]));
    }
  }

  edsl::Tensor output;

  if (hlo_module->has_entry_computation()) {
    output = fn_returns[legalize_computation_name(hlo_module->entry_computation()->name())].as_tensor();
  }

  auto program = makeProgram("hlo_module", {output});
  VLOG(1) << "ProgramFromHloModule complete";
  return std::move(program);
}

}  // namespace plaidml
}  // namespace xla
