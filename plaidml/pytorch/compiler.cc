// Copyright 2019, Intel Corporation.

#include "plaidml/pytorch/compiler.h"

#include "plaidml/pytorch/logging.h"

using namespace torch::jit;  // NOLINT
namespace edsl = vertexai::plaidml::edsl;

using OpFunction = std::function<edsl::Tensor(const std::vector<edsl::Tensor>& args)>;

template <typename X, typename Y>
inline constexpr auto ceil_div(X x, Y y) -> decltype((x + y - 1) / y) {
  return (x + y - 1) / y;
}

namespace ops {

// aten::addmm(
//   Tensor self,
//   Tensor mat1,
//   Tensor mat2,
//   *,
//   Scalar beta,
//   Scalar alpha
// ) -> Tensor;
edsl::Tensor addmm(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "addmm");

  const auto& A = args[0];
  IVLOG(2, "  A: " << A.shape().str());

  const auto& B = args[1];
  IVLOG(2, "  B: " << B.shape().str());

  const auto& C = args[2];
  IVLOG(2, "  C: " << C.shape().str());

  const auto& beta = args[3];
  IVLOG(2, "  beta: " << beta.str());

  const auto& alpha = args[4];
  IVLOG(2, "  alpha: " << alpha.str());

  edsl::TensorDim I, J, K;
  edsl::TensorIndex i, j, k;
  B.bind_dims(I, K);
  C.bind_dims(K, J);
  auto O = edsl::TensorOutput(I, J);
  O(i, j) += B(i, k) * C(k, j);
  return beta * A + alpha * O;
}

// aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor
edsl::Tensor add(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "add");
  IVLOG(2, "  " << args[0].shape().str() << " + " << args[1].shape().str() << " * " << args[2].str());
  return args[0] + args[1] * args[2];
}

// aten::mul(Tensor self, Scalar other) -> Tensor
edsl::Tensor mul(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "mul");
  IVLOG(2, "  " << args[0].shape().str() << " * " << args[1].shape().str());
  return args[0] * args[1];
}

// aten::batch_norm(
//   Tensor input,
//   Tensor? weight,
//   Tensor? bias,
//   Tensor? running_mean,
//   Tensor? running_var,
//   bool training,
//   float momentum,
//   float eps,
//   bool cudnn_enabled
// ) -> Tensor
edsl::Tensor batch_norm(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "batch_norm");

  const auto& I = args[0];  // input
  IVLOG(2, "  I: " << I.shape().str());

  edsl::TensorDim N, C, H, W;
  I.bind_dims(N, C, H, W);

  std::vector<int64_t> dims{static_cast<int64_t>(N.value()), static_cast<int64_t>(C.value()), 1, 1};
  IVLOG(2, "  Weight: " << args[1].str());
  auto Weight = args[1];  // weight
  if (!Weight.is_none()) {
    Weight = edsl::reshape(Weight, dims);
  }

  IVLOG(2, "  Bias: " << args[2].str());
  auto Bias = args[2];  // bias
  if (!Bias.is_none()) {
    Bias = edsl::reshape(Bias, dims);
  }

  IVLOG(2, "  Mean: " << args[3].str());
  auto Mean = edsl::reshape(args[3], dims);  // running_mean

  IVLOG(2, "  Var: " << args[4].str());
  auto Var = edsl::reshape(args[4], dims);  // running_var

  auto is_training = args[5].as_int();  // training
  IVLOG(2, "  is_training: " << is_training);

  auto momentum = args[6].as_float();  // momentum
  IVLOG(2, "  momentum: " << momentum);

  auto epsilon = args[7].as_float();  // eps
  IVLOG(2, "  epsilon: " << epsilon);

  auto O = (I - Mean);
  if (!Weight.is_none()) {
    O = O * Weight;
  }
  O = O / edsl::sqrt(Var + epsilon);
  if (!Bias.is_none()) {
    O = O + Bias;
  }
  return O;
}

// aten::_convolution(
//   Tensor input,
//   Tensor weight,
//   Tensor? bias,
//   int[] stride,
//   int[] padding,
//   int[] dilation,
//   bool transposed,
//   int[] output_padding,
//   int groups,
//   bool benchmark,
//   bool deterministic,
//   bool cudnn_enabled
// ) -> Tensor
edsl::Tensor convolution(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "convolution");

  const auto& I = args[0];  // input
  IVLOG(2, "  I: " << I.shape().str());

  const auto& K = args[1];  // weight
  IVLOG(2, "  K: " << K.str());

  const auto& bias = args[2];  // bias
  IVLOG(2, "  bias: " << bias.str());

  const auto& S = args[3].as_tuple();  // stride
  IVLOG(2, "  strides: " << args[3].str());

  const auto& P = args[4].as_tuple();  // padding
  IVLOG(2, "  padding: " << args[4].str());

  const auto& D = args[5].as_tuple();  // dilation
  IVLOG(2, "  dilation: " << args[5].str());

  const auto& is_transposed = args[6];  // transposed
  IVLOG(2, "  is_transposed: " << is_transposed.str());

  const auto& output_padding = args[7];  // output_padding
  IVLOG(2, "  output_padding: " << output_padding.str());

  const auto& groups = args[8];  // groups
  IVLOG(2, "  groups: " << groups.str());

  auto rank = S.size();

  edsl::TensorDim N, CI, CO;
  edsl::TensorIndex n, ci, co;

  std::vector<edsl::TensorDim> I_spatial_dims(rank);
  std::vector<edsl::TensorDim> I_dims = {N, CI};
  std::vector<edsl::TensorIndex> I_idxs = {n, ci};
  I_dims.insert(std::end(I_dims), std::begin(I_spatial_dims), std::end(I_spatial_dims));
  I.bind_dims(I_dims);

  std::vector<edsl::TensorDim> K_spatial_dims(rank);
  std::vector<edsl::TensorDim> K_dims = {CO, CI};
  std::vector<edsl::TensorIndex> K_idxs = {co, ci};
  K_dims.insert(std::end(K_dims), std::begin(K_spatial_dims), std::end(K_spatial_dims));
  K.bind_dims(K_dims);

  std::vector<edsl::TensorDim> O_dims = {N, CO};
  std::vector<edsl::TensorIndex> O_idxs = {n, co};

  for (size_t i = 0; i < rank; i++) {
    edsl::TensorIndex x;
    edsl::TensorIndex k;
    I_idxs.emplace_back(S[i].as_int() * x + D[i].as_int() * k - P[i].as_int());
    O_idxs.push_back(x);
    K_idxs.push_back(k);
    O_dims.emplace_back(
        (I_spatial_dims[i] + 2 * P[i].as_int() - D[i].as_int() * (K_spatial_dims[i] - 1) - 1) / S[i].as_int() + 1);
  }

  edsl::Tensor O(O_dims);
  O(O_idxs) += I(I_idxs) * K(K_idxs);
  if (!bias.is_none()) {
    std::vector<int64_t> dims{static_cast<int64_t>(N.value()), static_cast<int64_t>(CO.value()), 1, 1};
    O = O + edsl::reshape(bias, dims);
  }
  return O;
}

// aten::avg_pool2d(
//   Tensor self,
//   int[2] kernel_size,
//   int[2] stride=[],
//   int[2] padding=0,
//   bool ceil_mode=False,
//   bool count_include_pad=True
// ) -> Tensor;
edsl::Tensor avg_pool2d(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "avg_pool2d");

  const auto& I = args[0];
  IVLOG(2, "  I: " << I.shape().str());

  const auto& kernel_size = args[1].as_tuple();
  IVLOG(2, "  kernel_size: " << args[1].str());

  auto strides = args[2].as_tuple();
  IVLOG(2, "  strides: " << args[2].str());
  if (strides.empty()) {
    strides = kernel_size;
  }

  const auto& padding = args[3].as_tuple();
  IVLOG(2, "  padding: " << args[3].str());

  edsl::TensorDim N, C, H, W;
  edsl::TensorIndex n, c, h, w, i, j;
  I.bind_dims(N, C, H, W);
  auto P0 = padding[0].as_int();
  auto P1 = padding[1].as_int();
  auto K0 = kernel_size[0].as_int();
  auto K1 = kernel_size[1].as_int();
  auto S0 = strides[0].as_int();
  auto S1 = strides[1].as_int();
  auto O = edsl::TensorOutput(N, C, (H + 2 * P0 - K0) / S0 + 1, (W + 2 * P1 - K1) / S1 + 1);
  if (i < K0 && j < K1) {
    O(n, c, h, w) += I(n, c, S0 * h + i, S1 * w + j);
  }
  return O / (K0 * K1);
}

// aten::adaptive_avg_pool2d(
//   Tensor self,
//   int[2] output_size
// ) -> Tensor
edsl::Tensor adaptive_avg_pool2d(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "adaptive_avg_pool2d");

  const auto& I = args[0];
  IVLOG(2, "  I: " << I.shape().str());

  const auto& output_size = args[1].as_tuple();
  IVLOG(2, "  output_size: " << args[1].str());

  edsl::TensorDim N, C, H, W;
  I.bind_dims(N, C, H, W);

  auto X0 = static_cast<int64_t>(H.value());
  auto X1 = static_cast<int64_t>(W.value());
  auto P0 = 0;
  auto P1 = 0;
  // auto K0 = ceil_div(X0, output_size[0].as_int());
  // auto K1 = ceil_div(X1, output_size[1].as_int());
  // auto S0 = K0;
  // auto S1 = K1;
  auto S0 = ceil_div(X0, output_size[0].as_int());
  auto S1 = ceil_div(X1, output_size[1].as_int());
  auto K0 = X0 - (output_size[0].as_int() - 1) * S0;
  auto K1 = X1 - (output_size[1].as_int() - 1) * S1;

  return avg_pool2d({
      I,                         // self
      edsl::make_tuple(K0, K1),  // kernel_size
      edsl::make_tuple(S0, S1),  // stride
      edsl::make_tuple(P0, P1),  // padding
      edsl::Tensor(0),           // ceil_mode
      edsl::Tensor(1)            // count_include_pad
  });
}

// aten::max_pool2d(
//   Tensor self,
//   int[2] kernel_size,
//   int[2] stride=[],
//   int[2] padding=0,
//   int[2] dilation=1,
//   bool ceil_mode=False
// ) -> Tensor;
edsl::Tensor max_pool2d(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "max_pool2d");

  const auto& I = args[0];
  IVLOG(2, "  I: " << I.shape().str());

  const auto& kernel_size = args[1].as_tuple();
  IVLOG(2, "  kernel_size: " << args[1].str());

  auto strides = args[2].as_tuple();
  IVLOG(2, "  strides: " << args[2].str());
  if (strides.empty()) {
    strides = kernel_size;
  }

  const auto& padding = args[3].as_tuple();
  IVLOG(2, "  padding: " << args[3].str());

  const auto& dilation = args[4].as_tuple();
  IVLOG(2, "  dilation: " << args[4].str());

  auto ceil_mode = args[5].as_int();
  IVLOG(2, "  ceil_mode: " << ceil_mode);

  edsl::TensorDim N, C, H, W;
  edsl::TensorIndex n, c, h, w, k0, k1;
  I.bind_dims(N, C, H, W);
  auto P0 = padding[0].as_int();
  auto P1 = padding[1].as_int();
  auto K0 = kernel_size[0].as_int();
  auto K1 = kernel_size[1].as_int();
  auto S0 = strides[0].as_int();
  auto S1 = strides[1].as_int();
  auto D0 = dilation[0].as_int();
  auto D1 = dilation[1].as_int();
  auto O = edsl::TensorOutput(N, C,                                       //
                              (H + 2 * P0 - D0 * (K0 - 1) - 1) / S0 + 1,  //
                              (W + 2 * P1 - D1 * (K1 - 1) - 1) / S1 + 1);
  if (k0 < K0 && k1 < K1) {
    O(n, c, h, w) >= I(n, c, S0 * h + k0 - P0, S1 * w + k1 - P1);
  }
  return O;
}

// aten::relu(Tensor self) -> Tensor
edsl::Tensor relu(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "relu");

  const auto& I = args[0];
  IVLOG(2, "  I: " << I.shape().str());

  edsl::Tensor Z{0.0};
  return edsl::select(I < Z, Z, I);
}

// aten::reshape(Tensor self, int[] shape) -> Tensor
edsl::Tensor reshape(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "reshape");

  const auto& I = args[0];
  auto I_shape = I.shape();
  IVLOG(2, "  I: " << I_shape.str());

  const auto& shape = args[1].as_tuple();
  IVLOG(2, "  shape: " << args[1].str());

  // reshape((6, 6), (3, -1)) -> reshape((6, 6), (3, 12))
  // reshape((2, 3, 4), (2, -1)) -> reshape((2, 3, 4), (2, 12))
  int64_t product = 1;
  for (size_t i = 0; i < I_shape.rank(); i++) {
    if (i < shape.size()) {
      if (shape[i].as_int() == -1) {
        product *= I_shape.size_at(i);
      } else {
        product *= ceil_div(I_shape.size_at(i), shape[i].as_int());
      }
    } else {
      product *= I_shape.size_at(i);
    }
  }

  std::vector<int64_t> dims;
  for (const auto& dim : shape) {
    auto dim_int = dim.as_int();
    if (dim_int == -1) {
      dims.push_back(product);
    } else {
      dims.push_back(dim_int);
    }
  }

  return edsl::reshape(I, dims);
}

// aten::t(Tensor self) -> Tensor
edsl::Tensor transpose(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "transpose");

  const auto& I = args[0];
  IVLOG(2, "  I: " << I.shape().str());

  edsl::TensorDim X, Y;
  edsl::TensorIndex x, y;
  I.bind_dims(X, Y);
  auto O = edsl::TensorOutput(Y, X);
  O(y, x) = I(x, y);
  return O;
}

// aten::linear(Tensor input, Tensor weight, Tensor? bias) -> Tensor
edsl::Tensor linear(const std::vector<edsl::Tensor>& args) {
  IVLOG(1, "linear");

  const auto& input = args[0];
  IVLOG(2, "  input: " << input.shape().str());

  const auto& weight = args[1];
  IVLOG(2, "  weight: " << weight.str());

  const auto& bias = args[2];
  IVLOG(2, "  bias: " << bias.str());

  edsl::TensorDim I, J, K;
  edsl::TensorIndex i, j, k;
  input.bind_dims(I, K);
  weight.bind_dims(J, K);
  auto O = edsl::TensorOutput(I, J);
  O(i, j) += input(i, k) * weight(j, k);
  if (!bias.is_none()) {
    O = bias + O;
  }
  return O;
}

}  // namespace ops

std::map<Symbol, OpFunction> kSupportedOps = {
    {Symbol::fromQualString("aten::addmm"), ops::addmm},                              //
    {Symbol::fromQualString("aten::add"), ops::add},                                  //
    {Symbol::fromQualString("aten::add_"), ops::add},                                 //
    {Symbol::fromQualString("aten::_convolution"), ops::convolution},                 //
    {Symbol::fromQualString("aten::adaptive_avg_pool2d"), ops::adaptive_avg_pool2d},  //
    {Symbol::fromQualString("aten::avg_pool2d"), ops::avg_pool2d},                    //
    {Symbol::fromQualString("aten::batch_norm"), ops::batch_norm},                    //
    {Symbol::fromQualString("aten::linear"), ops::linear},                            //
    {Symbol::fromQualString("aten::mul"), ops::mul},                                  //
    {Symbol::fromQualString("aten::max_pool2d"), ops::max_pool2d},                    //
    {Symbol::fromQualString("aten::relu"), ops::relu},                                //
    {Symbol::fromQualString("aten::relu_"), ops::relu},                               //
    {Symbol::fromQualString("aten::reshape"), ops::reshape},                          //
    {Symbol::fromQualString("aten::t"), ops::transpose},                              //
    // Symbol::fromQualString("aten::threshold_"),           //
};

const at::Symbol Compiler::symbol = Symbol::fromQualString("plaidml::CompilationGroup");

Compiler::Compiler(const vertexai::plaidml::device& device, const Node* node)
    : device_(device),                      //
      subgraph_(node->g(attr::Subgraph)) {  //
}

bool Compiler::is_supported(Node* node) {
  IVLOG(2, "Compiler::is_supported> " << node->kind().toQualString());
  if (node->kind() == Compiler::symbol) {
    return true;
  }
  switch (node->kind()) {
    case prim::Constant:
      return true;
    case prim::Param:
      return false;
    default:
      break;
  }
  auto it = kSupportedOps.find(node->kind());
  if (it != kSupportedOps.end()) {
    return true;
  }
  IVLOG(1, "NOT is_supported: " << node->kind().toQualString());
  return false;
}

void Compiler::run(Stack* stack) {
  IVLOG(1, "Compiler::run>");
  size_t num_inputs = subgraph_->inputs().size();
  at::ArrayRef<IValue> inputs = last(*stack, num_inputs);

  CompleteArgumentSpec spec{false, inputs};
  auto it = cache_.find(spec);
  if (it == cache_.end()) {
    std::tie(it, std::ignore) = cache_.emplace(spec, compile(&inputs));
  }

  auto outputs = it->second->run(&inputs);

  drop(*stack, num_inputs);
  for (const auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack->push_back(IValue(var));
  }
}

std::shared_ptr<Executable> Compiler::compile(at::ArrayRef<IValue>* inputs) {
  IVLOG(1, "Compiler::compile>");
  std::vector<edsl::Tensor> input_tensors;
  std::unordered_map<const Value*, edsl::Tensor> tensors_by_value;
  for (size_t i = 0; i < inputs->size(); i++) {
    const auto& ival = inputs->at(i);
    if (!ival.isTensor()) {
      throw std::runtime_error("Unexpected non-tensor input");
    }
    const auto& tensor = ival.toTensor();
    std::vector<uint64_t> sizes;
    for (const auto& size : tensor.sizes()) {
      sizes.emplace_back(size);
    }
    // TODO: convert dtype
    edsl::TensorShape shape(PLAIDML_DATA_FLOAT32, sizes);
    edsl::Tensor input_tensor(shape);
    input_tensors.push_back(input_tensor);
    const auto& input = subgraph_->inputs()[i];
    tensors_by_value.emplace(input, input_tensor);
  }

  for (auto node : subgraph_->nodes()) {
    switch (node->kind()) {
      case prim::Constant: {
        auto opt_ivalue = toIValue(node->output());
        if (!opt_ivalue.has_value()) {
          throw std::runtime_error("prim::Constant has no value");
        }
        auto ivalue = opt_ivalue.value();
        if (ivalue.isDouble()) {
          tensors_by_value.emplace(node->output(), edsl::Tensor(ivalue.toDouble()));
        } else if (ivalue.isInt()) {
          tensors_by_value.emplace(node->output(), edsl::Tensor(ivalue.toInt()));
        } else if (ivalue.isBool()) {
          tensors_by_value.emplace(node->output(), edsl::Tensor(ivalue.toBool()));
        } else if (ivalue.isNone()) {
          tensors_by_value.emplace(node->output(), edsl::None());
        } else if (ivalue.isIntList()) {
          std::vector<edsl::Tensor> elts;
          for (const auto& elt : ivalue.toIntList()) {
            elts.emplace_back(elt);
          }
          tensors_by_value.emplace(node->output(), edsl::Tuple(elts));
        } else {
          throw std::runtime_error("prim::Constant has unsupported value type");
        }
      } break;
      default: {
        auto it_op = kSupportedOps.find(node->kind());
        if (it_op == kSupportedOps.end()) {
          throw std::runtime_error("Unsupported op");
        }
        std::vector<edsl::Tensor> args;
        for (const auto& input : node->inputs()) {
          auto it = tensors_by_value.find(input);
          if (it == tensors_by_value.end()) {
            throw std::runtime_error("Missing input.");
          }
          args.emplace_back(it->second);
        }
        tensors_by_value.emplace(node->output(), it_op->second(args));
      } break;
    }
  }

  IVLOG(2, "Outputs:")
  std::vector<edsl::Tensor> output_tensors;
  for (size_t i = 0; i < subgraph_->outputs().size(); i++) {
    const auto& output = subgraph_->outputs()[i];
    auto it = tensors_by_value.find(output);
    if (it == tensors_by_value.end()) {
      throw std::runtime_error("Missing output.");
    }
    IVLOG(2, "  " << i << ": " << it->second.str());
    output_tensors.emplace_back(it->second);
  }

  IVLOG(1, "Compiler::compile> done");
  return std::make_shared<Executable>(device_, input_tensors, output_tensors);
}

static size_t g_program_id = 1;

Executable::Executable(const vertexai::plaidml::device& device,  //
                       const std::vector<edsl::Tensor>& inputs,  //
                       const std::vector<edsl::Tensor>& outputs)
    : device_(device),  //
      input_bindings_(inputs.size()),
      output_ivalues_(outputs.size()),
      ctx_(std::make_shared<vertexai::ctx>()) {
  for (size_t i = 0; i < inputs.size(); i++) {
    input_bindings_[i] = vertexai::plaidml::binding{
        inputs[i],                                       // tensor
        device_.allocate(inputs[i].shape().byte_size())  // buffer
    };
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    const auto& shape = outputs[i].shape();
    output_bindings_.push_back(vertexai::plaidml::binding{
        outputs[i],                          // tensor
        device_.allocate(shape.byte_size())  // buffer
    });
    std::vector<int64_t> sizes;
    for (size_t j = 0; j < shape.rank(); j++) {
      sizes.push_back(shape.size_at(j));
    }
    output_ivalues_[i] = at::empty(at::IntArrayRef(sizes));
  }
  std::stringstream ss;
  ss << "pytorch_" << g_program_id++;
  name_ = ss.str();
  program_ = std::make_unique<edsl::Program>(name_, outputs);
  IVLOG(1, "Executable::Executable>");
  IVLOG(2, program_->str());
  exec_ = device_.compile(program_->ptr(), input_bindings_, output_bindings_);
  IVLOG(1, "Executable::Executable> done");
}

std::vector<torch::jit::IValue> Executable::run(at::ArrayRef<torch::jit::IValue>* inputs) {
  IVLOG(1, "Executable::run> " << name_);
  for (size_t i = 0; i < input_bindings_.size(); i++) {
    input_bindings_[i].buf.copy_from(ctx_, (*inputs)[i].toTensor().data_ptr());
  }
  exec_.run();
  for (size_t i = 0, j = 0; i < output_bindings_.size(); i++) {
    if (output_ivalues_[i].isTensor()) {
      output_bindings_[j++].buf.copy_into(ctx_, output_ivalues_[i].toTensor().data_ptr());
    }
  }
  IVLOG(1, "Executable::run> done");
  return output_ivalues_;
}
