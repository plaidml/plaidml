// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ie_core.hpp"
#include "ie_plugin_ptr.hpp"
#include "ir_gen_helper.hpp"

#include "cpp/ie_cnn_net_reader.h"
#include "single_layer_common.hpp"
#include "tests_common.hpp"

#include "pmlc/util/logging.h"

using namespace ::testing;           // NOLINT[build/namespaces]
using namespace InferenceEngine;     // NOLINT[build/namespaces]
using namespace single_layer_tests;  // NOLINT[build/namespaces]

const char* kDeviceName = "PlaidML";

struct activation_base_params {
  struct {
    size_t c;
    size_t h;
    size_t w;
  } in;
};

struct activation_test_params : activation_base_params {
  std::string activationType;
  std::vector<float> activationParams;

  activation_test_params(activation_base_params params, std::string activationFunctionType,
                         std::vector<float> activationFunctionParams = {})
      : activation_base_params(std::move(params)),
        activationType(std::move(activationFunctionType)),
        activationParams(std::move(activationFunctionParams)) {}
};

template <typename data_t>
void ref_activation(const TBlob<data_t>& src, TBlob<data_t>& dst, activation_test_params prm) {
  size_t IW = prm.in.w;
  size_t IH = prm.in.h;
  size_t IC = prm.in.c;

  const data_t* src_data = src.readOnly();
  data_t* dst_data = dst.data();

  for (uint32_t c = 0; c < IC; c++) {
    for (uint32_t h = 0; h < IH; h++) {
      for (uint32_t w = 0; w < IW; w++) {
        uint32_t oidx = c * IH * IW + h * IW + w;

        if (prm.activationType == "exp") {
          dst_data[oidx] = exp(src_data[oidx]);
        } else if (prm.activationType == "not") {
          dst_data[oidx] = !(src_data[oidx]);
        } else if (prm.activationType == "sigmoid") {
          dst_data[oidx] = 1 / (1 + exp(-src_data[oidx]));
        } else if (prm.activationType == "relu") {
          dst_data[oidx] = src_data[oidx] >= 0.0 ? src_data[oidx] : 0.0;
        } else if (prm.activationType == "power") {
          dst_data[oidx] =
              std::pow(prm.activationParams[2] * src_data[oidx] + prm.activationParams[1], prm.activationParams[0]);
        } else if (prm.activationType == "clamp") {
          dst_data[oidx] = src_data[oidx] >= prm.activationParams[1]
                               ? prm.activationParams[1]
                               : src_data[oidx] <= prm.activationParams[0] ? prm.activationParams[0] : src_data[oidx];
        }
      }
    }
  }
}

class ActivationTest : public TestsCommon, public WithParamInterface<activation_test_params> {
  std::string layers_t = R"(
        <layer id="1" name="_ACTIVATION_TYPE_" type="_ACTIVATION_TYPE_" version="opset1">
            _DATA_
            <input>
                <port id="0" precision="FP32">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
)";

  std::string edges_t = R"(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
)";

  // TODO: This is a hack replacement for getIRTemplate which isn't working quite right...
  std::string getCustomIRTemplate(const std::string& name,
      const std::vector<size_t>& input_shape,
      const std::string& precision,
      const std::string& layers,
      const std::string& edges,
      const unsigned ir_version,
      const std::string& metadata = "") {
    const std::vector< std::vector<size_t>> input_shape_vector = { input_shape };
    return getCustomIRTemplate(name, input_shape_vector, precision, layers, edges, ir_version, metadata);
  }

  // TODO: This is a hack replacement for getIRTemplate which isn't working quite right...
  std::string getCustomIRTemplate(const std::string& name,
      const std::vector<std::vector<size_t>>& input_shape,
      const std::string& precision,
      const std::string& layers,
      const std::string& edges,
      const unsigned ir_version,
      const std::string& metadata = "") {
    // TODO: Be consistent in whether I'm loading precision or just using fp32

    std::string model_input_t = R"V0G0N(
                <layer id="_ID_" name="_input_name_" type="Parameter" version="opset1">
                    <data element_type="f32" shape="__SRC_DIMS_COMMAS__"/>
                    <output>
                        <port id="0" precision="_PR_">__SRC_DIMS__
                        </port>
                    </output>
                </layer>
        )V0G0N";

    std::string model_output_t = R"V0G0N(
                <layer name="_output_name_" type="Result" id="_ID_" version="opset1">
                    <input>
                        <port id="0" precision="_PR_">__SRC_DIMS__
                        </port>
                    </input>
                </layer>
        )V0G0N";

    std::string model_t = R"V0G0N(
        <net name="_NAME_" version="_IRv_">
            <layers>
                __INPUT_LAYERS_
                _LAYERS_
                __OUTPUT_LAYER__
            </layers>
            <edges>
                _EDGES_
            </edges>
            <meta_data>
                _META_DATA_
            </meta_data>
        </net>
        )V0G0N";

    std::string model = model_t;
    REPLACE_WITH_STR(model, "_NAME_", name);
    REPLACE_WITH_NUM(model, "_IRv_", ir_version);
    std::string input_layers;
    std::string output_layer;
    for (size_t input_idx = 0; input_idx < input_shape.size(); input_idx++) {
        std::string model_input = model_input_t;
        std::string s_dims;
        std::string s_dims_commas;
        bool first_iter = true;
        for (auto& dim : input_shape[0]) {
            if (first_iter) {
              first_iter = false;
            } else {
              s_dims_commas += ",";
            }
            s_dims_commas += std::to_string(dim);
            s_dims += "\n\t                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model_input, "_ID_", std::to_string(input_idx));
        std::string input_name = "in" + std::to_string(input_idx + 1);
        REPLACE_WITH_STR(model_input, "_input_name_", input_name);
        REPLACE_WITH_STR(model_input, "__SRC_DIMS__", s_dims);
        REPLACE_WITH_STR(model_input, "__SRC_DIMS_COMMAS__", s_dims_commas);
        input_layers += model_input;
        if (input_idx == 0) {
          // The output follows the first input's shape, so build at same time
          std::string model_output = model_output_t;
          REPLACE_WITH_STR(model_output, "_ID_", "2");
          REPLACE_WITH_STR(model_output, "_output_name_", "out");
          REPLACE_WITH_STR(model_input, "__SRC_DIMS__", s_dims);
          output_layer += model_output;
        }
    }
    REPLACE_WITH_STR(model, "__INPUT_LAYERS_", input_layers);
    REPLACE_WITH_STR(model, "__OUTPUT_LAYER__", output_layer);
    REPLACE_WITH_STR(model, "_PR_", precision);
    REPLACE_WITH_STR(model, "_LAYERS_", layers);
    REPLACE_WITH_STR(model, "_EDGES_", edges);
    REPLACE_WITH_STR(model, "_META_DATA_", metadata);

    return model;
  }

  std::string getModel(activation_test_params p) {
    std::string model = layers_t;
    std::string activation_info;
    if (p.activationType == "exp") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Exp");
    } else if (p.activationType == "not") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Not");
    } else if (p.activationType == "sigmoid") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Sigmoid");
    } else if (p.activationType == "relu") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "ReLU");
    } else if (p.activationType == "power") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Power");
      activation_info = R"(<data power="_POWER_" scale="_SCALE_" shift="_OFFSET_"/>)";
      REPLACE_WITH_NUM(activation_info, "_POWER_", p.activationParams[0]);
      REPLACE_WITH_NUM(activation_info, "_OFFSET_", p.activationParams[1]);
      REPLACE_WITH_NUM(activation_info, "_SCALE_", p.activationParams[2]);
    } else if (p.activationType == "clamp") {
      REPLACE_WITH_STR(model, "_ACTIVATION_TYPE_", "Clamp");
      activation_info = R"(<data max="_MAX_" min="_MIN_"/>)";
      REPLACE_WITH_NUM(activation_info, "_MIN_", p.activationParams[0]);
      REPLACE_WITH_NUM(activation_info, "_MAX_", p.activationParams[1]);
    }

    REPLACE_WITH_STR(model, "_DATA_", activation_info);
    REPLACE_WITH_NUM(model, "_IN_", 1);
    REPLACE_WITH_NUM(model, "_IW_", p.in.w);
    REPLACE_WITH_NUM(model, "_IH_", p.in.h);
    REPLACE_WITH_NUM(model, "_IC_", p.in.c);

    model = getCustomIRTemplate(p.activationType + "_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model,
                                               edges_t, 10);

    IVLOG(1, "Requesting testing of IR: " << model);
    return model;
  }

 protected:
  virtual void SetUp() {
    activation_test_params p = ::testing::WithParamInterface<activation_test_params>::GetParam();
    std::string model = getModel(p);

    // TODO: can we use the newer API instead of using this legacy one?

    SizeVector dims_src = {p.in.w, p.in.h, p.in.c};
    TBlob<float>::Ptr src = make_shared_blob<float>(TensorDesc(Precision::FP32, dims_src, CHW));
    src->allocate();
    fill_data(src->data(), src->size());

    SizeVector dims_dst = dims_src;
    TBlob<float>::Ptr dst = make_shared_blob<float>(TensorDesc(Precision::FP32, dims_dst, CHW));
    dst->allocate();

    TBlob<float>::Ptr dst_ref = make_shared_blob<float>(TensorDesc(Precision::FP32, dims_dst, CHW));
    dst_ref->allocate();

    // NOTE; using the new API here as I couldn't get the older one to work.
    Core ie("plaidml/bridge/openvino/plugins.xml");
    auto devices = ie.GetAvailableDevices();
    IVLOG(1, "devices: " << devices);
    auto network = ie.ReadNetwork(model, Blob::Ptr());  // TODO: Weights??
    auto exeNetwork = ie.LoadNetwork(network, kDeviceName);
    InferRequest request = exeNetwork.CreateInferRequest();
    OutputsDataMap outInfo;
    outInfo = network.getOutputsInfo();
    IVLOG(1, "Just added outInfo as " << outInfo);
    Blob::Ptr srcPtr = std::shared_ptr<Blob>(src);
    Blob::Ptr dstPtr = std::shared_ptr<Blob>(dst);
    request.SetBlob(network.getInputsInfo().begin()->first, srcPtr);
    request.SetBlob(outInfo.begin()->first, dstPtr);

    request.Infer();

    ref_activation(*src, *dst_ref, p);

    compare(*dst, *dst_ref);
  }
};

#define case_1 activation_base_params({{3, 228, 228}})
#define case_2 activation_base_params({{2, 228, 228}})
#define case_3 activation_base_params({{1, 112, 112}})
#define case_4 activation_base_params({{192, 56, 56}})
#define case_5 activation_base_params({{1, 228, 228}})

TEST_P(ActivationTest, TestsActivationFunctions) {}

std::string getTestCaseName(testing::TestParamInfo<activation_test_params> obj) {
  return "w" + std::to_string(obj.param.in.w) + "_h" + std::to_string(obj.param.in.h) + "_c" +
         std::to_string(obj.param.in.c) + "_" + obj.param.activationType;
}

activation_test_params test_cases[] = {
    activation_test_params(case_1, "relu"),

    // There used to be non-zero negative slope tests (i.e. for LeakyReLU), but with an nGraph-based opset those show up
    // as a fused PReLU; re-enable once we have PReLU support
    // activation_test_params(case_2, "relu", {0.1f}),  // n_slope
    // activation_test_params(case_3, "relu", {0.1f}),  //
    // activation_test_params(case_4, "relu", {0.1f}),  //
    // activation_test_params(case_5, "relu", {0.1f}),

    // activation_test_params(case_1, "sigmoid"),  //
    // activation_test_params(case_2, "sigmoid"),  //
    // activation_test_params(case_3, "sigmoid"),  //
    // activation_test_params(case_4, "sigmoid"),  //
    // activation_test_params(case_5, "sigmoid"),  //

    // activation_test_params(case_1, "power", {1.0f, 0.0f, 1.0f}),  // power offset scale
    // activation_test_params(case_2, "power", {2.0f, 3.0f, 5.0f}),
    // activation_test_params(case_3, "power", {4.0f, 1.0f, 7.0f}),

    activation_test_params(case_1, "clamp", {0.0f, 1.0f}),  // min max
    activation_test_params(case_2, "clamp", {0.0f, 6.0f}),  //
    activation_test_params(case_3, "clamp", {0.0f, 3.0f}),  //

    // FIXME doesn't support exp and not activation now
    // activation_test_params(case_1, "exp"),
    // activation_test_params(case_1, "not"),
};

INSTANTIATE_TEST_CASE_P(TestsActivationFunctions, ActivationTest, ::testing::ValuesIn(test_cases), getTestCaseName);
