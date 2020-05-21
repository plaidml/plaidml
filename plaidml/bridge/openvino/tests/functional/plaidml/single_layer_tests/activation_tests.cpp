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
          dst_data[oidx] = src_data[oidx] >= 0.0 ? src_data[oidx] : src_data[oidx] * prm.activationParams[0];
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
        <layer name="_ACTIVATION_TYPE_" id="1" type="_ACTIVATION_TYPE_" precision="FP32">
            _DATA_
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="1">
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
)";

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
      activation_info = R"(<data negative_slope="_N_SLOPE_"/>)";
      REPLACE_WITH_NUM(activation_info, "_N_SLOPE_", p.activationParams[0]);
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

    model = IRTemplateGenerator::getIRTemplate(p.activationType + "_Only", {1lu, p.in.c, p.in.h, p.in.w}, "FP32", model,
                                               edges_t);

    return model;
  }

 protected:
  virtual void SetUp() {
    activation_test_params p = ::testing::WithParamInterface<activation_test_params>::GetParam();
    std::string model = getModel(p);

    // TODO: can we use the newer API instead of using this legacy one?
    CNNNetReader net_reader;
    net_reader.ReadNetwork(model.data(), model.length());

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
    auto exeNetwork = ie.LoadNetwork(net_reader.getNetwork(), kDeviceName);
    InferRequest request = exeNetwork.CreateInferRequest();
    OutputsDataMap outInfo;
    outInfo = net_reader.getNetwork().getOutputsInfo();
    Blob::Ptr srcPtr = std::shared_ptr<Blob>(src);
    Blob::Ptr dstPtr = std::shared_ptr<Blob>(dst);
    request.SetBlob(net_reader.getNetwork().getInputsInfo().begin()->first, srcPtr);
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
    activation_test_params(case_1, "relu", {0.0f}),  // n_slope
    activation_test_params(case_2, "relu", {0.1f}), activation_test_params(case_3, "relu", {0.1f}),
    activation_test_params(case_4, "relu", {0.1f}), activation_test_params(case_5, "relu", {0.1f}),

    // activation_test_params(case_1, "sigmoid"),
    // activation_test_params(case_2, "sigmoid"),
    // activation_test_params(case_3, "sigmoid"),
    // activation_test_params(case_4, "sigmoid"),
    // activation_test_params(case_5, "sigmoid"),

    // activation_test_params(case_1, "power",
    //                        {1.0f, 0.0f, 1.0f}), // power offset scale
    // activation_test_params(case_2, "power", {2.0f, 3.0f, 5.0f}),
    // activation_test_params(case_3, "power", {4.0f, 1.0f, 7.0f}),

    // activation_test_params(case_1, "clamp", {0.0f, 1.0f}), // min max
    // activation_test_params(case_2, "clamp", {0.0f, 6.0f}),
    // activation_test_params(case_3, "clamp", {0.0f, 3.0f}),

    // FIXME doesn't support exp and not activation now
    // activation_test_params( case_1, "exp"),
    // activation_test_params( case_1, "not"),
};

INSTANTIATE_TEST_CASE_P(TestsActivationFunctions, ActivationTest, ::testing::ValuesIn(test_cases), getTestCaseName);
