// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "plaidml/edsl/edsl.h"

#include <ie_blob.h>
#include <ie_layers_property.hpp>

#include "plaidml_state.hpp"
#include "plaidml_util.hpp"

using namespace InferenceEngine;
using namespace PlaidMLPlugin;

using namespace InferenceEngine;

struct Context {
  std::vector<util::Any> inputs_;
  std::vector<util::Any> outputs_;
};

class Op {
 public:
  static std::shared_ptr<Op> Create(const CNNLayerPtr& layer);

  virtual void Apply(State* ctx);
  virtual ~Op() = default;

 protected:
  void Init(const CNNLayerPtr& layer);

  virtual void LoadWeights(State* state);
  virtual void PackInputs(State* state);
  virtual void PackOutputs(State* state);
  virtual void PackWeights(State* state);
  virtual void Execute();

 protected:
  Context ctx_;
  CNNLayerPtr layer_;
};

template <class T>
struct get_in;
template <class T>
struct get_in {
  static const T& get(const Context& ctx, size_t idx) { return ctx.inputs_.at(idx).get<T>(); }
};

template <class T>
struct get_out;
template <class T>
struct get_out {
  static T& get(Context& ctx, size_t idx) { return *(ctx.outputs_.at(idx).get<T*>()); }
};

template <typename, typename>
struct CallHelper;

template <typename... Ins, typename... Outs>
struct CallHelper<std::tuple<Ins...>, std::tuple<Outs...>> : public Op {
  template <typename Impl, int... I, int... O>
  static void call_impl(Context& ctx, Impl* impl, util::sequence<I...>, util::sequence<O...>) {
    impl->run(get_in<Ins>::get(ctx, I)..., get_out<Outs>::get(ctx, O)...);
  }

  template <typename Impl>
  static void call(Context& ctx, Impl* impl) {
    call_impl(ctx, impl, util::make_sequence<sizeof...(Ins)>{}, util::make_sequence<sizeof...(Outs)>{});
  }
};

template <typename Res, typename... Args>
struct parse_function_api;

template <typename Res, typename... Args>
struct parse_function_api<Res(Args...)> {
  using Outs = std::tuple<Res>;
  using Ins = std::tuple<Args...>;
};

template <typename Impl>
struct TypedOp : public Op {
  void Execute() override { CallHelper<typename Impl::Ins, typename Impl::Outs>::call(ctx_, static_cast<Impl*>(this)); }
};

// FIXME: Now Id isn't used, but I plan to use it for register layer
#define PLAIDML_LAYER(Name, API, Id)                        \
  struct Name : public parse_function_api API, /* NOLINT */ \
                public TypedOp<Name>
