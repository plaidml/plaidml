// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"

namespace plaidml {
namespace edsl {

// Given a forward pass tensor operation that takes inputs `wrt` and produces output `loss`,
// compute the gradients for each Tensor in `wrt`.
inline std::vector<Tensor> Gradient(const std::vector<Tensor>& wrt, const Tensor& loss) {
  std::vector<plaidml_expr*> wrt_exprs(wrt.size());
  std::vector<plaidml_expr*> deriv_exprs(wrt.size());
  for (size_t i = 0; i < wrt.size(); ++i) {
    wrt_exprs[i] = wrt[i].as_ptr();
  }
  ffi::call_void(             //
      plaidml_expr_gradient,  //
      wrt_exprs.size(),       //
      wrt_exprs.data(),       //
      loss.as_ptr(),          //
      deriv_exprs.data());
  std::vector<Tensor> ret(wrt.size());
  for (size_t i = 0; i < wrt.size(); ++i) {
    ret[i] = Tensor(deriv_exprs[i]);
  }
  return ret;
}

inline void RegisterTensorDeriv(const std::string& name, TensorDeriv fn) {
  auto thunk = [](void* user_ctx,          //
                  plaidml_expr* Y_expr,    //
                  plaidml_expr* dY_expr,   //
                  size_t nXs,              //
                  plaidml_expr** X_exprs,  //
                  plaidml_expr** dX_exprs) {
    auto fn = reinterpret_cast<TensorDeriv>(user_ctx);
    Tensor Y(Y_expr);
    Tensor dY(dY_expr);
    std::vector<Tensor> Xs(nXs);
    for (size_t i = 0; i < Xs.size(); i++) {
      Xs[i] = Tensor(X_exprs[i]);
    }
    auto dXs = fn(Y, dY, Xs);
    for (size_t i = 0; i < Xs.size(); i++) {
      dX_exprs[i] = ffi::call<plaidml_expr*>(plaidml_expr_clone, dXs[i].as_ptr());
    }
  };
  ffi::call_void(plaidml_deriv_register, name.c_str(), thunk, reinterpret_cast<void*>(fn));
}

}  // namespace edsl
}  // namespace plaidml
