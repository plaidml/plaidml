// Copyright 2019, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <string>

#include "pmlc/dialect/stripe/mlir.h"
#include "tile/stripe/stripe.h"

namespace pmlc {
namespace dialect {
namespace stripe {

namespace stripe = vertexai::tile::stripe;

mlir::OwningModuleRef IntoMLIR(MLIRContext* ctx, const stripe::Program& prog);
std::shared_ptr<stripe::Program> FromMLIR(mlir::ModuleOp module);

using SymbolValueMap = std::map<std::string, mlir::Value*>;

mlir::Value* AffineIntoMLIR(     //
    mlir::OpBuilder* builder,    //
    const SymbolValueMap& idxs,  //
    const stripe::Affine& affine);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
