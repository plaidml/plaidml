// Copyright 2019 Intel Corporation.

#include "plaidml2/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml2/core/internal.h"
#include "tile/targets/targets.h"

#ifdef PLAIDML_AST
#include "tile/lang/gen_stripe.h"
#endif
#ifdef PLAIDML_MLIR
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/tile/lowering.h"
#include "pmlc/dialect/tile/program.h"
#endif

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GetPlatform;
using plaidml::core::GlobalContext;
using vertexai::context::Context;
using vertexai::tile::Allocator;
using vertexai::tile::Buffer;
using vertexai::tile::BufferPtr;
using vertexai::tile::ConstBufferManager;
using vertexai::tile::Program;
using vertexai::tile::View;
using vertexai::tile::targets::GetConfigs;

#ifdef PLAIDML_AST
using vertexai::tile::lang::ast::ExprPtr;
using vertexai::tile::lang::ast::ParamExpr;
#endif
#ifdef PLAIDML_MLIR
using pmlc::dialect::stripe::Dialect;
using pmlc::dialect::stripe::FromMLIR;
using pmlc::dialect::tile::LowerIntoStripe;
#endif

namespace {

class PlatformAllocator : public Allocator {
 public:
  explicit PlatformAllocator(const std::string& device_id) : device_id_(device_id) {}

  std::shared_ptr<Buffer> allocate(size_t size) {
    Context ctx;
    return GetPlatform()->MakeBuffer(ctx, device_id_, size);
  }

 private:
  std::string device_id_;
};

}  // namespace

extern "C" {

struct plaidml_executable {
  using BufferMap = std::map<std::string, std::shared_ptr<Buffer>>;
  BufferMap input_bufs;
  BufferMap output_bufs;
  std::shared_ptr<Program> program;
};

void plaidml_exec_init(  //
    plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      IVLOG(1, "plaidml_exec_init");
      GetPlatform();
    });
  });
}

plaidml_strings* plaidml_devices_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    auto devices = GetPlatform()->ListDevices();
    auto strs = new plaidml_string*[devices.size()];
    for (size_t i = 0; i < devices.size(); i++) {
      strs[i] = new plaidml_string{devices[i]};
    }
    return new plaidml_strings{devices.size(), strs};
  });
}

plaidml_strings* plaidml_targets_get(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_strings*>(err, nullptr, [&] {
    auto configs = GetConfigs().configs();
    auto strs = new plaidml_string*[configs.size()];
    size_t i = 0;
    for (const auto& [key, value] : configs) {
      strs[i++] = new plaidml_string{key};
    }
    return new plaidml_strings{configs.size(), strs};
  });
}

plaidml_executable* plaidml_compile(  //
    plaidml_error* err,               //
    plaidml_program* program,         //
    const char* device,               //
    const char* target,               //
    size_t ninputs,                   //
    plaidml_binding** inputs,         //
    size_t noutputs,                  //
    plaidml_binding** outputs) {
  return ffi_wrap<plaidml_executable*>(err, nullptr, [&] {
    IVLOG(1, "Compiling with device: " << device << ", target: " << target);
    auto configs = GetConfigs().configs();
    if (!configs.count(target)) {
      throw std::runtime_error(llvm::formatv("Unknown target specified: {0}", target).str());
    }
#ifdef PLAIDML_AST
    ConstBufferManager const_bufs;
    const_bufs.allocator = std::make_shared<PlatformAllocator>(device);
    std::unique_ptr<plaidml_executable> exec{new plaidml_executable};
    auto stripe = vertexai::tile::lang::GenerateStripe(program->eval.runinfo);
    Context ctx;
    exec->program = GetPlatform()->MakeProgram(ctx, device, target, stripe, &const_bufs);
    std::unordered_map<ExprPtr, BufferPtr> input_bindings;
    for (size_t i = 0; i < ninputs; i++) {
      auto param_expr = std::dynamic_pointer_cast<ParamExpr>(inputs[i]->expr->expr);
      if (!param_expr) {
        throw std::runtime_error("Buffers can only be bound to ParamExprs");
      }
      param_expr->buffer = inputs[i]->buffer->buffer;
      input_bindings[inputs[i]->expr->expr] = inputs[i]->buffer->buffer;
    }
    std::unordered_map<ExprPtr, BufferPtr> output_bindings;
    for (size_t i = 0; i < noutputs; i++) {
      output_bindings[outputs[i]->expr->expr] = outputs[i]->buffer->buffer;
    }
    for (const auto& arg : program->eval.args) {
      if (arg.is_input) {
        auto it = input_bindings.find(arg.expr);
        auto param_expr = std::dynamic_pointer_cast<ParamExpr>(arg.expr);
        auto buffer = (it == input_bindings.end()) ? param_expr->buffer : it->second;
        exec->input_bufs[arg.name] = buffer;
      } else {
        auto it = output_bindings.find(arg.expr);
        if (it == output_bindings.end()) {
          throw std::runtime_error("Invalid program, unbound output");
        }
        exec->output_bufs[arg.name] = it->second;
      }
    }
    for (const auto& kvp : program->eval.updates) {
      exec->output_bufs[kvp.first] = kvp.second->buffer;
    }
    return exec.release();
#endif
#ifdef PLAIDML_MLIR
    ConstBufferManager const_bufs;
    const_bufs.allocator = std::make_shared<PlatformAllocator>(device);
    std::unique_ptr<plaidml_executable> exec{new plaidml_executable};

    // 1. lower tile dialect -> stripe dialect
    auto module = LowerIntoStripe(*program->program->module);
    // 2. convert MLIR -> stripe
    auto stripe = FromMLIR(*module);
    Context ctx;
    exec->program = GetPlatform()->MakeProgram(ctx, device, target, stripe, &const_bufs);
    IVLOG(1, "After make program");

    std::unordered_map<mlir::Value*, BufferPtr> input_bindings;
    for (unsigned i = 0; i < ninputs; i++) {
      input_bindings[inputs[i]->expr->value] = inputs[i]->buffer->buffer;
    }

    std::unordered_map<mlir::Value*, BufferPtr> output_bindings;
    for (unsigned i = 0; i < noutputs; i++) {
      output_bindings[outputs[i]->expr->value] = outputs[i]->buffer->buffer;
    }

    auto attrName = Dialect::getDialectAttrName("name");
    auto stripeFuncOp = llvm::cast<mlir::FuncOp>(module->getBody()->front());
    for (unsigned i = 0; i < program->program->arguments.size(); i++) {
      const auto& arg = program->program->arguments[i];
      auto attr = stripeFuncOp.getArgAttrOfType<mlir::StringAttr>(i, attrName);
      if (!attr) {
        throw std::runtime_error("Missing expected argument attribute");
      }
      auto name = attr.getValue().str();
      if (arg.isInput) {
        auto it = input_bindings.find(arg.value);
        auto is_implicit = (it == input_bindings.end());
        auto buffer = is_implicit ? arg.buffer : it->second;
        IVLOG(1, " Input[" << i << "]: " << name << ": " << buffer << ", implicit: " << is_implicit);
        if (!buffer) {
          throw std::runtime_error("Unbound input");
        }
        exec->input_bufs[name] = buffer;
      } else {
        auto it = output_bindings.find(arg.value);
        auto is_implicit = (it == output_bindings.end());
        auto buffer = is_implicit ? arg.buffer : it->second;
        IVLOG(1, "Output[" << i << "]: " << name << ": " << buffer << ", implicit: " << is_implicit);
        if (!buffer) {
          throw std::runtime_error("Unbound output");
        }
        exec->output_bufs[name] = buffer;
      }
    }

    return exec.release();
#endif
  });
}

void plaidml_executable_free(  //
    plaidml_error* err,        //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {  //
    delete exec;
  });
}

void plaidml_executable_run(  //
    plaidml_error* err,       //
    plaidml_executable* exec) {
  ffi_wrap_void(err, [&] {
    auto ctx = GlobalContext::getContext();
    exec->program->Run(*ctx, exec->input_bufs, exec->output_bufs).get();
  });
}

}  // extern "C"
