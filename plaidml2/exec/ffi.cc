// Copyright 2019 Intel Corporation.

#include "plaidml2/exec/ffi.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>

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
using vertexai::tile::ConstBufferManager;
using vertexai::tile::Program;
using vertexai::tile::View;
using vertexai::tile::targets::GetConfigs;

#ifdef PLAIDML_AST
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

size_t plaidml_device_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return GetPlatform()->ListDevices().size();
  });
}

void plaidml_device_list(  //
    plaidml_error* err,    //
    size_t ndevices,       //
    plaidml_string** device_ids) {
  ffi_wrap_void(err, [&] {
    auto devices = GetPlatform()->ListDevices();
    for (size_t i = 0; i < std::min(ndevices, devices.size()); i++) {
      device_ids[i] = new plaidml_string{devices[i]};
    }
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
    for (size_t i = 0; i < ninputs; i++) {
      auto param_expr = std::dynamic_pointer_cast<ParamExpr>(inputs[i]->expr->expr);
      if (!param_expr) {
        throw std::runtime_error("Buffers can only be bound to ParamExprs");
      }
      param_expr->buffer = inputs[i]->buffer->buffer;
    }
    for (const auto& input : program->eval.inputs) {
      auto it = program->eval.names_by_expr.find(input);
      if (it == program->eval.names_by_expr.end()) {
        throw std::runtime_error("Invalid program, input with unknown name");
      }
      exec->input_bufs[it->second] = input->buffer;
    }
    for (size_t i = 0; i < noutputs; i++) {
      if (!outputs[i] || !outputs[i]->expr || !outputs[i]->buffer) {
        throw std::runtime_error("Undefined output bindings");
      }
      auto expr = outputs[i]->expr->expr;
      auto it = program->eval.names_by_expr.find(expr.get());
      if (it == program->eval.names_by_expr.end()) {
        throw std::runtime_error("Invalid program, output with unknown name");
      }
      exec->output_bufs[it->second] = outputs[i]->buffer->buffer;
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

    // add user supplied input buffers to the ioMap
    for (size_t i = 0; i < ninputs; i++) {
      IVLOG(1, "Input[" << i << "]");
      auto stagingValue = inputs[i]->expr->value;
      program->program->ioMap.emplace(stagingValue, inputs[i]->buffer->buffer);
    }

    // bind all input buffers
    auto stripeFuncOp = llvm::dyn_cast<mlir::FuncOp>(module->getBody()->front());
    if (!stripeFuncOp) {
      throw std::runtime_error("Missing stripe FuncOp");
    }
    for (const auto& [stagingValue, buffer] : program->program->ioMap) {
      auto programValue = program->program->mapper.lookupOrNull(stagingValue);
      if (!programValue) {
        // This must be an unused input, ignore it.
        continue;
      }
      auto arg = llvm::dyn_cast<mlir::BlockArgument>(programValue);
      if (!arg) {
        throw std::runtime_error("Expected input to be a block argument");
      }
      auto argNumber = arg->getArgNumber();
      auto attrName = Dialect::getDialectAttrName("name");
      auto attr = stripeFuncOp.getArgAttrOfType<mlir::StringAttr>(argNumber, attrName);
      if (!attr) {
        throw std::runtime_error("Missing expected argument attribute");
      }
      auto inputName = attr.getValue().str();
      IVLOG(1, "  name: " << inputName);
      exec->input_bufs[inputName] = buffer;
    }

    // bind output buffers
    auto tileFuncOp = llvm::dyn_cast<mlir::FuncOp>(program->program->module->getBody()->front());
    if (!tileFuncOp) {
      throw std::runtime_error("Missing tile FuncOp");
    }
    auto numInputs = tileFuncOp.getNumArguments();
    for (size_t i = 0; i < noutputs; i++) {
      IVLOG(1, "Output[" << i << "]: ");
      auto argNumber = numInputs + i;
      auto attrName = Dialect::getDialectAttrName("name");
      auto attr = stripeFuncOp.getArgAttrOfType<mlir::StringAttr>(argNumber, attrName);
      if (!attr) {
        throw std::runtime_error("Missing expected argument attribute");
      }
      auto outputName = attr.getValue().str();
      IVLOG(1, "  name: " << outputName);
      exec->output_bufs[outputName] = outputs[i]->buffer->buffer;
    }

    // TODO(MLIR): updates
    return exec.release();
#endif
  });
}

size_t plaidml_target_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return GetConfigs().configs().size();
  });
}

void plaidml_target_list(  //
    plaidml_error* err,    //
    size_t ntargets,       //
    plaidml_string** targets) {
  ffi_wrap_void(err, [&] {
    auto configs = GetConfigs().configs();
    size_t i = 0;
    for (const auto& [key, value] : configs) {
      if (i >= ntargets) {
        break;
      }
      targets[i++] = new plaidml_string{key};
    }
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
