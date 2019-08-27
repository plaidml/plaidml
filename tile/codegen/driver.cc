// Copyright 2018, Intel Corporation

#include "tile/codegen/driver.h"

#include <memory>
#include <unordered_map>

#include <boost/format.hpp>

#include "base/config/config.h"
#include "base/util/any_factory_map.h"
#include "base/util/throw.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/emitc.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

void DumpProgram(const Block& program,            //
                 const OptimizeOptions& options,  //
                 const std::string& name,         //
                 size_t counter) {
  if (options.dump_passes || options.dump_passes_proto || options.dump_code) {
    boost::filesystem::create_directories(options.dbg_dir);
    if (options.dump_passes) {
      auto filename = str(boost::format("%02zu_%s.txt") % counter % name);
      auto path = (options.dbg_dir / filename).string();
      std::ofstream fout(path);
      fout << program << std::endl;
    }
    if (options.dump_passes_proto) {
      auto filename = str(boost::format("%02zu_%s.pb") % counter % name);
      auto path = (options.dbg_dir / filename).string();
      std::ofstream fout(path, std::ofstream::binary);
      // Save without Buffers
      Program true_program;
      true_program.entry = std::make_shared<Block>(program);
      auto proto = IntoProto(true_program);
      proto.SerializeToOstream(&fout);
    }
    if (options.dump_code) {
      auto filename = str(boost::format("%02zu_%s.c") % counter % name);
      auto path = (options.dbg_dir / filename).string();
      std::ofstream fout(path);
      fout << EmitC(program);
    }
  }
}

void ValidateBlock(Block* root) {
  RunOnBlocks(  //
      root, {},
      [&](auto map, auto block) {
        for (const auto& ref : block->refs) {
          if (ref.dir == RefDir::None && !ref.from.empty()) {
            throw_with_trace(std::runtime_error(
                str(boost::format("ref.dir == RefDir::None && !ref.from.empty(). ref: %1% in block: %2%") % ref.into() %
                    block->name)));
          }
          if (ref.from.empty() && ref.dir != RefDir::None) {
            throw_with_trace(std::runtime_error(
                str(boost::format("ref.from.empty() && ref.dir != RefDir::None. ref: %1% in block: %2%") % ref.into() %
                    block->name)));
          }
        }
      },
      true);
}

class ConfigsRegistry {
 public:
  static ConfigsRegistry* Instance() {
    static ConfigsRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, const std::string& cfg_bytes) {  //
    registry_[name] = cfg_bytes;
  }

  proto::Config Resolve(const std::string& name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      throw_with_trace(std::runtime_error(str(boost::format("Could not find config: %s") % name)));
    }
    return ParseConfig<proto::Config>(it->second);
  }

 private:
  std::unordered_map<std::string, std::string> registry_;
};

void ConvertToStripe(CompilerState* state) {
  IVLOG(1, "Converting to Stripe");
  mlir::FuncOp op = mlir::cast<mlir::FuncOp>(state->module.front());
  *state->prog = pmlc::dialect::stripe::ToStripe(op);
  // TODO: Erase
}

void ConvertToStripeMLIR(CompilerState* state) {
  IVLOG(1, "Converting to Stripe MLIR");
  state->module.push_back(pmlc::dialect::stripe::ToStripeMLIR(&state->ctx, *state->prog));
}

}  // namespace

void Optimize(CompilerState* state, const Passes& passes, const OptimizeOptions& options) {
  size_t counter = 0;
  DumpProgram(*state->entry(), options, "initial", counter++);
  bool in_stripe = true;
  for (const auto& pass : passes) {
    IVLOG(1, "Optimization Pass " << pass.name());
    std::unique_ptr<CompilePass> compile_pass =
        AnyFactoryMap<CompilePass>::Instance()->MakeInstanceIfSupported(context::Context{}, pass.pass());
    if (!compile_pass) {
      throw_with_trace(std::runtime_error(
          str(boost::format("Unsupported pass: %1% -> %2%") % pass.name() % pass.pass().type_url())));
    }
    bool wants_stripe = compile_pass->is_stripe();
    if (!in_stripe && wants_stripe) {
      ConvertToStripe(state);
    } else if (in_stripe && !wants_stripe) {
      ConvertToStripeMLIR(state);
    }
    in_stripe = wants_stripe;
    compile_pass->Apply(state);
    if (in_stripe) {
      DumpProgram(*state->entry(), options, pass.name(), counter);
    } else {
      // DUMP MLIR
    }
    counter++;
    ValidateBlock(state->entry());
  }
  if (!in_stripe) {
    ConvertToStripe(state);
  }
  // Remove constants that are no longer used
  if (state->const_bufs == nullptr) {
    return;
  }
  auto& cbufs = state->const_bufs->buffers;
  for (auto it = cbufs.begin(); it != cbufs.end();) {
    if (state->entry()->ref_by_into(it->first, false) == state->entry()->refs.end()) {
      it = cbufs.erase(it);
    } else {
      ++it;
    }
  }
  IVLOG(3, "All optimization passes complete");
}

void Configs::Register(const std::string& name, const std::string& pb_bytes) {
  ConfigsRegistry::Instance()->Register(name, pb_bytes);
}

proto::Config Configs::Resolve(const std::string& name) {  //
  return ConfigsRegistry::Instance()->Resolve(name);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
