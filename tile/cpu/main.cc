#include <iostream>

#include "base/config/config.h"
#include "base/util/file.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/lib.h"
#include "tile/stripe/stripe.h"
#include "tile/targets/cpu/jit.h"
#include "tile/targets/targets.h"

template <typename F>
void with_profile(F f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto d = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Execution took: " << std::chrono::duration<double>(d).count() * 1000 << "ms" << std::endl;
}

int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  el::Loggers::setVerboseLevel(2);

  using namespace vertexai::tile;  // NOLINT
  std::cout << "Hey!" << std::endl;

  // Express
  auto in1 = SimpleShape(DataType::FLOAT32, {1024, 1024});
  auto in2 = SimpleShape(DataType::FLOAT32, {1024, 1024});
  auto runinfo = lib::LoadMatMul("test", in1, in2);
  auto program = lang::GenerateStripe(runinfo);
  std::cout << *program->entry << std::endl;

  const auto& cfgs = targets::GetConfigs();
  const auto& cfg = cfgs.configs().at("cpu");
  const auto& stage = cfg.stages().at("default");
  codegen::CompilerState state(program);
  codegen::OptimizeOptions options;
  options.dump_passes = true;
  options.dbg_dir = "/tmp/stripe_cpu/passes";
  codegen::Optimize(&state, stage.passes(), options);

  std::cout << "============================================================\n" << *program->entry << std::endl;

  // Run
  std::vector<float> a_data(1024 * 1024);
  std::vector<float> b_data(1024 * 1024);
  std::vector<float> c_data(1024 * 1024);

  a_data[0] = 1.f;
  b_data[0] = 1.f;

  std::map<std::string, void*> io;
  io["A"] = a_data.data();
  io["B"] = b_data.data();
  io["C"] = c_data.data();

  targets::cpu::Native native;
  std::map<std::string, targets::cpu::External> externals;
  native.compile(*program->entry, externals);

  for (int i = 0; i < 10; i++) {
    for (auto& f : c_data) {
      f = 0.f;
    }
    with_profile([&]() {  //
      native.run(io);
    });
  }

  std::cout << c_data[0] << std::endl;

  return 0;
}
