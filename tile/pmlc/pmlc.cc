// Copyright 2019, Intel Corporation

#include "tile/pmlc/pmlc.h"

#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "base/config/config.h"
#include "base/util/file.h"
#include "base/util/throw.h"
#include "tile/codegen/driver.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lib/tests.h"
#ifdef ENABLE_LLVM_BITCODE
#include "tile/targets/cpu/jit.h"
#endif
#include "tile/util/tile_file.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace vertexai {
namespace tile {
namespace pmlc {

using namespace codegen;  // NOLINT
using namespace stripe;   // NOLINT

App::App() : opts{"Allowed options"} {
  opts.add_options()                      //
      ("help,h", "produce help message")  //
      ("verbose,v", po::value<int>()->default_value(0), "increase verbosity");
}

App* App::Instance() {
  static App app;
  return &app;
}

bool App::parse(int argc, char* argv[]) {
  auto parser = po::command_line_parser(argc, argv).options(opts).positional(pos_opts);
  po::store(parser.run(), args);
  if (args.count("help")) {
    std::cout << opts << std::endl;
    return false;
  }
  if (args.count("verbose")) {
    el::Loggers::setVerboseLevel(args["verbose"].as<int>());
  }
  args.notify();
  return true;
}

[[gnu::unused]] auto init = []() {
  auto app = App::Instance();
  app->pos_opts.add("input", 1);
  app->opts.add_options()                                                                 //
      ("input", po::value<fs::path>()->required(), "input file path")                     //
      ("config,c", po::value<fs::path>()->required(), "config file path")                 //
      ("target,t", po::value<std::string>()->required(), "name of target within config")  //
      ("stage,s", po::value<std::string>()->required(), "name of stage within config")    //
      ("outdir,D", po::value<fs::path>()->default_value("."), "output directory")         //
      ("int8", "treat all datatypes as int8")                                             //
      ("internal", "input specifies an internally defined network")                       //
      ("dump-passes", "dump passes in *.txt format")                                      //
      ("dump-passes-proto", "dump passes in *.pb format")                                 //
#ifdef ENABLE_LLVM_BITCODE
      ("llvm", "enable LLVM bitcode output")  //
#endif
      ;  // NOLINT
  return 0;
}();

lang::RunInfo LoadTile(const fs::path& filename, bool is_internal) {
  if (is_internal) {
    auto test = lib::CreateTest(filename.string());
    if (!test) {
      throw std::runtime_error(str(boost::format("Internal test not found: %1%") % filename));
    }
    return *test;
  }
  return util::TileFile(filename).Load();
}

std::shared_ptr<Program> DefaultStage(const App& app,                      //
                                      const fs::path& input_path,          //
                                      const fs::path& out_dir,             //
                                      const codegen::proto::Stage& stage,  //
                                      const codegen::OptimizeOptions& options) {
  bool is_internal = app.args.count("internal");
  bool enable_int8_mode = app.args.count("int8");
  auto runinfo = LoadTile(input_path, is_internal);
  auto program = GenerateStripe(runinfo, enable_int8_mode);
  for (const auto& kvp : runinfo.input_buffers) {
    auto buf = std::dynamic_pointer_cast<util::SimpleBuffer>(kvp.second);
    if (buf) {
      std::string str(reinterpret_cast<const char*>(buf->bytes.data()), buf->bytes.size());
      program->buffers[kvp.first].sections.emplace("data", str);
    }
  }
  for (const auto& kvp : runinfo.qparams_buffers) {
    auto buf = std::dynamic_pointer_cast<util::SimpleBuffer>(kvp.second);
    if (buf) {
      std::string str(reinterpret_cast<const char*>(buf->bytes.data()), buf->bytes.size());
      program->buffers[kvp.first].sections.emplace("qparams", str);
    }
  }
  CompilerState state(program);
  Optimize(&state, stage.passes(), options);
  WriteFile(out_dir / "stripe.txt", false, [&program](std::ofstream& fout) {  //
    fout << *program->entry << std::endl;
  });
  WriteFile(out_dir / "stripe.pb", true, [&program](std::ofstream& fout) {  //
    auto proto = IntoProto(*program);
    proto.SerializeToOstream(&fout);
  });
#ifdef ENABLE_LLVM_BITCODE
  if (app.args.count("llvm")) {
    targets::cpu::Native native;
    targets::cpu::Config config;
    native.compile(*program->entry, config);
    native.save((out_dir / "stripe.bc").string());
  }
#endif
  return program;
}

codegen::proto::Configs LoadConfigs() {
  auto app = App::Instance();
  auto config_path = app->args["config"].as<fs::path>();
  auto json = ReadFile(config_path);
  return ParseConfig<codegen::proto::Configs>(json);
}

std::shared_ptr<Program> Main() {
  auto app = App::Instance();
  auto input_path = app->args["input"].as<fs::path>();
  auto out_dir = app->args["outdir"].as<fs::path>();
  auto configs = LoadConfigs();
  auto target = configs.configs().at(app->args["target"].as<std::string>());
  auto stage = target.stages().at(app->args["stage"].as<std::string>());
  OptimizeOptions options;
  if (app->args.count("dump-passes")) {
    options.dump_passes = true;
    options.dbg_dir = out_dir / "passes";
  }
  if (app->args.count("dump-passes-proto")) {
    options.dump_passes_proto = true;
    options.dbg_dir = out_dir / "passes";
  }
  return DefaultStage(*app, input_path, out_dir, stage, options);
}

}  // namespace pmlc
}  // namespace tile
}  // namespace vertexai
