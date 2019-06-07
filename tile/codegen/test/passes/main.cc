// Copyright 2019, Intel Corporation

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/test/passes/runner.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  using vertexai::tile::codegen::test::passes::VerifyPasses;
  START_EASYLOGGINGPP(argc, argv);

  try {
    po::variables_map args;
    po::options_description opts;
    po::positional_options_description pos_opts;
    opts.add_options()                                                                   //
        ("help,h", "produce help message")                                               //
        ("verbose,v", po::value<int>()->default_value(0), "increase verbosity")          //
        ("passes", po::value<fs::path>()->required(), "directory that contains passes")  //
        ;                                                                                // NOLINT
    pos_opts.add("passes", 1);
    auto parser = po::command_line_parser(argc, argv).options(opts).positional(pos_opts);
    po::store(parser.run(), args);
    if (args.count("help")) {
      std::cout << opts << std::endl;
      return EXIT_SUCCESS;
    }
    if (args.count("verbose")) {
      el::Loggers::setVerboseLevel(args["verbose"].as<int>());
    }
    args.notify();

    auto passes = args["passes"].as<fs::path>();
    if (!fs::exists(passes)) {
      throw std::runtime_error("--passes directory does not exist.");
    }
    if (!fs::is_directory(passes)) {
      throw std::runtime_error("--passes is not a directory.");
    }
    if (VerifyPasses(passes)) {
      return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    auto stacktrace = boost::get_error_info<traced>(ex);
    if (stacktrace) {
      std::cerr << *stacktrace << std::endl;
    }
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return EXIT_FAILURE;
  }
}
