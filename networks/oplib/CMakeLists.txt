pml_cc_binary(
  NAME resnet50
  SRCS
    resnet50.cc
    run.cc
  DEPS
    plaidml::core
    plaidml::edsl
    plaidml::exec
    plaidml::op
    pmlc::rt::all_runtimes
    pmlc::target::all
)
