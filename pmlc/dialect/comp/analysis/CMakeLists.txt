pml_cc_library(
  NAME analysis
  HDRS
    mem_sync_tracker.h
  SRCS
    mem_sync_tracker.cc
  DEPS
    pmlc::dialect::comp::ir
)
