// Copyright 2019, Intel Corporation.

#pragma once

#include <iostream>

extern size_t g_verbosity;

#define IVLOG(lvl, msg)            \
  if (lvl <= g_verbosity) {        \
    std::cout << msg << std::endl; \
  }
