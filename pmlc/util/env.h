// Copyright 2018 Intel Corporation

#pragma once

#include <string>

namespace pmlc::util {

// Environment variable helpers.
// We use our own environment variable access routines so that we can read them
// from the correct locations on various systems even when they're being
// modified by higher-level components, e.g. Python.

// Reads the requested environment variable.  If the variable is not set, an
// empty string is returned.
std::string getEnvVar(const std::string &key,
                      const std::string &default_value = "");

// Writes the requested environment variable.
void setEnvVar(const std::string &key, const std::string &value);

} // namespace pmlc::util
