// Copyright 2018 Intel Corporation

#pragma once

#include <string>

namespace vertexai {
namespace env {

// Environment variable helpers.
// We use our own environment variable access routines so that we can read them
// from the correct locations on various systems even when they're being modified
// by higher-level components, e.g. Python.

// Reads the requested environment variable.  If the variable is not set, an empty
// string is returned.
std::string Get(std::string key);

// Writes the requested environment variable.
void Set(std::string key, std::string value);

}  // namespace env
}  // namespace vertexai
