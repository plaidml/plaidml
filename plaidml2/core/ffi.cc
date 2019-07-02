// Copyright 2019 Intel Corporation.

#include "plaidml2/core/ffi.h"

#include <boost/filesystem.hpp>

#include "base/util/env.h"
#include "plaidml2/core/internal.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using vertexai::tile::DataType;
using vertexai::tile::TensorDimension;
using vertexai::tile::TensorShape;
namespace fs = boost::filesystem;

namespace {

std::map<std::string, std::string> g_settings;

fs::path user_path() {
  auto home = vertexai::env::Get("HOME");
  if (home.size()) {
    return home;
  }
  auto user_profile = vertexai::env::Get("USERPROFILE");
  if (user_profile.size()) {
    return user_profile;
  }
  auto home_drive = vertexai::env::Get("HOMEDRIVE");
  auto home_path = vertexai::env::Get("HOMEPATH");
  if (home_drive.size() && home_path.size()) {
    return home_drive + home_path;
  }
  throw std::runtime_error("Could not detect HOME/USERPROFILE");
}

void load_settings() {
  fs::path settings_path = vertexai::env::Get("PLAIDML_SETTINGS");
  if (!fs::exists(settings_path)) {
    settings_path = user_path() / ".plaidml2";
  }
  if (!fs::exists(settings_path)) {
    LOG(WARNING) << "No PlaidML settings found.";
    return;
  }
  fs::ifstream file(settings_path);
  for (std::string line; std::getline(file, line);) {
    auto pos = line.find('=');
    if (pos != std::string::npos) {
      auto key = line.substr(0, pos);
      auto value = line.substr(pos + 1);
      IVLOG(1, key << " = " << value);
      vertexai::env::Set(key, value);
      g_settings[key] = value;
    }
  }
}

}  // namespace

extern "C" {

void plaidml_init(plaidml_error* err) {
  static std::atomic<bool> is_initialized{false};
  ffi_wrap_void(err, [&] {
    if (!is_initialized) {
      auto level_str = vertexai::env::Get("PLAIDML_VERBOSE");
      if (level_str.size()) {
        auto level = std::atoi(level_str.c_str());
        if (level) {
          el::Loggers::setVerboseLevel(level);
        }
      }
      IVLOG(1, "plaidml_init");
      load_settings();
      is_initialized = true;
    }
  });
}

size_t plaidml_settings_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return g_settings.size();
  });
}

void plaidml_settings_list(  //
    plaidml_error* err,      //
    size_t nitems,           //
    plaidml_string** keys,   //
    plaidml_string** values) {
  ffi_wrap_void(err, [&] {
    size_t i = 0;
    for (const auto& kvp : g_settings) {
      keys[i] = new plaidml_string{kvp.first};
      values[i] = new plaidml_string{kvp.second};
      i++;
    }
  });
}

const char* plaidml_string_ptr(plaidml_string* str) { return str->str.c_str(); }

void plaidml_string_free(plaidml_string* str) {
  plaidml_error err;
  ffi_wrap_void(&err, [&] {  //
    delete str;
  });
}

void plaidml_shape_free(  //
    plaidml_error* err,   //
    plaidml_shape* shape) {
  ffi_wrap_void(err, [&] {  //
    delete shape;
  });
}

plaidml_shape* plaidml_shape_alloc(  //
    plaidml_error* err,              //
    plaidml_datatype dtype,          //
    size_t ndims,                    //
    const int64_t* sizes,            //
    const int64_t* strides) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    std::vector<TensorDimension> dims(ndims);
    for (size_t i = 0; i < ndims; i++) {
      dims[i] = TensorDimension{strides[i], static_cast<uint64_t>(sizes[i])};
    }
    return new plaidml_shape{TensorShape{static_cast<DataType>(dtype), dims}};
  });
}

plaidml_string* plaidml_shape_repr(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {
    std::stringstream ss;
    ss << shape->shape;
    return new plaidml_string{ss.str()};
  });
}

size_t plaidml_shape_get_ndims(  //
    plaidml_error* err,          //
    plaidml_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->shape.dims.size();
  });
}

plaidml_datatype plaidml_shape_get_dtype(  //
    plaidml_error* err,                    //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {  //
    return static_cast<plaidml_datatype>(shape->shape.type);
  });
}

int64_t plaidml_shape_get_dim_size(  //
    plaidml_error* err,              //
    plaidml_shape* shape,            //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).size;
  });
}

int64_t plaidml_shape_get_dim_stride(  //
    plaidml_error* err,                //
    plaidml_shape* shape,              //
    size_t dim) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.dims.at(dim).stride;
  });
}

uint64_t plaidml_shape_get_nbytes(  //
    plaidml_error* err,             //
    plaidml_shape* shape) {
  return ffi_wrap<int64_t>(err, 0, [&] {  //
    return shape->shape.byte_size();
  });
}

}  // extern "C"
