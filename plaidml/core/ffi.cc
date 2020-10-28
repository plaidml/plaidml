// Copyright 2019 Intel Corporation.

#include "plaidml/core/ffi.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "plaidml/core/internal.h"
#include "plaidml/core/settings.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using plaidml::core::convertFromDataType;
using plaidml::core::convertIntoDataType;
using plaidml::core::ffi_vector;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::Settings;
using pmlc::util::AdoptedBuffer;
using pmlc::util::DataType;
using pmlc::util::SimpleBuffer;
using pmlc::util::TensorShape;

namespace ast = pmlc::ast;

extern const char* PLAIDML_VERSION;

namespace plaidml::core {

plaidml_datatype convertIntoDataType(DataType type) { return static_cast<plaidml_datatype>(type); }

DataType convertFromDataType(plaidml_datatype dtype) { return static_cast<DataType>(dtype); }

}  // namespace plaidml::core

extern "C" {

void plaidml_init(plaidml_error* err) {
  static std::once_flag is_initialized;
  ffi_wrap_void(err, [&] {
    std::call_once(is_initialized, []() {
      auto level_str = pmlc::util::getEnvVar("PLAIDML_VERBOSE");
      if (level_str.size()) {
        auto level = std::atoi(level_str.c_str());
        if (level) {
          el::Loggers::setVerboseLevel(level);
        }
      }
      IVLOG(1, "plaidml_init");
      Settings::Instance()->load();
    });
  });
}

void plaidml_shutdown(plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    IVLOG(1, "plaidml_shutdown");
  });
}

const char* plaidml_version(  //
    plaidml_error* err) {
  return ffi_wrap<const char*>(err, nullptr, [&] {  //
    return PLAIDML_VERSION;
  });
}

size_t plaidml_settings_list_count(  //
    plaidml_error* err) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return Settings::Instance()->all().size();
  });
}

void plaidml_kvps_free(  //
    plaidml_error* err,  //
    plaidml_kvps* kvps) {
  ffi_wrap_void(err, [&] {
    delete[] kvps->elts;
    delete kvps;
  });
}

plaidml_kvps* plaidml_settings_list(  //
    plaidml_error* err) {
  return ffi_wrap<plaidml_kvps*>(err, nullptr, [&] {
    const auto& settings = Settings::Instance()->all();
    auto ret = new plaidml_kvps{settings.size(), new plaidml_kvp[settings.size()]};
    size_t i = 0;
    for (auto it = settings.begin(); it != settings.end(); it++, i++) {
      ret->elts[i].key = new plaidml_string{it->first};
      ret->elts[i].value = new plaidml_string{it->second};
    }
    return ret;
  });
}

void plaidml_settings_load(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->load();
  });
}

void plaidml_settings_save(  //
    plaidml_error* err) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->save();
  });
}

plaidml_string* plaidml_settings_get(  //
    plaidml_error* err,                //
    const char* key) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&]() -> plaidml_string* {  //
    return new plaidml_string{Settings::Instance()->get(key)};
  });
}

void plaidml_settings_set(  //
    plaidml_error* err,     //
    const char* key,        //
    const char* value) {
  ffi_wrap_void(err, [&] {  //
    Settings::Instance()->set(key, value);
  });
}

const char* plaidml_string_ptr(plaidml_string* str) {  //
  return str->str.c_str();
}

void plaidml_string_free(plaidml_string* str) {
  plaidml_error err;
  ffi_wrap_void(&err, [&] {  //
    delete str;
  });
}

void plaidml_integers_free(  //
    plaidml_error* err,      //
    plaidml_integers* ints) {
  ffi_wrap_void(err, [&] {
    delete[] ints->elts;
    delete ints;
  });
}

void plaidml_strings_free(  //
    plaidml_error* err,     //
    plaidml_strings* strs) {
  ffi_wrap_void(err, [&] {
    delete[] strs->elts;
    delete strs;
  });
}

void plaidml_shapes_free(  //
    plaidml_error* err,    //
    plaidml_shapes* shapes) {
  ffi_wrap_void(err, [&] {
    delete[] shapes->elts;
    delete shapes;
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
    size_t rank,                     //
    const int64_t* sizes,            //
    const int64_t* strides) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    auto vecSizes = llvm::makeArrayRef(sizes, rank).vec();
    llvm::ArrayRef<int64_t> vecStrides;
    if (strides) {
      vecStrides = llvm::makeArrayRef(strides, rank).vec();
    }
    return new plaidml_shape{TensorShape(convertFromDataType(dtype), vecSizes, vecStrides)};
  });
}

plaidml_shape* plaidml_shape_clone(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_shape");
    return new plaidml_shape{shape->shape};
  });
}

plaidml_string* plaidml_shape_repr(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{shape->shape.str()};
  });
}

size_t plaidml_shape_get_rank(  //
    plaidml_error* err,         //
    plaidml_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->shape.getRank();
  });
}

plaidml_datatype plaidml_shape_get_dtype(  //
    plaidml_error* err,                    //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID,
                                    [&] { return convertIntoDataType(shape->shape.elementType); });
}

plaidml_integers* plaidml_shape_get_sizes(  //
    plaidml_error* err,                     //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_integers*>(err, nullptr, [&] {  //
    return ffi_vector<plaidml_integers>(shape->shape.sizes);
  });
}

plaidml_integers* plaidml_shape_get_strides(  //
    plaidml_error* err,                       //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_integers*>(err, nullptr, [&] {  //
    return ffi_vector<plaidml_integers>(shape->shape.strides);
  });
}

uint64_t plaidml_shape_get_nbytes(  //
    plaidml_error* err,             //
    plaidml_shape* shape) {
  return ffi_wrap<uint64_t>(err, 0, [&]() -> uint64_t {  //
    return shape->shape.getByteSize();
  });
}

void plaidml_buffer_free(  //
    plaidml_error* err,    //
    plaidml_buffer* buffer) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_buffer_free: " << buffer->buffer.get() << ": " << buffer->buffer->size());
    delete buffer;
  });
}

plaidml_buffer* plaidml_buffer_clone(  //
    plaidml_error* err,                //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_buffer_clone: " << buffer->buffer.get());
    return new plaidml_buffer{buffer->buffer};
  });
}

plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    auto buffer = std::make_shared<SimpleBuffer>(shape->shape);
    IVLOG(3, "plaidml_buffer_alloc: " << buffer.get() << ": " << buffer->shape().str());
    return new plaidml_buffer{buffer};
  });
}

plaidml_buffer* plaidml_buffer_adopt(  //
    plaidml_error* err,                //
    plaidml_shape* shape,              //
    char* data,                        //
    size_t size) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_buffer_adopt: " << static_cast<void*>(data));
    auto buffer = std::make_shared<AdoptedBuffer>(shape->shape, size, data);
    return new plaidml_buffer{buffer};
  });
}

plaidml_shape* plaidml_buffer_shape(  //
    plaidml_error* err,               //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {  //
    return new plaidml_shape{buffer->buffer->shape()};
  });
}

char* plaidml_buffer_data(  //
    plaidml_error* err,     //
    plaidml_buffer* buffer) {
  return ffi_wrap<char*>(err, nullptr, [&] {  //
    return buffer->buffer->data();
  });
}

size_t plaidml_buffer_size(  //
    plaidml_error* err,      //
    plaidml_buffer* buffer) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return buffer->buffer->size();
  });
}

void plaidml_program_free(  //
    plaidml_error* err,     //
    plaidml_program* program) {
  ffi_wrap_void(err, [&] {
    IVLOG(3, "plaidml_program_free");
    delete program;
  });
}

plaidml_string* plaidml_program_repr(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{program->program->tileIR};
  });
}

plaidml_shapes* plaidml_program_get_inputs(  //
    plaidml_error* err,                      //
    plaidml_program* program) {
  return ffi_wrap<plaidml_shapes*>(err, nullptr, [&] {  //
    std::vector<TensorShape> shapes;
    for (mlir::Type type : program->program->inputs) {
      shapes.emplace_back(TensorShape::fromType(type));
    }
    return ffi_vector<plaidml_shapes, plaidml_shape>(shapes);
  });
}

plaidml_shapes* plaidml_program_get_outputs(  //
    plaidml_error* err,                       //
    plaidml_program* program) {
  return ffi_wrap<plaidml_shapes*>(err, nullptr, [&] {  //
    std::vector<TensorShape> shapes;
    for (mlir::Type type : program->program->outputs) {
      shapes.emplace_back(TensorShape::fromType(type));
    }
    return ffi_vector<plaidml_shapes, plaidml_shape>(shapes);
  });
}

plaidml_kvps* plaidml_program_get_passes(  //
    plaidml_error* err,                    //
    plaidml_program* program) {
  return ffi_wrap<plaidml_kvps*>(err, nullptr, [&] {
    const auto& passes = program->program->passes;
    auto ret = new plaidml_kvps{passes.size(), new plaidml_kvp[passes.size()]};
    size_t i = 0;
    for (auto it = passes.begin(), eit = passes.end(); it != eit; ++it, ++i) {
      ret->elts[i].key = new plaidml_string{it->name};
      ret->elts[i].value = new plaidml_string{it->ir};
    }
    return ret;
  });
}

void plaidml_program_compile(  //
    plaidml_error* err,        //
    plaidml_program* program,  //
    bool debug,                //
    const char* raw_target) {
  return ffi_wrap_void(err, [&] {  //
    std::string target(raw_target);
    if (target.empty()) {
      target = Settings::Instance()->get("PLAIDML_TARGET");
    }
    auto dumpDir = pmlc::util::getEnvVar("PLAIDML_DUMP");
    program->program->compile(target, /*collectPasses=*/debug, /*dumpDir=*/dumpDir);
  });
}

plaidml_buffer* plaidml_program_save(  //
    plaidml_error* err,                //
    plaidml_program* program) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {  //
    return new plaidml_buffer{program->program->save()};
  });
}

}  // extern "C"
