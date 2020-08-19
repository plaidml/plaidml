// Copyright 2019 Intel Corporation.

#include "plaidml/core/ffi.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "mlir/Support/DebugStringHelper.h"

#include "plaidml/core/internal.h"
#include "plaidml/core/settings.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using mlir::FloatType;
using mlir::IntegerType;
using mlir::MLIRContext;
using mlir::Type;
using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
using plaidml::core::GlobalContext;
using plaidml::core::Settings;
using pmlc::util::SimpleBuffer;

extern const char* PLAIDML_VERSION;

namespace plaidml::core {

plaidml_datatype convertIntoDataType(Type type) {
  if (type.isInteger(1)) {
    return PLAIDML_DATA_BOOLEAN;
  }
  if (type.isSignedInteger(8)) {
    return PLAIDML_DATA_INT8;
  }
  if (type.isUnsignedInteger(8)) {
    return PLAIDML_DATA_UINT8;
  }
  if (type.isSignedInteger(16)) {
    return PLAIDML_DATA_INT16;
  }
  if (type.isUnsignedInteger(16)) {
    return PLAIDML_DATA_UINT16;
  }
  if (type.isSignedInteger(32)) {
    return PLAIDML_DATA_INT32;
  }
  if (type.isUnsignedInteger(32)) {
    return PLAIDML_DATA_UINT32;
  }
  if (type.isSignedInteger(64)) {
    return PLAIDML_DATA_INT64;
  }
  if (type.isUnsignedInteger(64)) {
    return PLAIDML_DATA_UINT64;
  }
  if (type.isBF16()) {
    return PLAIDML_DATA_BFLOAT16;
  }
  if (type.isF16()) {
    return PLAIDML_DATA_FLOAT16;
  }
  if (type.isF32()) {
    return PLAIDML_DATA_FLOAT32;
  }
  if (type.isF64()) {
    return PLAIDML_DATA_FLOAT64;
  }
  llvm_unreachable("Cannot convert into plaidml_datatype");
}

Type convertFromDataType(plaidml_datatype dtype, MLIRContext* context) {
  switch (dtype) {
    case PLAIDML_DATA_BOOLEAN:
      return IntegerType::get(1, context);
    case PLAIDML_DATA_BFLOAT16:
      return FloatType::getBF16(context);
    case PLAIDML_DATA_FLOAT16:
      return FloatType::getF16(context);
    case PLAIDML_DATA_FLOAT32:
      return FloatType::getF32(context);
    case PLAIDML_DATA_FLOAT64:
      return FloatType::getF64(context);
    case PLAIDML_DATA_INT8:
      return IntegerType::get(8, IntegerType::Signed, context);
    case PLAIDML_DATA_INT16:
      return IntegerType::get(16, IntegerType::Signed, context);
    case PLAIDML_DATA_INT32:
      return IntegerType::get(32, IntegerType::Signed, context);
    case PLAIDML_DATA_INT64:
      return IntegerType::get(64, IntegerType::Signed, context);
    case PLAIDML_DATA_UINT8:
      return IntegerType::get(8, IntegerType::Unsigned, context);
    case PLAIDML_DATA_UINT16:
      return IntegerType::get(16, IntegerType::Unsigned, context);
    case PLAIDML_DATA_UINT32:
      return IntegerType::get(32, IntegerType::Unsigned, context);
    case PLAIDML_DATA_UINT64:
      return IntegerType::get(64, IntegerType::Unsigned, context);
    default:
      break;
  }
  llvm_unreachable("Invalid plaidml_datatype");
}

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
    auto ctx = GlobalContext::get();
    auto elementType = plaidml::core::convertFromDataType(dtype, ctx->getContext());
    auto sizesArray = llvm::makeArrayRef(sizes, ndims);
    auto stridesArray = llvm::makeArrayRef(strides, ndims);
    auto type = ctx->MakeMemRefType(elementType, sizesArray, stridesArray);
    return new plaidml_shape{type};
  });
}

plaidml_shape* plaidml_shape_clone(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_shape*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_shape");
    return new plaidml_shape{shape->type};
  });
}

plaidml_string* plaidml_shape_repr(  //
    plaidml_error* err,              //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_string*>(err, nullptr, [&] {  //
    return new plaidml_string{mlir::debugString(shape->type)};
  });
}

size_t plaidml_shape_get_rank(  //
    plaidml_error* err,         //
    plaidml_shape* shape) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return shape->type.getRank();
  });
}

plaidml_datatype plaidml_shape_get_dtype(  //
    plaidml_error* err,                    //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_datatype>(err, PLAIDML_DATA_INVALID, [&] {
    auto elementType = shape->type.getElementType();
    if (auto floatType = elementType.dyn_cast<mlir::FloatType>()) {
      switch (floatType.getWidth()) {
        case 16:
          return PLAIDML_DATA_FLOAT16;
        case 32:
          return PLAIDML_DATA_FLOAT32;
        case 64:
          return PLAIDML_DATA_FLOAT64;
        default:
          break;
      }
    }
    if (auto integerType = elementType.dyn_cast<mlir::IntegerType>()) {
      switch (integerType.getWidth()) {
        case 1:
          return PLAIDML_DATA_BOOLEAN;
        case 8:
          return PLAIDML_DATA_INT8;
        case 16:
          return PLAIDML_DATA_INT16;
        case 32:
          return PLAIDML_DATA_INT32;
        case 64:
          return PLAIDML_DATA_INT64;
        default:
          break;
      }
    }
    throw std::runtime_error("Invalid DType for plaidml_shape");
  });
}

plaidml_integers* plaidml_shape_get_sizes(  //
    plaidml_error* err,                     //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_integers*>(err, 0, [&] {
    const auto& sizes = shape->type.getShape();
    auto ret = new plaidml_integers{sizes.size(), new int64_t[sizes.size()]};
    for (unsigned i = 0; i < sizes.size(); i++) {
      if (sizes[i] < 0) {
        ret->elts[i] = 0;
      } else {
        ret->elts[i] = sizes[i];
      }
    }
    return ret;
  });
}

plaidml_integers* plaidml_shape_get_strides(  //
    plaidml_error* err,                       //
    plaidml_shape* shape) {
  return ffi_wrap<plaidml_integers*>(err, 0, [&] {
    int64_t offset;
    llvm::SmallVector<int64_t, 8> strides;
    if (failed(mlir::getStridesAndOffset(shape->type, strides, offset))) {
      throw std::runtime_error("Could not retrieve strides");
    }
    auto ret = new plaidml_integers{strides.size(), new int64_t[strides.size()]};
    for (unsigned i = 0; i < strides.size(); i++) {
      ret->elts[i] = strides[i];
    }
    return ret;
  });
}

uint64_t plaidml_shape_get_nbytes(  //
    plaidml_error* err,             //
    plaidml_shape* shape) {
  return ffi_wrap<uint64_t>(err, 0, [&]() -> uint64_t {  //
    return pmlc::util::getByteSize(shape->type);
  });
}

void plaidml_buffer_free(  //
    plaidml_error* err,    //
    plaidml_buffer* buffer) {
  ffi_wrap_void(err, [&] {  //
    delete buffer;
  });
}

plaidml_buffer* plaidml_buffer_clone(  //
    plaidml_error* err,                //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    IVLOG(3, "plaidml_buffer_clone");
    return new plaidml_buffer{buffer->buffer};
  });
}

plaidml_buffer* plaidml_buffer_alloc(  //
    plaidml_error* err,                //
    const char* device_id,             //
    size_t size) {
  return ffi_wrap<plaidml_buffer*>(err, nullptr, [&] {
    auto buffer = std::make_shared<SimpleBuffer>(size);
    return new plaidml_buffer{buffer};
  });
}

plaidml_view* plaidml_buffer_mmap_current(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    return new plaidml_view{buffer->buffer->MapCurrent()};
  });
}

plaidml_view* plaidml_buffer_mmap_discard(  //
    plaidml_error* err,                     //
    plaidml_buffer* buffer) {
  return ffi_wrap<plaidml_view*>(err, nullptr, [&] {  //
    return new plaidml_view{buffer->buffer->MapDiscard()};
  });
}

void plaidml_view_free(  //
    plaidml_error* err,  //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {  //
    delete view;
  });
}

char* plaidml_view_data(  //
    plaidml_error* err,   //
    plaidml_view* view) {
  return ffi_wrap<char*>(err, nullptr, [&] {  //
    return view->view->data();
  });
}

size_t plaidml_view_size(  //
    plaidml_error* err,    //
    plaidml_view* view) {
  return ffi_wrap<size_t>(err, 0, [&] {  //
    return view->view->size();
  });
}

void plaidml_view_writeback(  //
    plaidml_error* err,       //
    plaidml_view* view) {
  ffi_wrap_void(err, [&] {  //
    view->view->WriteBack();
  });
}

VariantHolder::VariantHolder(const Variant& inner) : inner(inner) {}

}  // extern "C"
