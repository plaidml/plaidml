/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include "utils/utils.hpp"

#include <assert.h>
#include <fstream>
#include <iostream>

namespace level_zero_tests {

ze_context_handle_t get_default_context() {
  ze_result_t result = ZE_RESULT_SUCCESS;

  static ze_context_handle_t context = nullptr;

  if (context) {
    return context;
  }

  ze_context_desc_t context_desc = {};
  context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  result = zeContextCreate(get_default_driver(), &context_desc, &context);

  if (ZE_RESULT_SUCCESS != result) {
    throw std::runtime_error("zeContextCreate failed: " + to_string(result));
  }

  return context;
}

ze_driver_handle_t get_default_driver() {
  ze_result_t result = ZE_RESULT_SUCCESS;

  static ze_driver_handle_t driver = nullptr;
  int default_idx = 0;

  if (driver)
    return driver;

  char *user_driver_index = getenv("LZT_DEFAULT_DRIVER_IDX");
  if (user_driver_index != nullptr) {
    default_idx = std::stoi(user_driver_index);
  }

  std::vector<ze_driver_handle_t> drivers =
      level_zero_tests::get_all_driver_handles();
  if (drivers.size() == 0) {
    throw std::runtime_error("zeDriverGet failed: " + to_string(result));
  }

  if (default_idx >= drivers.size()) {
    LOG_ERROR << "Default Driver index " << default_idx
              << " invalid on this machine.";
    throw std::runtime_error("Get Default Driver failed");
  }
  driver = drivers[default_idx];
  if (!driver) {
    LOG_ERROR << "Invalid Driver handle at index " << default_idx;
    throw std::runtime_error("Get Default Driver failed");
  }

  LOG_INFO << "Default Driver retrieved at index " << default_idx;

  return driver;
}

ze_context_handle_t create_context() {
  return create_context(get_default_driver());
}

ze_context_handle_t create_context(ze_driver_handle_t driver) {
  ze_context_handle_t context;
  ze_context_desc_t ctxtDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  assert(ZE_RESULT_SUCCESS == zeContextCreate(driver, &ctxtDesc, &context));
  return context;
}

void destroy_context(ze_context_handle_t context) {
  assert(ZE_RESULT_SUCCESS == zeContextDestroy(context));
}

ze_device_handle_t get_default_device(ze_driver_handle_t driver) {
  ze_result_t result = ZE_RESULT_SUCCESS;

  static ze_device_handle_t device = nullptr;
  int default_idx = 0;
  char *default_name = nullptr;
  if (device)
    return device;

  char *user_device_index = getenv("LZT_DEFAULT_DEVICE_IDX");
  if (user_device_index != nullptr) {
    default_idx = std::stoi(user_device_index);
  }
  default_name = getenv("LZT_DEFAULT_DEVICE_NAME");

  std::vector<ze_device_handle_t> devices =
      level_zero_tests::get_ze_devices(driver);
  if (devices.size() == 0) {
    throw std::runtime_error("zeDeviceGet failed: " + to_string(result));
  }

  if (default_name != nullptr) {
    LOG_INFO << "Default Device to use has NAME:" << default_name;
    for (auto d : devices) {
      ze_device_properties_t device_props =
          level_zero_tests::get_device_properties(d);
      LOG_TRACE << "Device Name :" << device_props.name;
      if (strcmp(default_name, device_props.name) == 0) {
        device = d;
        break;
      }
    }
    if (!device) {
      LOG_ERROR << "Default Device name " << default_name
                << " invalid on this machine.";
      throw std::runtime_error("Get Default Device failed");
    }
  } else {
    if (default_idx >= devices.size()) {
      LOG_ERROR << "Default Device index " << default_idx
                << " invalid on this machine.";
      throw std::runtime_error("Get Default Device failed");
    }
    device = devices[default_idx];
    LOG_INFO << "Default Device retrieved at index " << default_idx;
  }
  return device;
}

uint32_t get_device_count(ze_driver_handle_t driver) {
  uint32_t count = 0;
  ze_result_t result = zeDeviceGet(driver, &count, nullptr);

  if (result) {
    throw std::runtime_error("zeDeviceGet failed: " + to_string(result));
  }
  return count;
}

uint32_t get_driver_handle_count() {
  uint32_t count = 0;
  ze_result_t result = zeDriverGet(&count, nullptr);

  if (result) {
    throw std::runtime_error("zeDriverGet failed: " + to_string(result));
  }
  return count;
}

uint32_t get_sub_device_count(ze_device_handle_t device) {
  uint32_t count = 0;

  ze_result_t result = zeDeviceGetSubDevices(device, &count, nullptr);

  if (result) {
    throw std::runtime_error("zeDeviceGetSubDevices failed: " +
                             to_string(result));
  }
  return count;
}

std::vector<ze_driver_handle_t> get_all_driver_handles() {
  ze_result_t result = ZE_RESULT_SUCCESS;
  uint32_t driver_handle_count = get_driver_handle_count();

  std::vector<ze_driver_handle_t> driver_handles(driver_handle_count);

  result = zeDriverGet(&driver_handle_count, driver_handles.data());

  if (result) {
    throw std::runtime_error("zeDriverGet failed: " + to_string(result));
  }
  return driver_handles;
}

std::vector<ze_device_handle_t> get_devices(ze_driver_handle_t driver) {

  ze_result_t result = ZE_RESULT_SUCCESS;

  uint32_t device_count = get_device_count(driver);
  std::vector<ze_device_handle_t> devices(device_count);

  result = zeDeviceGet(driver, &device_count, devices.data());

  if (result) {
    throw std::runtime_error("zeDeviceGet failed: " + to_string(result));
  }
  return devices;
}

std::string to_string(const ze_api_version_t version) {
  std::stringstream ss;
  ss << ZE_MAJOR_VERSION(version) << "." << ZE_MINOR_VERSION(version);
  return ss.str();
}

std::string to_string(const ze_result_t result) {
  if (result == ZE_RESULT_SUCCESS) {
    return "ZE_RESULT_SUCCESS";
  } else if (result == ZE_RESULT_NOT_READY) {
    return "ZE_RESULT_NOT_READY";
  } else if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  } else if (result == ZE_RESULT_ERROR_DEVICE_LOST) {
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  } else if (result == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  } else if (result == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  } else if (result == ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS) {
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  } else if (result == ZE_RESULT_ERROR_NOT_AVAILABLE) {
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_VERSION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_HANDLE) {
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  } else if (result == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE) {
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_POINTER) {
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  } else if (result == ZE_RESULT_ERROR_INVALID_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  } else if (result == ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT) {
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  } else if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  } else if (result == ZE_RESULT_ERROR_INVALID_NATIVE_BINARY) {
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  } else if (result == ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE) {
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  } else if (result == ZE_RESULT_ERROR_OVERLAPPING_REGIONS) {
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  } else if (result == ZE_RESULT_ERROR_UNKNOWN) {
    return "ZE_RESULT_ERROR_UNKNOWN";
  } else {
    throw std::runtime_error("Unknown ze_result_t value: " +
                             std::to_string(static_cast<int>(result)));
  }
}

std::string to_string(const ze_bool_t ze_bool) {
  if (ze_bool) {
    return "True";
  } else {
    return "False";
  }
}

std::string to_string(const ze_command_queue_flag_t flags) {
  if (flags == 0) {
    return "Default";
  } else if (flags == ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY) {
    return "ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY";
  } else if (flags == ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32) {
    return "ZE_COMMAND_QUEUE_FLAG_FORCE_UINT32";
  } else {
    return "Unknown ze_command_queue_flag_t value: " +
           std::to_string(static_cast<int>(flags));
  }
}

std::string to_string(const ze_command_queue_mode_t mode) {
  if (mode == ZE_COMMAND_QUEUE_MODE_DEFAULT) {
    return "ZE_COMMAND_QUEUE_MODE_DEFAULT";
  } else if (mode == ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS) {
    return "ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS";
  } else if (mode == ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS) {
    return "ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS";
  } else {
    return "Unknown ze_command_queue_mode_t value: " +
           std::to_string(static_cast<int>(mode));
  }
}

std::string to_string(const ze_command_queue_priority_t priority) {
  if (priority == ZE_COMMAND_QUEUE_PRIORITY_NORMAL) {
    return "ZE_COMMAND_QUEUE_PRIORITY_NORMAL";
  } else if (priority == ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW) {
    return "ZE_COMMAND_QUEUE_PRIORITY_LOW";
  } else if (priority == ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH) {
    return "ZE_COMMAND_QUEUE_PRIORITY_HIGH";
  } else {
    return "Unknown ze_command_queue_priority_t value: " +
           std::to_string(static_cast<int>(priority));
  }
}

std::string to_string(const ze_image_format_layout_t layout) {
  if (layout == ZE_IMAGE_FORMAT_LAYOUT_8) {
    return "ZE_IMAGE_FORMAT_LAYOUT_8";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_16) {
    return "ZE_IMAGE_FORMAT_LAYOUT_16";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_32) {
    return "ZE_IMAGE_FORMAT_LAYOUT_32";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_8_8) {
    return "ZE_IMAGE_FORMAT_LAYOUT_8_8";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8) {
    return "ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_16_16) {
    return "ZE_IMAGE_FORMAT_LAYOUT_16_16";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16) {
    return "ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_32_32) {
    return "ZE_IMAGE_FORMAT_LAYOUT_32_32";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32) {
    return "ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2) {
    return "ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_11_11_10) {
    return "ZE_IMAGE_FORMAT_LAYOUT_11_11_10";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_5_6_5) {
    return "ZE_IMAGE_FORMAT_LAYOUT_5_6_5";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1) {
    return "ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4) {
    return "ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_Y8) {
    return "ZE_IMAGE_FORMAT_LAYOUT_Y8";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_NV12) {
    return "ZE_IMAGE_FORMAT_LAYOUT_NV12";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_YUYV) {
    return "ZE_IMAGE_FORMAT_LAYOUT_YUYV";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_VYUY) {
    return "ZE_IMAGE_FORMAT_LAYOUT_VYUY";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_YVYU) {
    return "ZE_IMAGE_FORMAT_LAYOUT_YVYU";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_UYVY) {
    return "ZE_IMAGE_FORMAT_LAYOUT_UYVY";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_AYUV) {
    return "ZE_IMAGE_FORMAT_LAYOUT_AYUV";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_P010) {
    return "ZE_IMAGE_FORMAT_LAYOUT_P010";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_Y410) {
    return "ZE_IMAGE_FORMAT_LAYOUT_Y410";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_P012) {
    return "ZE_IMAGE_FORMAT_LAYOUT_P012";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_Y16) {
    return "ZE_IMAGE_FORMAT_LAYOUT_Y16";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_P016) {
    return "ZE_IMAGE_FORMAT_LAYOUT_P016";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_Y216) {
    return "ZE_IMAGE_FORMAT_LAYOUT_Y216";
  } else if (layout == ZE_IMAGE_FORMAT_LAYOUT_P216) {
    return "ZE_IMAGE_FORMAT_LAYOUT_P216";
  } else {
    return "Unknown ze_image_format_layout_t value: " +
           std::to_string(static_cast<int>(layout));
  }
}

ze_image_format_layout_t to_layout(const std::string layout) {
  if (layout == "8") {
    return ZE_IMAGE_FORMAT_LAYOUT_8;
  } else if (layout == "16") {
    return ZE_IMAGE_FORMAT_LAYOUT_16;
  } else if (layout == "32") {
    return ZE_IMAGE_FORMAT_LAYOUT_32;
  } else if (layout == "8_8") {
    return ZE_IMAGE_FORMAT_LAYOUT_8_8;
  } else if (layout == "8_8_8_8") {
    return ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
  } else if (layout == "16_16") {
    return ZE_IMAGE_FORMAT_LAYOUT_16_16;
  } else if (layout == "16_16_16_16") {
    return ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
  } else if (layout == "32_32") {
    return ZE_IMAGE_FORMAT_LAYOUT_32_32;
  } else if (layout == "32_32_32_32") {
    return ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
  } else if (layout == "10_10_10_2") {
    return ZE_IMAGE_FORMAT_LAYOUT_10_10_10_2;
  } else if (layout == "11_11_10") {
    return ZE_IMAGE_FORMAT_LAYOUT_11_11_10;
  } else if (layout == "5_6_5") {
    return ZE_IMAGE_FORMAT_LAYOUT_5_6_5;
  } else if (layout == "5_5_5_1") {
    return ZE_IMAGE_FORMAT_LAYOUT_5_5_5_1;
  } else if (layout == "4_4_4_4") {
    return ZE_IMAGE_FORMAT_LAYOUT_4_4_4_4;
  } else if (layout == "Y8") {
    return ZE_IMAGE_FORMAT_LAYOUT_Y8;
  } else if (layout == "NV12") {
    return ZE_IMAGE_FORMAT_LAYOUT_NV12;
  } else if (layout == "YUYV") {
    return ZE_IMAGE_FORMAT_LAYOUT_YUYV;
  } else if (layout == "VYUY") {
    return ZE_IMAGE_FORMAT_LAYOUT_VYUY;
  } else if (layout == "YVYU") {
    return ZE_IMAGE_FORMAT_LAYOUT_YVYU;
  } else if (layout == "UYVY") {
    return ZE_IMAGE_FORMAT_LAYOUT_UYVY;
  } else if (layout == "AYUV") {
    return ZE_IMAGE_FORMAT_LAYOUT_AYUV;
  } else if (layout == "P010") {
    return ZE_IMAGE_FORMAT_LAYOUT_P010;
  } else if (layout == "Y410") {
    return ZE_IMAGE_FORMAT_LAYOUT_Y410;
  } else if (layout == "P012") {
    return ZE_IMAGE_FORMAT_LAYOUT_P012;
  } else if (layout == "Y16") {
    return ZE_IMAGE_FORMAT_LAYOUT_Y16;
  } else if (layout == "P016") {
    return ZE_IMAGE_FORMAT_LAYOUT_P016;
  } else if (layout == "Y216") {
    return ZE_IMAGE_FORMAT_LAYOUT_Y216;
  } else if (layout == "P216") {
    return ZE_IMAGE_FORMAT_LAYOUT_P216;
  } else {
    std::cout << "Unknown ze_image_format_layout_t value: " << layout;
    return static_cast<ze_image_format_layout_t>(-1);
  }
}

std::string to_string(const ze_image_format_type_t type) {
  if (type == ZE_IMAGE_FORMAT_TYPE_UINT) {
    return "ZE_IMAGE_FORMAT_TYPE_UINT";
  } else if (type == ZE_IMAGE_FORMAT_TYPE_SINT) {
    return "ZE_IMAGE_FORMAT_TYPE_SINT";
  } else if (type == ZE_IMAGE_FORMAT_TYPE_UNORM) {
    return "ZE_IMAGE_FORMAT_TYPE_UNORM";
  } else if (type == ZE_IMAGE_FORMAT_TYPE_SNORM) {
    return "ZE_IMAGE_FORMAT_TYPE_SNORM";
  } else if (type == ZE_IMAGE_FORMAT_TYPE_FLOAT) {
    return "ZE_IMAGE_FORMAT_TYPE_FLOAT";
  } else {
    return "Unknown ze_image_format_type_t value: " +
           std::to_string(static_cast<int>(type));
  }
}

ze_image_format_type_t to_format_type(const std::string format_type) {
  if (format_type == "UINT") {
    return ZE_IMAGE_FORMAT_TYPE_UINT;
  } else if (format_type == "SINT") {
    return ZE_IMAGE_FORMAT_TYPE_SINT;
  } else if (format_type == "UNORM") {
    return ZE_IMAGE_FORMAT_TYPE_UNORM;
  } else if (format_type == "SNORM") {
    return ZE_IMAGE_FORMAT_TYPE_SNORM;
  } else if (format_type == "FLOAT") {
    return ZE_IMAGE_FORMAT_TYPE_FLOAT;
  } else {
    std::cout << "Unknown ze_image_format_type_t value: ";
    return (static_cast<ze_image_format_type_t>(-1));
  }
}

uint32_t num_bytes_per_pixel(ze_image_format_layout_t layout) {
  switch (layout) {
  case ZE_IMAGE_FORMAT_LAYOUT_8:
    return 1;
  case ZE_IMAGE_FORMAT_LAYOUT_16:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_32:
    return 4;
  case ZE_IMAGE_FORMAT_LAYOUT_8_8:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_16_16:
    return 4;
  case ZE_IMAGE_FORMAT_LAYOUT_32_32:
    return 8;
  case ZE_IMAGE_FORMAT_LAYOUT_5_6_5:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_11_11_10:
    return 4;
  case ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8:
    return 4;
  case ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16:
    return 8;
  case ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32:
    return 16;
  case ZE_IMAGE_FORMAT_LAYOUT_AYUV:
    return 4;
  case ZE_IMAGE_FORMAT_LAYOUT_NV12: // 12 bits per pixel
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_P010: // 10 bits per pixel
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_P012: // 12 bits per pixel
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_P016:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_P216:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_UYVY:
  case ZE_IMAGE_FORMAT_LAYOUT_VYUY:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_Y8:
    return 1;
  case ZE_IMAGE_FORMAT_LAYOUT_Y16:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_Y216:
    return 2;
  case ZE_IMAGE_FORMAT_LAYOUT_Y410: // 10 bits per pixel
    return 2;
  default:
    LOG_ERROR << "Unrecognized image format layout: " << layout;
    return 0;
  }
}

std::string to_string(const ze_image_format_swizzle_t swizzle) {
  if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_R) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_R";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_G) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_G";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_B) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_B";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_A) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_A";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_0) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_0";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_1) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_1";
  } else if (swizzle == ZE_IMAGE_FORMAT_SWIZZLE_X) {
    return "ZE_IMAGE_FORMAT_SWIZZLE_X";
  } else {
    return "Unknown ze_image_format_swizzle_t value: " +
           std::to_string(static_cast<int>(swizzle));
  }
}

std::string to_string(const ze_image_flag_t flag) {
  std::string flags = "";
  if (flag & ZE_IMAGE_FLAG_KERNEL_WRITE) {
    flags.append("|ZE_IMAGE_FLAG_KERNEL_WRITE|");
  }
  if (flag & ZE_IMAGE_FLAG_BIAS_UNCACHED) {
    flags.append("|ZE_IMAGE_FLAG_BIAS_UNCACHED|");
  }

  return flags;
}

ze_image_flags_t to_image_flag(const std::string flag) {

  // by default setting to READ
  ze_image_flags_t image_flags = 0;

  // check if "READ" position is found in flag string
  if (flag.find("WRITE") != std::string::npos) {
    image_flags = image_flags | ZE_IMAGE_FLAG_KERNEL_WRITE;
  }
  if (flag.find("UNCACHED") != std::string::npos) {
    image_flags = image_flags | ZE_IMAGE_FLAG_BIAS_UNCACHED;
  }

  return image_flags;
}

std::string to_string(const ze_image_type_t type) {
  if (type == ZE_IMAGE_TYPE_1D) {
    return "ZE_IMAGE_TYPE_1D";
  } else if (type == ZE_IMAGE_TYPE_2D) {
    return "ZE_IMAGE_TYPE_2D";
  } else if (type == ZE_IMAGE_TYPE_3D) {
    return "ZE_IMAGE_TYPE_3D";
  } else if (type == ZE_IMAGE_TYPE_1DARRAY) {
    return "ZE_IMAGE_TYPE_1DARRAY";
  } else if (type == ZE_IMAGE_TYPE_2DARRAY) {
    return "ZE_IMAGE_TYPE_2DARRAY";
  } else {
    return "Unknown ze_image_type_t value: " +
           std::to_string(static_cast<int>(type));
  }
}

ze_image_type_t to_image_type(const std::string type) {
  if (type == "1D") {
    return ZE_IMAGE_TYPE_1D;
  } else if (type == "2D") {
    return ZE_IMAGE_TYPE_2D;
  } else if (type == "3D") {
    return ZE_IMAGE_TYPE_3D;
  } else if (type == "1DARRAY") {
    return ZE_IMAGE_TYPE_1DARRAY;
  } else if (type == "2DARRAY") {
    return ZE_IMAGE_TYPE_2DARRAY;
  } else {
    std::cout << "Unknown ze_image_type_t value: ";
    return (static_cast<ze_image_type_t>(-1));
  }
}

std::string to_string(const ze_device_fp_flags_t capabilities) {
  std::string capabilities_str = "";
  if (capabilities == 0) {
    capabilities_str.append("|NONE|");
    return capabilities_str;
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_DENORM) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_DENORM|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_INF_NAN) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_INF_NAN|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_ROUND_TO_INF) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_ROUND_TO_INF|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_FMA) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_FMA|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT|");
  }
  if (capabilities & ZE_DEVICE_FP_FLAG_SOFT_FLOAT) {
    capabilities_str.append("|ZE_DEVICE_FP_FLAG_SOFT_FLOAT|");
  }
  return capabilities_str;
}

std::string to_string(const ze_driver_uuid_t uuid) {
  std::ostringstream result;
  result << "{";
  for (int i = 0; i < ZE_MAX_DRIVER_UUID_SIZE; i++) {
    result << "0x" << std::hex << uuid.id[i] << ",";
  }
  result << "\b}";
  return result.str();
}

std::string to_string(const ze_native_kernel_uuid_t uuid) {
  std::ostringstream result;
  result << "{";
  for (int i = 0; i < ZE_MAX_NATIVE_KERNEL_UUID_SIZE; i++) {
    result << "0x" << std::hex << uuid.id[i] << ",";
  }
  result << "\b}";
  return result.str();
}

void print_driver_version(ze_driver_handle_t driver) {
  ze_driver_properties_t properties;

  properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
  properties.pNext = nullptr;
  ze_result_t result = zeDriverGetProperties(driver, &properties);
  if (result) {
    std::runtime_error("zeDriverGetProperties failed: " + to_string(result));
  }
  LOG_TRACE << "Driver version retrieved";
  LOG_INFO << "Driver version: " << properties.driverVersion;
}

void print_driver_overview(const ze_driver_handle_t driver) {
  ze_result_t result = ZE_RESULT_SUCCESS;

  ze_device_properties_t device_properties;
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  device_properties.pNext = nullptr;
  auto devices = get_devices(driver);
  int device_index = 0;
  LOG_INFO << "Device Count: " << devices.size();
  for (auto device : devices) {
    result = zeDeviceGetProperties(device, &device_properties);
    if (result) {
      std::runtime_error("zeDeviceGetDeviceProperties failed: " +
                         to_string(result));
    }
    LOG_TRACE << "Device properties retrieved for device " << device_index;
    LOG_INFO << "Device name: " << device_properties.name;
    device_index++;
  }

  ze_api_version_t api_version;
  result = zeDriverGetApiVersion(driver, &api_version);
  if (result) {
    throw std::runtime_error("zeDriverGetApiVersion failed: " +
                             to_string(result));
  }
  LOG_TRACE << "Driver API version retrieved";

  LOG_INFO << "Driver API version: " << to_string(api_version);
}

void print_driver_overview(const std::vector<ze_driver_handle_t> driver) {
  for (const ze_driver_handle_t driver : driver) {
    print_driver_version(driver);
    print_driver_overview(driver);
  }
}

void print_platform_overview(const std::string context) {
  LOG_INFO << "Platform overview";
  if (context.size() > 0) {
    LOG_INFO << " (Context: " << context << ")";
  }

  const std::vector<ze_driver_handle_t> drivers = get_all_driver_handles();
  LOG_INFO << "Driver Handle count: " << drivers.size();

  print_driver_overview(drivers);
}

void print_platform_overview() { print_platform_overview(""); }

std::vector<uint8_t> load_binary_file(const std::string &file_path) {
  LOG_ENTER_FUNCTION
  LOG_DEBUG << "File path: " << file_path;
  std::ifstream stream(file_path, std::ios::in | std::ios::binary);

  std::vector<uint8_t> binary_file;
  if (!stream.good()) {
    LOG_ERROR << "Failed to load binary file: " << file_path << "error "
              << strerror(errno);

    LOG_EXIT_FUNCTION
    return binary_file;
  }

  size_t length = 0;
  stream.seekg(0, stream.end);
  length = static_cast<size_t>(stream.tellg());
  stream.seekg(0, stream.beg);
  LOG_DEBUG << "Binary file length: " << length;

  binary_file.resize(length);
  stream.read(reinterpret_cast<char *>(binary_file.data()), length);
  LOG_DEBUG << "Binary file loaded";

  LOG_EXIT_FUNCTION
  return binary_file;
}

void save_binary_file(const std::vector<uint8_t> &data,
                      const std::string &file_path) {
  LOG_ENTER_FUNCTION
  LOG_DEBUG << "File path: " << file_path;

  std::ofstream stream(file_path, std::ios::out | std::ios::binary);
  stream.write(reinterpret_cast<const char *>(data.data()),
               size_in_bytes(data));

  LOG_EXIT_FUNCTION
}

} // namespace level_zero_tests

std::ostream &operator<<(std::ostream &os, const ze_api_version_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_result_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_bool_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_command_queue_flag_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_command_queue_mode_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os,
                         const ze_command_queue_priority_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_image_format_layout_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_image_format_type_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_image_format_swizzle_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_image_flag_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_image_type_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_device_fp_flags_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_driver_uuid_t &x) {
  return os << level_zero_tests::to_string(x);
}

std::ostream &operator<<(std::ostream &os, const ze_native_kernel_uuid_t &x) {
  return os << level_zero_tests::to_string(x);
}

bool operator==(const ze_device_uuid_t &id_a, const ze_device_uuid_t &id_b) {
  return !(memcmp(id_a.id, id_b.id, ZE_MAX_DEVICE_UUID_SIZE));
}

bool operator!=(const ze_device_uuid_t &id_a, const ze_device_uuid_t &id_b) {
  return memcmp(id_a.id, id_b.id, ZE_MAX_DEVICE_UUID_SIZE);
}
