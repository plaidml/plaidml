// Copyright 2018 Intel Corporation.
//
// This is the PlaidML library interface, used to construct and manipulate
// programs defined as a sequence of operations over tensors.

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <memory>
#else
#include <stdbool.h>
#include <stddef.h>
#endif  // __cplusplus

#include "plaidml/base/base.h"

#define PLAIDML_API VAI_API

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// PlaidML devconf objects represent device configurations.  Devconf objects are
// contained by PlaidML device enumerators, examined and potentially modified by
// PlaidML library consumers, and then used to open PlaidML devices for compute
// access.
#ifdef __cplusplus
struct plaidml_devconf;
#else
typedef struct plaidml_devconf plaidml_devconf;
#endif  // __cplusplus

// Platform configuration properties.  This set may be extended in the future.
typedef enum {
  // Platform names are NUL-terminated strings.
  PLAIDML_DEVICE_ID = 1,

  // Platform configs are NUL-terminated prototxt.
  PLAIDML_DEVICE_CONFIG = 2,

  // Platform descriptions are NUL-terminated prototxt.
  PLAIDML_DEVICE_DESCRIPTION = 3,

  // Platform descriptions are NUL-terminated prototxt.
  PLAIDML_DEVICE_DETAILS = 4
} plaidml_device_property;

// Returns the version of plaidml
PLAIDML_API const char* plaidml_get_version();

// Queries the supplied device configuration property.
//
// The supplied output buffer pointer should point to a property-specific value
// to be filled in with the requested information; the output size is used for
// property versioning and buffer overflow protection.  Fields in the output
// buffer not supported by the property implementation will be zero-filled
// (i.e. zero is the default value for all properties, including unsupported
// properties).
//
// The value pointed to by output_buffer_size_required, if provided, will be
// filled in with the size required for the output buffer.  This is most useful
// with queries that return string values or arrays of values; a common pattern
// is to make a call with a NULL output buffer and zero size, allocate a buffer
// of the indicated output size required, and then to re-issue the query.
PLAIDML_API bool plaidml_query_devconf(vai_ctx* ctx, plaidml_devconf* devconf, plaidml_device_property property,
                                       void* output_buffer, size_t output_buffer_size,
                                       size_t* output_buffer_size_required);

// PlaidML devices are used to supply resources that might be backed by a
// variety of hardware.
#ifdef __cplusplus
struct plaidml_device;
#else
typedef struct plaidml_device plaidml_device;
#endif  // __cplusplus

// Opens a device, using the supplied device configuration.  A NULL
// configuration is permitted; this will use the system default PlaidML compute
// device.
PLAIDML_API plaidml_device* plaidml_open_device(vai_ctx* ctx, plaidml_devconf* devconf);

// Closes a device.  After this call, the device should not be used for any
// subsequent calls.  The device may share resources with other objects (such
// as buffers and functions); those resources will only be released when they
// are no longer needed by those other objects.  Closing a NULL device is a
// no-op.
PLAIDML_API void plaidml_close_device(plaidml_device* device);

// PlaidML device enumerators offer access to sets of compute devices.  Enumerators may
// supply devices that are in-process, machine-local, or cross-network, depending on
// the deployment configuration.
#ifdef __cplusplus
struct plaidml_device_enumerator;
#else
typedef struct plaidml_device_enumerator plaidml_device_enumerator;
#endif  // __cplusplus

// Allocates a device enumerator and initializes it using automatic configuration.
//
// If the supplied callback is NULL, the call will block until the enumerator
// is fully initialized, and then will return a pointer to the enumerator
// (unless an error occurs).
//
// If the supplied callback is non-NULL, the call will immediately return NULL,
// and will arrange for the callback to be invoked with the enumerator once the
// enumerator becomes available (calling it with a NULL enumerator on error).  The
// library guarantees to invoke the callback exactly once.
//
// If the library returns an enumerator, there will be at least one
// configured device available; otherwise, the call will fail with
// VAI_STATUS_NOT_FOUND.
//
// The library may invoke the callback synchronously if the enumerator is
// immediately available or in error conditions.
PLAIDML_API plaidml_device_enumerator* plaidml_alloc_device_enumerator(
    vai_ctx* ctx, void (*callback)(void* arg, plaidml_device_enumerator* enumerator), void* arg);

// Allocates a device enumerator and initializes it using the supplied
// configuration. Otherwise identical to the version that uses system
// config.
PLAIDML_API plaidml_device_enumerator* plaidml_alloc_device_enumerator_with_config(
    vai_ctx* ctx, const char* configuration, void (*callback)(void* arg, plaidml_device_enumerator* enumerator),
    void* arg);

// Frees a device enumerator.  After this call, the enumerator should not be
// used for any subsequent calls.  The enumerator may share resources with
// other objects (such as devices, buffers, and functions); those resources
// will only be released when they are no longer needed by those other objects.
// Freeing a NULL enumerator is a no-op.
PLAIDML_API void plaidml_free_device_enumerator(plaidml_device_enumerator* enumerator);

// Gets the configuration file that was used to initialize devices.
PLAIDML_API const char* plaidml_get_enumerator_config_source(plaidml_device_enumerator* enumerator);

// Gets the number of device valid or invalid configurations available.
PLAIDML_API size_t plaidml_get_devconf_count(vai_ctx* ctx, plaidml_device_enumerator* enumerator, bool valid_devices);

// Gets one device configuration from a device enumerator.  The lifetime of the
// device configuration is bounded by the device enumerator; there's no need to
// separately free the configuration.  If the requested configuration index is
// out of range, or if the enumerator is NULL, this call will return NULL.
PLAIDML_API plaidml_devconf* plaidml_get_devconf(vai_ctx* ctx, plaidml_device_enumerator* enumerator, size_t index);

// Same as above, only returns invalid devices
PLAIDML_API plaidml_devconf* plaidml_get_invalid_devconf(vai_ctx* ctx, plaidml_device_enumerator* enumerator,
                                                         size_t index);

// PlaidML buffers are used to create bindings between actual data and the data
// elements in PlaidML programs.
#ifdef __cplusplus
struct plaidml_buffer;
#else
typedef struct plaidml_buffer plaidml_buffer;
#endif  // __cplusplus

// PlaidML mappings are used to view and manipulate the contents of buffers.
#ifdef __cplusplus
struct plaidml_mapping;
#else
typedef struct plaidml_mapping plaidml_mapping;
#endif  // __cplusplus

// Allocates a buffer of the supplied raw memory size, or returns NULL if the
// library cannot allocate sufficient memory, or if the supplied device is
// NULL.
PLAIDML_API plaidml_buffer* plaidml_alloc_buffer(vai_ctx* ctx, plaidml_device* device, uint64_t size);

// Frees a buffer.  After this call, the buffer should not be used for any
// subsequent calls.  Freeing a NULL buffer is a no-op.
PLAIDML_API void plaidml_free_buffer(plaidml_buffer* buffer);

// Maps a buffer's current contents into memory.
//
// If the supplied callback is NULL, the call will block until the buffer is
// available, and then will return a pointer to a mapping for the buffer
// (unless an error occurs).
//
// If the supplied callback is non-NULL, the call will immediately return NULL,
// and will arrange for the callback to be invoked with the mapping once the
// buffer's data becomes available (calling it with a NULL mapping on error).
//
// The library may invoke the callback synchronously if the buffer's data is
// already available or in error conditions.
//
// A NULL buffer may be supplied; this will always result in a NULL address,
// and an out-of-memory error in the current thread's thread-local storage.
PLAIDML_API plaidml_mapping* plaidml_map_buffer_current(plaidml_buffer* buffer,
                                                        void (*callback)(void* arg, plaidml_mapping* mapping),
                                                        void* arg);

// Maps a buffer into memory, possibly discarding its current contents.
//
// The implementation may preserve or discard the buffer's contents when
// constructing the mapping; callers should not assume that the buffer has been
// initialized.
//
// The implementation may construct a non-coherent mapping: reads from the
// buffer may return arbitrary values, even after writing the same memory from
// the same processor.
//
// A NULL buffer may be supplied; this will always result in a NULL address,
// and an out-of-memory error in the current thread's thread-local storage.
PLAIDML_API plaidml_mapping* plaidml_map_buffer_discard(vai_ctx* ctx, plaidml_buffer* buffer);

// Gets the base address of a mapping's mapped memory region.  If the mapping
// has been written back to the buffer, this call will return NULL.  A NULL
// mapping may be supplied; this will always return NULL.
PLAIDML_API char* plaidml_get_mapping_base(vai_ctx* ctx, plaidml_mapping* mapping);

// Gets the size of a mapping's mapped memory region.  If the mapping has been
// written back to the buffer, this call will return 0.  A NULL mapping may be
// supplied; this will always return 0.
PLAIDML_API size_t plaidml_get_mapping_size(vai_ctx* ctx, plaidml_mapping* mapping);

// Synchronizes a mapping with its backing store (if required by the underlying
// device implementation), possibly removing the mapping from the current
// virtual address space.
//
// After this call, callers must not access the mapping's previous virtual
// memory region.
//
// Callers are allowed to free the mapping and use the buffer as a program input
// immediately after this call.
//
// A NULL mapping may be supplied; this will always result in a false result
// and an out-of-memory error in the current thread's thread-local storage.
PLAIDML_API bool plaidml_writeback_mapping(vai_ctx* ctx, plaidml_mapping* mapping);

// Removes a mapping that's no longer required by the caller.
//
// After this call, callers must not access the mapping's previous virtual
// memory region.
//
// Freeing a NULL mapping is a no-op.
PLAIDML_API void plaidml_free_mapping(plaidml_mapping* mapping);

// PlaidML shapes describe the layout of the data within a buffer as observed by a
// program.
#ifdef __cplusplus
struct plaidml_shape;
#else
typedef struct plaidml_shape plaidml_shape;
#endif  // __cplusplus

// Set the default datatype for floating-point computations.
PLAIDML_API void plaidml_set_floatx(plaidml_datatype datatype);

// Allocates a shape, or returns NULL if the library cannot allocate sufficient
// memory.  Note that shapes must have dimensions added before use.
PLAIDML_API plaidml_shape* plaidml_alloc_shape(vai_ctx* ctx, plaidml_datatype datatype);

// Frees a shape.  After this call, the shape should not be used for any
// subsequent calls.  Freeing a NULL shape is a no-op.
PLAIDML_API void plaidml_free_shape(plaidml_shape* shape);

// Sets a shape's offset, in elements, from the beginning of the data.
PLAIDML_API bool plaidml_set_shape_offset(vai_ctx* ctx, plaidml_shape* shape, uint64_t offset_in_elements);

// Set a shape's layout
PLAIDML_API bool plaidml_shape_set_layout(vai_ctx* ctx, plaidml_shape* shape, const char* layout);

// Adds a dimension to a shape.  Dimension sizes and strides are measured in
// elements of the shape's datatype, not by local buffer byte counts.
PLAIDML_API bool plaidml_add_dimension(vai_ctx* ctx, plaidml_shape* shape, uint64_t size_in_elements,
                                       int64_t stride_in_elements);

// Gets a shape's type.
PLAIDML_API plaidml_datatype plaidml_get_shape_type(plaidml_shape* shape);

// Gets a shape's offset.
PLAIDML_API uint64_t plaidml_get_shape_offset(plaidml_shape* shape);

// Get the number of dimensions for a shape.
PLAIDML_API size_t plaidml_get_shape_dimension_count(plaidml_shape* shape);

// Gets the size in elements for a given shape dimension.
// If the dimension is out of range, zero will be returned.
PLAIDML_API uint64_t plaidml_get_shape_dimension_size(plaidml_shape* shape, size_t dim);

// Gets the stride in elements for a given shape dimension.
// If the dimension is out of range, zero will be returned.
PLAIDML_API int64_t plaidml_get_shape_dimension_stride(plaidml_shape* shape, size_t dim);

// Gets the byte size required for a buffer to hold the given shape.
PLAIDML_API uint64_t plaidml_get_shape_buffer_size(plaidml_shape* shape);

// Gets the underlying element count described by the given shape.
PLAIDML_API uint64_t plaidml_get_shape_element_count(plaidml_shape* shape);

// A PlaidML function defines a transformation from some set of inputs to
// some set of outputs.
#ifdef __cplusplus
struct plaidml_function;
#else
typedef struct plaidml_function plaidml_function;
#endif  // __cplusplus

// Frees a function.  After this call, the function should not be used for any
// subsequent calls.  Freeing a NULL function is a no-op.
PLAIDML_API void plaidml_free_function(plaidml_function* function);

// Return the number of inputs to a function, 0 if function is NULL
PLAIDML_API size_t plaidml_get_function_input_count(plaidml_function* function);

// Return the name of input i for a function, or NULL if function is NULL or out of bounds
PLAIDML_API const char* plaidml_get_function_input(plaidml_function* function, size_t i);

// Return the number of outpus from function, 0 if function is NULL
PLAIDML_API size_t plaidml_get_function_output_count(plaidml_function* function);

// Return the name of output i for a function, or NULL if function is NULL or out of bounds
PLAIDML_API const char* plaidml_get_function_output(plaidml_function* function, size_t i);

// A PlaidML var is an input to or an output from a PlaidML function.
#ifdef __cplusplus
struct plaidml_var;
#else
typedef struct plaidml_var plaidml_var;
#endif  // __cplusplus

// Frees a var.  After this call, the var should not be used for any
// subsequent calls.  Freeing a NULL var is a no-op.
PLAIDML_API void plaidml_free_var(plaidml_var* var);

// Allocate a placeholder var.
//
// A placeholder can be used during function application: used as the output of
// one function application and the the input of another function application,
// the placeholder defines an information flow between the functions.  A
// placeholder can also during function composition: the placeholder can be
// bound to the inputs or outputs of the composed function.
//
// num_dimensions specifies the placeholder dimension count: PlaidML functions are
// polymorphic with respect to tensor sizes and datatypes, but not to the
// actual dimension count of their inputs and outputs, so placeholders need to
// indicate the number of dimensions of the variables they will eventually be
// bound to.  For scalar placeholders, specify zero for the dimension count.
PLAIDML_API plaidml_var* plaidml_alloc_placeholder(size_t num_dimensions);

// Allocates a var representing a signed integer constant.
PLAIDML_API plaidml_var* plaidml_alloc_int64(int64_t value);

// Allocates a var representing a floating point constant.
PLAIDML_API plaidml_var* plaidml_alloc_real(double value);

// Allocates a var representing a tensor, bound to the given shape and buffer.
PLAIDML_API plaidml_var* plaidml_alloc_tensor(vai_ctx* ctx, plaidml_buffer* buffer, plaidml_shape* shape);

// Attaches quantization parameters to a weights tensor
PLAIDML_API bool plaidml_tensor_attach_qparams(plaidml_var* tensor, plaidml_var* qparams);

// Builds a function from the supplied code written in the PlaidML operation
// description language.  If 'id' is not NULL, attach the id to the function
// for tracking purposes.
PLAIDML_API plaidml_function* plaidml_build_coded_function(const char* code, const char* id);

// TODO: Make more general method to serialize things.

// Load a function (possibly with bound tensors) from a file
PLAIDML_API plaidml_function* plaidml_load_function(vai_ctx* ctx, plaidml_device* dev, const char* filename);

// Store a function (possibly with bound tensors) from to a file
PLAIDML_API bool plaidml_save_function(plaidml_function* func, const char* filename);

// Predeclare applier
// A PlaidML applier describes the application of a PlaidML function to some
// particular set of inputs, yielding some particular set of outputs.  (For
// example, you can think of "+" as a function; applying it to "2" and "3"
// yields a particular output, "5".)
#ifdef __cplusplus
struct plaidml_applier;
#else
typedef struct plaidml_applier plaidml_applier;
#endif  // __cplusplus

// A PlaidML composer builds a new function out of a set of vars, where the values
// of the output vars have been previously defined (by using an applier),
// either in terms of placeholders (which become the new function inputs), or
// in terms of mutable tensors (which will be mutated each time the function is
// run).
#ifdef __cplusplus
struct plaidml_composer;
#else
typedef struct plaidml_composer plaidml_composer;
#endif  // __cplusplus

// Allocates a composer, or returns NULL if the library cannot allocate sufficient memory.
PLAIDML_API plaidml_composer* plaidml_alloc_composer();

// Binds a placeholder var to a named input of a composed function.
PLAIDML_API bool plaidml_add_composer_input(plaidml_composer* composer, const char* name, plaidml_var* var);

// Binds a computed value var to a named output of a composed function.
PLAIDML_API bool plaidml_add_composer_output(plaidml_composer* composer, const char* name, plaidml_var* var);

// Adds a dependency to the composed function.  Any updates induced by the function
// application will be updates of the newly generated function (in addition to any
// explicit updates, which will superseed them).
PLAIDML_API bool plaidml_add_composer_dependency(plaidml_composer* composer, plaidml_applier* must_run_before);

// Adds a tensor update to a composed function.  This allows the composed
// function to have externally visible side effects when run: the source tensor
// (which should be a placeholder bound to an output of some function) will be
// assigned to the destination tensor (either a placeholder or a tensor var)
// each time the composed function is run.
PLAIDML_API bool plaidml_add_composer_update(plaidml_composer* composer, plaidml_var* dest_tensor,
                                             plaidml_var* src_tensor);

// Builds the function described by the composer.  This should be called at
// most once per composer; after this call, the only valid operation on the
// composer is plaidml_free_composer().
PLAIDML_API plaidml_function* plaidml_build_composed_function(plaidml_composer* composer);

// Frees a composer.  After this call, the composer should not be used for any
// subsequent calls.  Freeing a NULL composer is a no-op.
PLAIDML_API void plaidml_free_composer(plaidml_composer* composer);

// Allocates an applier describing the application of the given function to some
// number of inputs, or returns NULL if the library cannot allocate sufficient memory.
PLAIDML_API plaidml_applier* plaidml_alloc_applier(plaidml_function* function);

// Adds a dependency to the applied function.  This is used to sequence tensor
// updates: if a sub-function of the applied function uses a mutable tensor as
// an input, the value it will observe for that tensor will be the value after
// the indicated function has run (presumably updating the tensor).  In addition
// the new function application will carry the updates forward.
PLAIDML_API bool plaidml_apply_add_dependency(plaidml_applier* applier, plaidml_applier* must_run_before);

// Adds a named input to a function application.  Note that the input variable
// is not consumed; the caller remains responsible for calling plaidml_free_var()
// when the supplied var is no longer needed.
PLAIDML_API bool plaidml_apply_add_input(plaidml_applier* applier, const char* name, plaidml_var* var);

// Allocates a var corresponding to the output of a function application.  The
// caller is responsible for calling plaidml_free_var() on the result when the
// variable is no longer needed.
//
// At the time when the output is allocated, all inputs to the function
// application must already be added (either as concrete values or as
// placeholders).
PLAIDML_API plaidml_var* plaidml_apply_alloc_output(plaidml_applier* applier, const char* name);

// Frees an applier.  After this call, the applier should not be used for any
// subsequent calls.  Freeing a NULL applier is a no-op.
PLAIDML_API void plaidml_free_applier(plaidml_applier* applier);

// A PlaidML invoker provides a mechanism for scheduling runs of a
// PlaidML function.
//
// The function need not be completely bound when supplied to the
// invoker; the invoker may be mutated to set the function inputs and
// outputs.  The input and output bindings must be fully specified,
// and must be dimensionally consistent with each other, at the time
// the invoker is used to invoke the supplied function.
//
// Invokers are not threadsafe, and they are stateful; callers are
// advised to synchronize concurrent access from the time the
// invoker's inputs are set through to when the function has completed
// running.

#ifdef __cplusplus
struct plaidml_invoker;
#else
typedef struct plaidml_invoker plaidml_invoker;
#endif  // __cplusplus

// Allocates an invoker for the supplied function, or returns NULL if
// the library cannot allocate sufficient memory, or if the supplied
// context or function is NULL.
PLAIDML_API plaidml_invoker* plaidml_alloc_invoker(vai_ctx* ctx, plaidml_function* function);

// Frees an invoker.  After this call, the invoker should not be used for any
// subsequent calls.  Freeing a NULL invoker is a no-op.
PLAIDML_API void plaidml_free_invoker(plaidml_invoker* invoker);

// Sets a named input for an invocation.  The variable must be NULL or
// a concrete object; placeholders are not permitted.  Note that the
// input variable is not consumed; the caller remains responsible for
// calling plaidml_free_var() when the supplied var is no longer
// needed.
PLAIDML_API bool plaidml_set_invoker_input(plaidml_invoker* invoker, const char* name, plaidml_var* var);

// Allocates a shape corresponding to the output of an invocation.
// The caller is responsible for calling plaidml_free_shape() on the
// result when the shape is no longer needed.
//
// At the time when the shape is allocated, all inputs to the
// invocation must already be set to concrete values that are
// consistent in size.
PLAIDML_API plaidml_shape* plaidml_alloc_invoker_output_shape(plaidml_invoker* invoker, const char* name);

// Sets a named output for an invocation.  The variable must be NULL
// or a concrete tensor; placeholders are not permitted.  Note that
// the output variable is not consumed; the caller remains responsible
// for calling plaidml_free_var() when the supplied var is no longer
// needed.
PLAIDML_API bool plaidml_set_invoker_output(plaidml_invoker* invoker, const char* name, plaidml_var* var);

// PlaidML Stripe file formats.
typedef enum {
  PLAIDML_FILE_FORMAT_TILE = 1,
  PLAIDML_FILE_FORMAT_STRIPE_HUMAN = 2,
  PLAIDML_FILE_FORMAT_STRIPE_PROTOTXT = 3,
  PLAIDML_FILE_FORMAT_STRIPE_BINARY = 4,
} plaidml_file_format;

// Mark a functions inputs as 'const', that is, subject to constant folding
// and other operations that assume they will no longer be changed
PLAIDML_API bool plaidml_set_invoker_const(plaidml_invoker* invoker);

// Serializes an invoker to a file.  All inputs to the invoker must
// already be set to concrete values that are consistent in size.
PLAIDML_API bool plaidml_save_invoker(plaidml_invoker* invoker, const char* filename, plaidml_file_format format);

// A PlaidML invocation describes one particular run of a function.
#ifdef __cplusplus
struct plaidml_invocation;
#else
typedef struct plaidml_invocation plaidml_invocation;
#endif  // __cplusplus

// Schedules a run of an invoker's function with the invoker's current
// input and output bindings.
//
// The invocation must be fully specified: all function inputs and
// outputs must have corresponding inputs and outputs set in the
// invoker.  Furthermore, all inputs and outputs must be consistently
// sized relative to each other and the function.
//
// All buffers used to back tensors that will be updated by the
// invocation should already be unmapped.  When this call returns,
// those buffers logically contain the updated values, and may be
// remapped; remapping requests will complete once the underlying
// invocation is complete.
//
// Note that this call may return before the computation described by
// the function has actually completed; the computation is scheduled,
// not complete.  Errors that occur asynchronously will be reported
// when the buffers updated by running the function are remapped.
//
// Once this call returns, the invoker's inputs and outputs may be set
// by the caller, and the invoker may be used for another run of the
// invoker's function, even if the first run has not yet completed.
PLAIDML_API plaidml_invocation* plaidml_schedule_invocation(vai_ctx* ctx, plaidml_invoker* invoker);

// Frees an invocation.  After this call, the invocation should not be
// used for any subsequent calls.  Freeing a NULL invocation is a no-op.
PLAIDML_API void plaidml_free_invocation(plaidml_invocation* invocation);

// A PlaidML gradient computes gradient data for a given scalar.
#ifdef __cplusplus
struct plaidml_gradient;
#else
typedef struct plaidml_gradient plaidml_gradient;
#endif  // __cplusplus

// Allocate and returns a gradient computer for a given scalar or NULL if error
PLAIDML_API plaidml_gradient* plaidml_alloc_gradient(plaidml_var* var);

// Frees a gradient computer.  After this call, the context should not be used for
// any subsequent calls.  Freeing a NULL context is a no-op.
PLAIDML_API void plaidml_free_gradient(plaidml_gradient* grad);

// Determines the gradient of with respect to some value
PLAIDML_API plaidml_var* plaidml_compute_grad_wrt(plaidml_gradient* grad, plaidml_var* wrt);

#ifdef __cplusplus
}  // extern "C"

namespace std {

template <>
struct default_delete<::plaidml_device> {
  void operator()(::plaidml_device* device) const noexcept { ::plaidml_close_device(device); }
};

template <>
struct default_delete<::plaidml_device_enumerator> {
  void operator()(::plaidml_device_enumerator* enumerator) const noexcept {
    ::plaidml_free_device_enumerator(enumerator);
  }
};

template <>
struct default_delete<::plaidml_buffer> {
  void operator()(::plaidml_buffer* buffer) const noexcept { ::plaidml_free_buffer(buffer); }
};

template <>
struct default_delete<::plaidml_mapping> {
  void operator()(::plaidml_mapping* mapping) const noexcept { ::plaidml_free_mapping(mapping); }
};

template <>
struct default_delete<::plaidml_shape> {
  void operator()(::plaidml_shape* shape) const noexcept { ::plaidml_free_shape(shape); }
};

template <>
struct default_delete<::plaidml_function> {
  void operator()(::plaidml_function* function) const noexcept { ::plaidml_free_function(function); }
};

template <>
struct default_delete<::plaidml_var> {
  void operator()(::plaidml_var* var) const noexcept { ::plaidml_free_var(var); }
};

template <>
struct default_delete<::plaidml_composer> {
  void operator()(::plaidml_composer* composer) const noexcept { ::plaidml_free_composer(composer); }
};

template <>
struct default_delete<::plaidml_applier> {
  void operator()(::plaidml_applier* applier) const noexcept { ::plaidml_free_applier(applier); }
};

template <>
struct default_delete<::plaidml_invoker> {
  void operator()(::plaidml_invoker* invoker) const noexcept { ::plaidml_free_invoker(invoker); }
};

template <>
struct default_delete<::plaidml_invocation> {
  void operator()(::plaidml_invocation* invocation) const noexcept { ::plaidml_free_invocation(invocation); }
};

template <>
struct default_delete<::plaidml_gradient> {
  void operator()(::plaidml_gradient* gradient) const noexcept { ::plaidml_free_gradient(gradient); }
};

}  // namespace std

#endif  // __cplusplus
