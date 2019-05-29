import os
import platform
import sys

import cffi

EDSL_CDEF = '''
typedef enum {
    PLAIDML_DATA_INVALID = 0,
    PLAIDML_DATA_BOOLEAN = 0x02,
    PLAIDML_DATA_INT8 = 0x10,
    PLAIDML_DATA_INT16 = 0x11,
    PLAIDML_DATA_INT32 = 0x12,
    PLAIDML_DATA_INT64 = 0x13,
    PLAIDML_DATA_INT128 = 0x14,
    PLAIDML_DATA_UINT8 = 0x20,
    PLAIDML_DATA_UINT16 = 0x21,
    PLAIDML_DATA_UINT32 = 0x22,
    PLAIDML_DATA_UINT64 = 0x23,
    PLAIDML_DATA_FLOAT16 = 0x31,
    PLAIDML_DATA_FLOAT32 = 0x32,
    PLAIDML_DATA_FLOAT64 = 0x33,
    PLAIDML_DATA_PRNG = 0x40,
} plaidml_datatype;

typedef enum {
    TILE_POLY_OP_NEG,
    TILE_POLY_OP_ADD,
    TILE_POLY_OP_SUB,
    TILE_POLY_OP_MUL,
    TILE_POLY_OP_DIV,
} tile_poly_op;

typedef enum {
    TILE_AGG_OP_NONE,
    TILE_AGG_OP_SUM,
    TILE_AGG_OP_MAX,
    TILE_AGG_OP_MIN,
    TILE_AGG_OP_PROD,
    TILE_AGG_OP_ASSIGN
} tile_agg_op;

typedef enum {
    TILE_COMBO_OP_NONE,
    TILE_COMBO_OP_MUL,
    TILE_COMBO_OP_ADD,
    TILE_COMBO_OP_EQ,
    TILE_COMBO_OP_COND,
} tile_combo_op;

typedef struct {
    // opaque
} tile_string;

typedef struct {
    // opaque
} tile_shape;

typedef struct {
    // opaque
} tile_expr;

typedef struct {
    // opaque
} tile_poly_expr;

typedef struct {
    size_t code;
    tile_string* msg;
} tile_error;

typedef struct {
    tile_error err;
} tile_ret_void;

typedef struct {
    tile_error err;
    size_t ret;
} tile_ret_uint;

typedef struct {
    tile_error err;
    int64_t ret;
} tile_ret_i64;

typedef struct {
    tile_error err;
    uint64_t ret;
} tile_ret_u64;

typedef struct {
    tile_error err;
    tile_string* ret;
} tile_ret_string;

typedef struct {
    tile_error err;
    tile_shape* ret;
} tile_ret_shape;

typedef struct {
    tile_error err;
    tile_expr* ret;
} tile_ret_expr;

typedef struct {
    tile_error err;
    tile_poly_expr* ret;
} tile_ret_poly_expr;

const char* tile_string_ptr(tile_string* str);
void tile_string_free(const char* ptr);

tile_ret_shape tile_shape_alloc(plaidml_datatype datatype);
tile_ret_string tile_shape_repr(tile_shape* shape);
tile_ret_void tile_shape_free(tile_shape* shape);
tile_ret_void tile_shape_add_dimension(tile_shape* shape, uint64_t size, int64_t stride);
tile_ret_uint tile_shape_get_rank(tile_shape* shape);
tile_ret_u64 tile_shape_get_dimension_size(tile_shape* shape, size_t dim);
tile_ret_i64 tile_shape_get_dimension_stride(tile_shape* shape, size_t dim);

tile_ret_void tile_expr_free(tile_expr* expr);
tile_ret_string tile_expr_repr(tile_expr* expr);
tile_ret_expr tile_expr_param(tile_shape* shape, const char* name);
tile_ret_expr tile_expr_int(int64_t value);
tile_ret_expr tile_expr_float(double value);
tile_ret_expr tile_expr_call(const char* fn, size_t nargs, tile_expr** args);
tile_ret_expr tile_expr_contraction(
    tile_agg_op agg_op,
    tile_combo_op combo_op,
    tile_expr* output,
    size_t ninputs,
    tile_expr** inputs,
    size_t nconstraints,
    tile_expr** constraints,
    bool no_defract,
    tile_expr* use_default
);
tile_ret_expr tile_expr_tensor_spec(tile_expr* ref, size_t rank, tile_poly_expr** idxs, size_t* sizes);
tile_ret_expr tile_expr_constraint(tile_poly_expr* lhs, size_t rhs);
tile_ret_expr tile_expr_contraction_set_no_defract(tile_expr* expr, bool no_defract);
tile_ret_expr tile_expr_contraction_set_use_default(tile_expr* expr, tile_expr* use_default);
tile_ret_shape tile_expr_evaluate_shape(tile_expr* expr);

tile_ret_void tile_poly_expr_free(tile_poly_expr* expr);
tile_ret_string tile_poly_expr_repr(tile_poly_expr* expr);
tile_ret_poly_expr tile_poly_expr_index(const void* ptr, const char* name);
tile_ret_poly_expr tile_poly_expr_literal(int64_t value);
tile_ret_poly_expr tile_poly_expr_op(tile_poly_op op, size_t nargs, tile_poly_expr** args);
'''


def __load_library():
    ffi = cffi.FFI()
    ffi.cdef(EDSL_CDEF)

    native_path = os.getenv('PLAIDML_NATIVE_PATH')
    if native_path:
        return ffi, ffi.dlopen(native_path)

    if platform.system() == 'Windows':
        libname = 'plaidml.dll'
        libdirs = ['Library', 'bin']
    if platform.system() == 'Darwin':
        libname = 'libplaidml.dylib'
        libdirs = ['lib']
    else:
        libname = 'libplaidml.so'
        libdirs = ['lib']
    # When running under Bazel with an uninstalled native
    # library, we'll be able to find correct native library to
    # use in this script's directory -- if it exists, we
    # should use it.  Note that when installed, the native
    # library will never be in the script's directory, so this
    # shouldn't find anything.
    self_path = os.path.abspath(__file__)
    self_dir = os.path.dirname(self_path)
    libpath = os.path.join(self_dir, libname)
    try:
        return ffi, ffi.dlopen(libpath)
    except:
        # If we're unable to load the PlaidML library from the
        # script's directory, we fall back on the system
        # installed version.  Note that if the system
        # installed version is missing (e.g. if PlaidML is
        # mis-installed), this will fail, and report the
        # correct path for diagnosis.
        libpath = os.path.join(sys.exec_prefix, *libdirs, libname)
        return ffi, ffi.dlopen(libpath)


ffi, lib = __load_library()


def ffi_call(func, *args):
    """Calls ffi function and does some error handling."""
    ret = func(*args)
    if ret.err.code:
        raise TileError(ret.err)
    if hasattr(ret, 'ret'):
        return ret.ret


def decode_str(ptr):
    if ptr:
        try:
            return ffi.string(lib.tile_string_ptr(ptr)).decode()
        finally:
            lib.tile_string_free(ptr)
    return None


class TileError(Exception):

    def __init__(self, err):
        Exception.__init__(self)
        self.code = err.code
        self.msg = decode_str(err.msg)
        # self.backtrace = decode_str(err.backtrace)

    def __str__(self):
        # if self.backtrace:
        #     return '{}\n\n{}'.format(self.message, self.backtrace)
        return self.msg


class NativeObject(object):
    __ffi_obj__ = None
    __ffi_del__ = None
    __ffi_repr__ = None

    def __init__(self, ffi_obj):
        self.__ffi_obj__ = ffi_obj

    def __del__(self):
        if self.__ffi_obj__ and self.__ffi_del__:
            self._methodcall(self.__ffi_del__)

    def __repr__(self):
        if self.__ffi_obj__ and self.__ffi_repr__:
            return decode_str(self._methodcall(self.__ffi_repr__))
        return super(NativeObject, self).__repr__()

    def _methodcall(self, func, *args):
        return ffi_call(func, self.__ffi_obj__, *args)

    def as_ptr(self, release=False):
        if self.__ffi_obj__ is None:
            return ffi.NULL
        ret = self.__ffi_obj__
        if release:
            self.__ffi_obj__ = None
        return ret

    def set_ptr(self, ffi_obj):
        self.__ffi_obj__ = ffi_obj

    def take_ptr(self, obj):
        self.__ffi_obj__ = obj.as_ptr(True)
