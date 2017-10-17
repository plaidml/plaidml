load("//bzl:plaidml.bzl", "plaidml_grammar", "plaidml_cc_library", "plaidml_cc_test", "plaidml_proto_library")

plaidml_proto_library(
    name = "proto",
    srcs = [
        "lang.proto",
    ],
    visibility = ["//visibility:public"],
)

plaidml_cc_library(
    name = "lang",
    srcs = [
        "bound.cc",
        "bound.h",
        "builtins.cc",
        "builtins.h",
        "compile.cc",
        "compose.cc",
        "defract.cc",
        "defract.h",
        "emitc.cc",
        "emitmetal.cc",
        "flat.cc",
        "flat.h",
        "fnv1a64.h",
        "fpconv.cc",
        "fpconv.h",
        "gen_contract.cc",
        "gen_contract.h",
        "gen_special.cc",
        "gen_special.h",
        "gen_zero.cc",
        "gen_zero.h",
        "generate.cc",
        "gid.cc",
        "gid.h",
        "loop.cc",
        "loop.h",
        "mutil.h",
        "ops.cc",
        "out_plan.cc",
        "out_plan.h",
        "parser.cc",
        "read_plan.cc",
        "read_plan.h",
        "reduce.cc",
        "reduce.h",
        "replace.cc",
        "replace.h",
        "semprinter.cc",
        "semtree.cc",
        "sym_poly.cc",
        "sym_poly.h",
        "symbolic.cc",
        "tile_cache.cc",
        "tile_cache.h",
        "tile_opt.cc",
        "tile_opt.h",
        "type.cc",
        "usedef.cc",
        "usedef.h",
        ":grammar",
    ],
    hdrs = [
        "compile.h",
        "compose.h",
        "emitc.h",
        "emitmetal.h",
        "generate.h",
        "ops.h",
        "parser.h",
        "sembuilder.h",
        "semprinter.h",
        "semtree.h",
        "shape.h",
        "symbolic.h",
        "type.h",
    ],
    copts = select({
        "//bzl:x64_windows": [],
        "//conditions:default": [
            "-Wno-format",
            "-Wno-sign-compare",
            "-Wno-macro-redefined",
            "-Wno-unused-variable",
            "-Wno-unused-function",
            "-Wno-deprecated-register",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":math",
        ":proto_cc",
        "//base/util:util_cc",
        "//external:boost",
        "//tile/lang/bilp",
    ],
)

plaidml_cc_test(
    name = "test",
    srcs = [
        "lang_test.cc",
        "math_test.cc",
        "sim_test.cc",
        "simulate.h",
        "test.cc",
    ],
    deps = [
        ":lang",
    ],
)

plaidml_cc_library(
    name = "math",
    srcs = [
        "basis.cc",
        "bignum.cc",
        "matrix.cc",
        "polynomial.cc",
    ],
    hdrs = [
        "basis.h",
        "bignum.h",
        "matrix.h",
        "polynomial.h",
    ],
    copts = select({
        "//bzl:x64_windows": [],
        "//conditions:default": ["-Wno-error"],
    }),
    visibility = ["//tile/lang/bilp:__pkg__"],
    deps = [
        "//base/util:util_cc",
        "//external:boost",
    ],
)

plaidml_grammar(
    name = "grammar",
    outs = [
        "parser.y.cc",
        "parser.y.h",
        "parser_lex.cc",
        "parser_lex.h",
    ],
    bison_src = "tile.y",
    flex_src = "tile.l",
)

plaidml_cc_test(
    name = "gid_test",
    srcs = [
        "gid_test.cc",
    ],
    deps = [
        ":lang",
        "//base/util:util_cc",
        "//external:gtest",
    ],
)
