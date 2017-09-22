# -*- mode: python; -*- PYTHON-PREPROCESSING-REQUIRED

def with_protobuf(root=""):
  native.new_git_repository(
    name="protobuf", # named thusly to appease tensorflow
    remote="https://github.com/google/protobuf",
    tag="v3.2.0",
    build_file = root + "//bzl:protobuf.BUILD")

  native.bind(name="protoc", actual="@protobuf//:protoc")
  native.bind(name="protobuf_clib", actual="@protobuf//:protoc_lib",)
  native.bind(name="protobuf_compiler", actual="@protobuf//:protoc_lib",)
  native.bind(name="protobuf_", actual="@protobuf//:protobuf",)
  native.bind(name="protobuf_python", actual="@protobuf//:protobuf_python",)
  native.bind(name="protobuf_python_headers", actual="@protobuf//base/util/python:python_headers",)
  native.bind(name="protobuf_cc_wkt_protos", actual="@protobuf//:cc_wkt_protos")
  native.bind(name="protobuf_cc_wkt_protos_genproto", actual="@protobuf//:cc_wkt_protos_genproto")
  native.bind(name="protobuf_python_genproto", actual="@protobuf//:protobuf_python_genproto")
  native.bind(name="protobuf_lib_ios", actual="@protobuf//:protobuf_ios")

#
# Cribbed from protobuf to provide py gRPC support
#/

def _GetPath(ctx, path):
  if ctx.label.workspace_root:
    return ctx.label.workspace_root + '/' + path
  else:
    return path

def _GenDir(ctx):
  if not ctx.attr.includes:
    return ctx.label.workspace_root
  if not ctx.attr.includes[0]:
    return _GetPath(ctx, ctx.label.package)
  if not ctx.label.package:
    return _GetPath(ctx, ctx.attr.includes[0])
  return _GetPath(ctx, ctx.label.package + '/' + ctx.attr.includes[0])

def _CcHdrs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
  return ret

def _CcSrcs(srcs, use_grpc_plugin=False):
  ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
  if use_grpc_plugin:
    ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
  return ret

def _PyOuts(srcs):
  return [s[:-len(".proto")] + "_pb2.py" for s in srcs]

def _RelativeOutputPath(path, include):
  if include == None:
    return path

  if not path.startswith(include):
    fail("Include path %s isn't part of the path %s." % (include, path))

  if include and include[-1] != '/':
    include = include + '/'

  path = path[len(include):]

  if not path.startswith(PACKAGE_NAME):
    fail("The package %s is not within the path %s" % (PACKAGE_NAME, path))

  if not PACKAGE_NAME:
    return path

  return path[len(PACKAGE_NAME) + 1:]

def _proto_gen_impl(ctx):
  """General implementation for generating protos"""
  srcs = ctx.files.srcs
  deps = []
  deps += ctx.files.srcs
  gen_dir = _GenDir(ctx)
  if gen_dir:
    import_flags = ["-I" + gen_dir]
  else:
    import_flags = ["-I."]

  for dep in ctx.attr.deps:
    import_flags += dep.proto.import_flags
    deps += dep.proto.deps

  args = []
  if ctx.attr.gen_cc:
    args += ["--cpp_out=" + ctx.var["GENDIR"] + "/" + gen_dir]
    args += ["--plugin=protoc-gen-cc11=" + ctx.executable.cc11_plugin.path]
    args += ["--cc11_out=" + ctx.var["GENDIR"] + "/" + gen_dir]
    deps += [ctx.executable.cc11_plugin]
  if ctx.attr.gen_py:
    args += ["--python_out=" + ctx.var["GENDIR"] + "/" + gen_dir]

  if ctx.executable.grpc_plugin:
    args += ["--plugin=protoc-gen-grpc=" + ctx.executable.grpc_plugin.path]
    args += ["--grpc_out=" + ctx.var["GENDIR"] + "/" + gen_dir]
    deps += [ctx.executable.grpc_plugin]

  if args:
    ctx.action(inputs=srcs + deps,
               outputs=ctx.outputs.outs,
               arguments=args + import_flags + [s.path for s in srcs],
               executable=ctx.executable.protoc,)

  return struct(proto=struct(srcs=srcs, import_flags=import_flags, deps=deps,),)

_proto_gen = rule(
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = ["proto"]),
        "includes": attr.string_list(),
        "protoc": attr.label(
            cfg = "host",
            executable = True,
            single_file = True,
            mandatory = True,
        ),
        "grpc_plugin": attr.label(
            cfg = "host",
            executable = True,
            single_file = True,
        ),
        "cc11_plugin": attr.label(
            cfg = "host",
            executable = True,
            single_file = True,
        ),
        "gen_cc": attr.bool(),
        "gen_py": attr.bool(),
        "outs": attr.output_list(),
    },
    output_to_genfiles = True,
    implementation = _proto_gen_impl,
)

def cc_proto_library(name,
                     srcs=[],
                     deps=[],
                     cc_libs=[],
                     include=None,
                     protoc="//external:protoc",
                     has_services=False,
                     default_runtime="//external:protobuf_",
                     copts=["--std=c++1y"],
                     **kargs):
  """Bazel rule to create a C++ protobuf library from proto source files
    NOTE: the rule is only an internal workaround to generate protos. The
    interface may change and the rule may be removed when bazel has introduced
    the native rule.
    Args:
      name: the name of the cc_proto_library.
      srcs: the .proto files of the cc_proto_library.
      deps: a list of dependency labels; must be cc_proto_library.
      cc_libs: a list of other cc_library targets depended by the generated
          cc_library.
      include: a string indicating the include path of the .proto files.
      protoc: the label of the protocol compiler to generate the sources.
      use_grpc_plugin: a flag to indicate whether to call the grpc C++ plugin
          when processing the proto files.
      default_runtime: the implicitly default runtime which will be depended on by
          the generated cc_library target.
      **kargs: other keyword arguments that are passed to cc_library.
    """

  includes = []
  if include != None:
    includes = [include]

  grpc_plugin = None
  if has_services:
    grpc_plugin = "//external:grpc_cpp_plugin"

  cc_hdrs = _CcHdrs(srcs, has_services)
  cc_srcs = _CcSrcs(srcs, has_services)

  _proto_gen(name=name + "_genproto",
             srcs=srcs,
             deps=[s + "_genproto" for s in deps] + ["//external:protobuf_cc_wkt_protos_genproto"],
             includes=includes,
             protoc=protoc,
             grpc_plugin=grpc_plugin,
             cc11_plugin="//external:protoc-gen-cc11",
             gen_cc=1,
             outs=cc_srcs + cc_hdrs,
             visibility=["//visibility:public"],)

  if default_runtime and not default_runtime in cc_libs:
    cc_libs += [default_runtime]
  if has_services:
    cc_libs += ["//external:grpc++_lib"]
    cc_libs += ["//external:zlib"]

  native.cc_library(name=name,
                    srcs=cc_srcs,
                    hdrs=cc_hdrs,
                    deps=cc_libs + deps + ["//external:protobuf_cc_wkt_protos"],
                    includes=includes,
                    copts=copts,
                    **kargs)

def py_proto_library(name,
                     srcs=[],
                     deps=[],
                     py_libs=[],
                     py_extra_srcs=[],
                     has_services=False,
                     srcs_version="PY2AND3",
                     default_runtime="//external:protobuf_python",
                     protoc="//external:protoc",
                     **kargs):
  """Bazel rule to create a Python protobuf library from proto source files
    NOTE: the rule is only an internal workaround to generate protos. The
    interface may change and the rule may be removed when bazel has introduced
    the native rule.
    Args:
      name: the name of the py_proto_library.
      srcs: the .proto files of the py_proto_library.
      deps: a list of dependency labels; must be py_proto_library.
      py_libs: a list of other py_library targets depended by the generated
          py_library.
      py_extra_srcs: extra source files that will be added to the output
          py_library. This attribute is used for internal bootstrapping.
      include: a string indicating the include path of the .proto files.
      default_runtime: the implicitly default runtime which will be depended on by
          the generated py_library target.
      protoc: the label of the protocol compiler to generate the sources.
      **kargs: other keyword arguments that are passed to cc_library.
    """
  outs = _PyOuts(srcs)

  includes = []

  grpc_plugin = None
  if has_services:
    grpc_plugin = "//external:grpc_python_plugin"

  _proto_gen(name=name + "_genproto",
             srcs=srcs,
             deps=[s + "_genproto" for s in deps] +
                  ["//external:protobuf_python_genproto"],
             includes=includes,
             protoc=protoc,
             grpc_plugin=grpc_plugin,
             gen_py=1,
             outs=outs,
             visibility=["//visibility:public"],)

  native.py_library(name=name,
                    srcs=outs + py_extra_srcs,
                    deps=py_libs + deps,
                    srcs_version=srcs_version,
                    **kargs)
