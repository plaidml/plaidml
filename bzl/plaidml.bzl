# Tile Bazel configurations

load("//bzl:protobuf.bzl", "py_proto_library", "cc_proto_library")

PLAIDML_COPTS = select({
    "//bzl:x64_windows": [
        "/std:c++14",
        "/DWIN32_LEAN_AND_MEAN",
    ],
    "//conditions:default": [
        "--std=c++1y",
        "-Werror",
    ],
})

PY_SRCS_VER = "PY2AND3"

PLAIDML_BASE_LINKOPTS = select({
    "//bzl:android": [
        "-Lexternal/androidndk/ndk/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/",
        "-pie",
    ],
    "//bzl:x64_windows": [],
    "//bzl:darwin": [],
    "//conditions:default": [
        "-pthread",
        "-lm",
    ],
})

def plaidml_cc_library(copts=[], linkopts=[], **kwargs):
  native.cc_library(
    copts=PLAIDML_COPTS + copts,
    linkopts=PLAIDML_BASE_LINKOPTS + linkopts,
    **kwargs)

def plaidml_objc_library(copts=[], linkopts=[], **kwargs):
  native.objc_library(copts=PLAIDML_COPTS + copts + ["-Wno-shorten-64-to-32"], **kwargs)

def plaidml_cc_binary(copts=[], **kwargs):
  native.cc_binary(copts=PLAIDML_COPTS + copts, **kwargs)

def plaidml_cc_test(copts=[], deps=[], linkopts=[], **kwargs):
  native.cc_test(
    copts=copts + PLAIDML_COPTS,
    deps=deps + ["//testing:gtest_main"],
    linkopts=PLAIDML_BASE_LINKOPTS + linkopts,
    **kwargs)

def plaidml_py_library(**kwargs):
  native.py_library(srcs_version=PY_SRCS_VER, **kwargs)

def plaidml_py_init(name, **kwargs):
  pyinit_name = name + "_init_py"

  native.genrule(
    name = pyinit_name,
    visibility = ["//visibility:private"],
    outs = ["__init__.py"],
    cmd = "touch $(location __init__.py)"
  )

  plaidml_py_library(
    name = name,
    srcs = [":" + pyinit_name],
    **kwargs
  )

def plaidml_py_test(**kwargs):
  native.py_test(srcs_version=PY_SRCS_VER, **kwargs)

def plaidml_py_binary(**kwargs):
  native.py_binary(srcs_version=PY_SRCS_VER, **kwargs)

def plaidml_proto_library(name, srcs=[], has_services=False, deps=(), visibility=(),
                       testonly=0, cc_libs=[], **kwargs):
    plaidml_cc_proto_library(name=name,
                          srcs=srcs,
                          deps=deps,
                          cc_libs=cc_libs,
                          testonly=testonly,
                          has_services=has_services,
                          visibility=visibility,
                          **kwargs)

    plaidml_py_proto_library(name=name,
                          srcs=srcs,
                          deps=deps,
                          has_services=has_services,
                          testonly=testonly,
                          visibility=visibility,
                          **kwargs)

def plaidml_py_proto_library(name, srcs=[], deps=(), has_services=False,
                          visibility=[], testonly=0,
                          srcs_version="PY2AND3", **kwargs):
    py_proto_library(name=name + "_py",
                     srcs=srcs,
                     srcs_version=srcs_version,
                     deps=[d + "_py" for d in deps],
                     visibility=visibility,
                     testonly=testonly,
                     has_services=has_services,
                     **kwargs)

def plaidml_cc_proto_library(name, srcs=[], has_services=False,
                          deps=[], visibility=(), testonly=0,
                          cc_libs=[], **kwargs):
    native.filegroup(name=name + "_proto_srcs",
                     srcs=srcs,
                     testonly=testonly, )
    cc_proto_library(name=name + "_cc",
                     srcs=srcs,
                     deps=[d + "_cc" for d in deps],
                     copts=PLAIDML_COPTS,
                     cc_libs=cc_libs,
                     has_services=has_services,
                     testonly=testonly,
                     visibility=visibility,
                     **kwargs)

def plaidml_ast(name, ast, output, template = "base", visibility = None):
  native.genrule(
    name = name,
    outs = [output],
    srcs = [ast],
    tools = ["//base/util/astgen:astgen"],
    cmd = '$(location //base/util/astgen) -i $(SRCS) -t ' + template + ' -o $(OUTS)',
  )

def plaidml_grammar(name, bison_src, flex_src, outs, visibility = None):
  native.genrule(
    name = name,
    outs = outs,
    srcs = [bison_src, flex_src],
    visibility = visibility,
    # The following is known to work when building ios;
    # we need to unify the ios and non-ios verions here:
    # cmd = 'ssrcs=($(SRCS)); /usr/local/opt/bison/bin/bison $${ssrcs[0]} ; /usr/local/opt/flex/bin/flex $${ssrcs[1]} ; cp %s $(@D)' % (" ".join(outs)),
    cmd = select({
      "//bzl:darwin":
         'ssrcs=($(SRCS)); /usr/local/opt/bison/bin/bison --verbose $${ssrcs[0]} ; /usr/local/opt/flex/bin/flex $${ssrcs[1]} ; cp %s $(@D)' % (" ".join(outs)),
      "//bzl:x64_windows":
         'ssrcs=($(SRCS)); bison --verbose $${ssrcs[0]} ; flex --nounistd $${ssrcs[1]} ; cp %s $(@D)' % (" ".join(outs)),
      "//conditions:default":
         'ssrcs=($(SRCS)); bison --verbose $${ssrcs[0]} ; flex $${ssrcs[1]} ; cp %s $(@D)' % (" ".join(outs)),
    }),
  )

load(
    "@bazel_tools//tools/build_defs/apple:shared.bzl",
    "label_scoped_path",
    "xcrun_action",
    "XCRUNWRAPPER_LABEL",
)

def _xcrun_args(ctx):
  if ctx.fragments.apple.xcode_toolchain:
    return ['--toolchain', ctx.fragments.apple.xcode_toolchain]
  return []

def _cc_src_filter(files):
  result=[]
  for f in files:
    if f.extension == 'cc':
      result.append(f)
    elif f.extension == 'cpp':
      result.append(f)
    elif f.extension == 'c':
      result.append(f)
  return result

def _plaidml_objc_cc_library_aspect_impl(target, ctx):
  apple_fragment = ctx.fragments.apple

  # print('target:', target)
  # print('dir(target):', dir(target))
  # print('target.label.name:', target.label.name)
  # if hasattr(target, 'objc'):
  #   print('target.objc:', target.objc)
  #   print('dir(target.objc):', dir(target.objc))
  # if hasattr(target, 'cc'):
  #   print('target.cc:', target.cc)
  #   print('dir(target.cc):', dir(target.cc))
  #   print('target.cc.compile_flags:', target.cc.compile_flags)
  #   print('target.cc.defines:', target.cc.defines)
  #   print('target.cc.include_directories:', target.cc.include_directories)
  #   print('target.cc.libs:', target.cc.libs)
  #   print('target.cc.link_flags:', target.cc.link_flags)
  #   print('target.cc.quote_include_directories:', target.cc.quote_include_directories)
  #   print('target.cc.system_include_directories:', target.cc.system_include_directories)
  #   print('target.cc.transitive_headers:', target.cc.transitive_headers)
  # print('ctx:', ctx)
  # print('dir(ctx):', dir(ctx))
  # print('ctx.attr:', ctx.attr)
  # print('ctx.executable:', ctx.executable)
  # print('ctx.rule:', ctx.rule)
  # print('ctx.rule.attr:', ctx.rule.attr)
  # print('ctx.rule.attr.deps:', ctx.rule.attr.deps)
  # for dep in ctx.rule.attr.deps:
  #   print('dir(', str(dep), '):', dir(dep))
  # print('ctx.rule.files:', ctx.rule.files)
  # print('ctx.fragments:', ctx.fragments)
  # print('ctx.fragments.apple:', ctx.fragments.apple)
  # print('dir(ctx.fragments.apple):', dir(ctx.fragments.apple))
  # print('ctx.fragments.apple.apple_host_system_env:', ctx.fragments.apple.apple_host_system_env())
  # print('ctx.fragments.apple.bitcode_mode:', ctx.fragments.apple.bitcode_mode)
  # print('ctx.fragments.apple.ios_cpu:', ctx.fragments.apple.ios_cpu())
  # print('ctx.fragments.apple.ios_cpu_platform:', ctx.fragments.apple.ios_cpu_platform())
  # print('ctx.fragments.apple.xcode_toolchain:', ctx.fragments.apple.xcode_toolchain)
  # print('ctx.fragments.objc:', ctx.fragments.objc)
  # print('dir(ctx.fragments.objc):', dir(ctx.fragments.objc))
  # print('ctx.fragments.objc.copts:', ctx.fragments.objc.copts)
  # print('dir(apple_common):', dir(apple_common))
  # print('dir(apple_common.apple_toolchain()):', dir(apple_common.apple_toolchain()))
  # print('apple_common.apple_toolchain().platform_developer_framework_dir:', apple_common.apple_toolchain().platform_developer_framework_dir(apple_fragment))
  # print('apple_common.apple_toolchain().sdk_dir():', apple_common.apple_toolchain().sdk_dir())

  includes = set(ctx.rule.attr.includes)
  defines = set(ctx.rule.attr.defines)
  libraries = set()

  objc_providers = [x.objc for x in ctx.rule.attr.deps if hasattr(x, 'objc')]
  for o in objc_providers:
    # print(target.label.name, 'dep:', o)
    if includes:
      includes = includes | o.include
    else:
      includes = o.include
    if defines:
      defines = defines | o.define
    else:
      defines = o.define
    libraries = libraries | o.library

  obj_path = label_scoped_path(ctx, '_objc_objs/')

  objs = []

  #  Note: for a compilation step, objc_library generates a command like this:
  #    bazel-out/host/bin/external/bazel_tools/tools/objc/xcrunwrapper clang '-stdlib=libc++' '-std=gnu++11' -Wshorten-64-to-32 -Wbool-conversion -Wconstant-conversion -Wduplicate-method-match -Wempty-body -Wenum-conversion -Wint-conversion -Wunreachable-code -Wmismatched-return-types -Wundeclared-selector -Wuninitialized -Wunused-function -Wunused-variable -DOS_IOS '-miphoneos-version-min=7.0' -arch armv7 -isysroot __BAZEL_XCODE_SDKROOT__ -F __BAZEL_XCODE_SDKROOT__/System/Library/Frameworks -F __BAZEL_XCODE_DEVELOPER_DIR__/Platforms/iPhoneOS.platform/Developer/Library/Frameworks -iquote . -iquote bazel-out/local-fastbuild/genfiles -I external/easylogging_repo/src -I bazel-out/local-fastbuild/genfiles/external/easylogging_repo/src -I external/gflags_repo -I bazel-out/local-fastbuild/genfiles/external/gflags_repo -I external/gflags_repo/include -I bazel-out/local-fastbuild/genfiles/external/gflags_repo/include -fobjc-arc '--std=c++1y' -Werror -Wno-ignored-attributes -Wno-missing-braces -Wno-unreachable-code -Wno-shorten-64-to-32 -c util/error.cc -o bazel-out/local-fastbuild/bin/util/_objs/util_objc/util/error.o -MD -MF bazel-out/local-fastbuild/bin/util/_objs/util_objc/util/error.d)
  #
  # For a link step, objc_library generates a command like this:
  #   bazel-out/host/bin/external/bazel_tools/tools/objc/libtool -static -filelist bazel-out/local-fastbuild/bin/util/util_objc-archive.objlist -arch_only armv7 -syslibroot __BAZEL_XCODE_SDKROOT__ -o bazel-out/local-fastbuild/bin/util/libutil_objc.a

  # if includes:
  #   includes_args = ['-I%s' % d for d in includes]
  # else:
  #   includes_args = []
  # if defines:
  #   defines_args = ['-D%s' % d for d in defines]
  # else:
  #   defines_args = []
  quote_includes_args = []
  for inc in target.cc.quote_include_directories:
    # TODO: For now, we're not including these in the actual command line,
    # since objc_library doesn't include them.  Reconsider whether that's in fact the
    # correct behavior.
    quote_includes_args += ['-iquote', inc]

  system_includes_args = []
  for inc in target.cc.system_include_directories:
    system_includes_args += ['-I', inc]

  defines_args = []
  for d in target.cc.defines:
    defines_args.append('-D%s' % d)

  arch = ctx.fragments.apple.ios_cpu()

  libs = set()
  for src in _cc_src_filter(ctx.rule.files.srcs):
    # print('Saw cc source:', src)
    obj = ctx.new_file(obj_path + src.basename + '.o')
    objs.append(obj)
    args = _xcrun_args(ctx) + [
      'clang', '-stdlib=libc++',
      '-arch', arch,
      '-isysroot', apple_common.apple_toolchain().sdk_dir(),
      '-miphoneos-version-min=7.0',  # TODO: Find a place to get this flag
      '-F', apple_common.apple_toolchain().platform_developer_framework_dir(apple_fragment),
      '-iquote', '.',
      '-iquote', ctx.genfiles_dir.path,
      '-Wno-deprecated-declarations',  # Appease the protobuf gods
      '-Wno-unused-const-variable',  # Appease the protobuf gods
      '-Wno-shorten-64-to-32', # Make the fact that size_t is not uint64_t happy
    ] + system_includes_args + defines_args + [
      '-fobjc-arc'
    ] + ctx.rule.attr.copts + [
      '-c', src.path, '-o', obj.path
    ]
    xcrun_action(ctx,
                 inputs=[src] + list(target.cc.transitive_headers),
                 outputs=[obj],
                 mnemonic='ObjCcCompile',
                 arguments=args,
                 progress_message=('Compiling %s [ios %s]' % (src.path, arch)))

  if objs:
    lib = ctx.new_file(label_scoped_path(ctx, 'lib' + target.label.name + '_objc.a'))
    xcrun_action(ctx,
                 inputs=objs,
                 outputs=(lib,),
                 mnemonic='ObjCcArchive',
                 arguments=[
                   'libtool', '-static',
                   '-arch_only', arch,
                   '-syslibroot', apple_common.apple_toolchain().sdk_dir(),
                   '-o', lib.path] + [x.path for x in objs],
                 progress_message=('Archiving %s [ios %s]' % (lib.path, arch)))

    libs = libs | [lib]
    libraries = libraries | [lib]

  # print(target.label.name, 'libs:', libs)

  kwargs = {}

  if target.cc.include_directories:
    # TODO: Maybe add set(target.cc.quote_include_directories)
    kwargs['include'] = set(target.cc.include_directories)
  if target.cc.system_include_directories:
    kwargs['include_system'] = set(target.cc.system_include_directories)
  if target.cc.defines:
    kwargs['define'] = set(target.cc.defines)
  if target.cc.transitive_headers:
    kwargs['header'] = target.cc.transitive_headers
  if libraries:
    kwargs['library'] = libraries

  objc_provider = apple_common.new_objc_provider(
    #header=set([output_header]),
    providers=objc_providers,
    # include=ctx.rule.attr.includes,
    #linkopt=_swift_linkopts(ctx) + extra_linker_args,
    #link_inputs=set([output_module]),
    **kwargs
  )

  return struct(
    objc=objc_provider,
    objc_libs=libs,
    files=libs,
  )

plaidml_objc_cc_library_aspect = aspect(
    attr_aspects = ["deps"],
    attrs = {
        "_xcrunwrapper": attr.label(
            default = Label(XCRUNWRAPPER_LABEL),
            executable = True,
            cfg = "host",
        ),
    },
    fragments = [
        "apple",
        "objc",
    ],
    implementation = _plaidml_objc_cc_library_aspect_impl,
)

def _plaidml_objc_cc_library_impl(ctx):
  return struct(
    objc=ctx.attr.cc_library.objc,
    files=ctx.attr.cc_library.objc_libs,
  )

plaidml_objc_cc_library = rule(
    attrs = {
        "cc_library": attr.label(aspects = [plaidml_objc_cc_library_aspect]),
    },
    implementation = _plaidml_objc_cc_library_impl,
)

def run_as_test(name, target, cmd, data=[], **kwargs):
  native.genrule(name=name+"_sh",
                 output_to_bindir=1,
                 outs=[name+".sh"],
                 cmd="echo 'exec " + ' '.join(cmd) + "' > $@")
  data = data+[target]
  native.sh_test(name=name, srcs=[":"+name+"_sh"], data=data, **kwargs)

def plaidml_cp(name, files):
  native.genrule(
    name = name,
    srcs = [files[k] for k in files],
    outs = [k for k in files],
    cmd = "; ".join(["cp $(location %s) $(location %s)" % (files[k], k) for k in files])
  )

def _plaidml_py_wheel_impl(ctx):
  tpl = ctx.file._setup_py_tpl
  setup_py = ctx.new_file(ctx.label.name + '.pkg/setup.py')
  pkg_inputs = depset([setup_py])
  version = ctx.var.get('version', default='unknown')
  cpsub = "$0 $1" if ctx.var['TARGET_CPU'] == 'x64_windows' else "$1 $2"
  if ctx.file.config:
    cfg = ctx.new_file(setup_py, 'setup.cfg')
    ctx.actions.run_shell(
      outputs = [cfg],
      inputs = [ctx.file.config],
      command = "cp " + cpsub,
      arguments = [ctx.file.config.path, cfg.path],
      mnemonic = "CopySetupCfg"
    )
    pkg_inputs += [cfg]
  for tgt in ctx.attr.srcs:
    for src in tgt.files:
      dest = ctx.new_file(setup_py, 'pkg/'+ctx.attr.package+'/'+src.basename)
      ctx.actions.run_shell(
        outputs = [dest],
        inputs = [src],
        command = "cp " + cpsub,
        arguments = [src.path, dest.path],
        mnemonic = "CopyPackageFile"
      )
      pkg_inputs += [dest]
  pkg_name = ctx.attr.package.replace('/', '_')
  ctx.actions.expand_template(
    template = tpl,
    output = setup_py,
    substitutions = {
      'bzl_package_name': pkg_name,
      'bzl_version': version,
      'bzl_target_cpu': ctx.var['TARGET_CPU'],
      '{CONSOLE_SCRIPTS}': ",\n".join(ctx.attr.console_scripts)
    }
  )
  wheel_filename = "dist/%s-%s-%s-%s-%s.whl" % (
    pkg_name,
    version,
    ctx.attr.python,
    ctx.attr.abi,
    ctx.attr.platform
  )
  wheel = ctx.new_file(setup_py, wheel_filename)
  bdist_wheel_args = [setup_py.path, "--no-user-cfg", "bdist_wheel"]
  if ctx.attr.platform != 'any':
    bdist_wheel_args.append("--plat-name")
    bdist_wheel_args.append(ctx.attr.platform)
  ctx.actions.run(
    outputs = [wheel],
    inputs = pkg_inputs.to_list(),
    executable = "python",
    arguments = bdist_wheel_args,
    mnemonic = "BuildWheel",
    use_default_shell_env = True,
  )
  output = ctx.new_file(ctx.bin_dir, wheel.basename)
  ctx.actions.run_shell(
    outputs = [output],
    inputs = [wheel],
    command = "cp " + cpsub,
    arguments = [wheel.path, output.path],
    mnemonic = "CopyWheel"
  )
  return DefaultInfo(files=depset([output]))

plaidml_py_wheel = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "config": attr.label(allow_single_file = [".cfg"]),
        "data_files": attr.label_list(),
        "package": attr.string(mandatory = True),
        "python": attr.string(mandatory = True),
        "abi": attr.string(default = "none"),
        "platform": attr.string(default = "any"),
        "console_scripts": attr.string_list(),
        "_setup_py_tpl": attr.label(
            default = Label("//bzl:setup.tpl.py"),
            allow_single_file = True,
        ),
    },
    implementation = _plaidml_py_wheel_impl,
)

def _plaidml_version_impl(ctx):
  ctx.actions.expand_template(
    template=ctx.file._version_cc_tpl,
    output=ctx.outputs.version_file,
    substitutions={
      "{PREFIX}": ctx.attr.prefix,
      "{VERSION}": ctx.var.get('version', default='unknown'),
    })

plaidml_cc_version = rule(
    attrs = {
        "prefix": attr.string(mandatory = True),
        "_version_cc_tpl": attr.label(
            default = Label("//bzl:version.cc.tpl"),
            allow_files = True,
            single_file = True,
        ),
    },
    output_to_genfiles = True,
    outputs = {"version_file": "_version.cc"},
    implementation = _plaidml_version_impl,
)
