load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool_path",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/gcc",
        ),
        tool_path(
            name = "ld",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/ld",
        ),
        tool_path(
            name = "ar",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/ar",
        ),
        tool_path(
            name = "cpp",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/cpp",
        ),
        tool_path(
            name = "gcov",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/gcov",
        ),
        tool_path(
            name = "nm",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/nm",
        ),
        tool_path(
            name = "objdump",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/objdump",
        ),
        tool_path(
            name = "strip",
            path = "crosstool_ng/" + ctx.attr.target + "-gcc_" + ctx.attr.version + "/wrappers/strip",
        ),
    ]

    compile_actions = [
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.lto_backend,
        ACTION_NAMES.clif_match,
    ]

    cpp_actions = [
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.lto_backend,
        ACTION_NAMES.clif_match,
    ]

    link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-fstack-protector",
                            "-Wall",
                            "-Wunused-but-set-parameter",
                            "-Wno-free-nonheap-object",
                            "-Wno-error=pragmas",
                            "-Wno-unknown-pragmas",
                            "-fno-omit-frame-pointer",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = compile_actions,
                flag_groups = [flag_group(flags = ["-g"])],
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-g0",
                            "-O2",
                            "-DNDEBUG",
                            "-ffunction-sections",
                            "-fdata-sections",
                            "-U_FORTIFY_SOURCE",
                            "-D_FORTIFY_SOURCE=1",
                        ],
                    ),
                ],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = cpp_actions,
                flag_groups = [flag_group(flags = ["-std=c++1y"])],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no-canonical-prefixes",
                            "-fno-canonical-system-headers",
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                        ],
                    ),
                ],
            ),
        ],
    )

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = link_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-static-libgcc",
                            "-Wl,-Bstatic",
                            "-lstdc++",
                            "-Wl,-Bdynamic",
                            "-Wl,-z,relro,-z,now",
                            "-no-canonical-prefixes",
                            "-pass-exit-codes",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = link_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wl,--gc-sections",
                        ],
                    ),
                ],
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    dbg_feature = feature(name = "dbg")
    opt_feature = feature(name = "opt")

    supports_pic_feature = feature(name = "supports_pic", enabled = True)
    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    objcopy_embed_flags_feature = feature(
        name = "objcopy_embed_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = ["objcopy_embed_data"],
                flag_groups = [flag_group(flags = ["-I", "binary"])],
            ),
        ],
    )

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = compile_actions + link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    cxx_builtin_include_directories = [
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/x86_64-unknown-linux-gnu/include/c++/" + ctx.attr.version,
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/x86_64-unknown-linux-gnu/include/c++/" + ctx.attr.version + "/x86_64-unknown-linux-gnu",
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/x86_64-unknown-linux-gnu/include/c++/" + ctx.attr.version + "/backward",
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/lib/gcc/x86_64-unknown-linux-gnu/" + ctx.attr.version + "/include",
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/lib/gcc/x86_64-unknown-linux-gnu/" + ctx.attr.version + "/include-fixed",
        "external/crosstool_ng_" + ctx.attr.target + "_gcc_" + ctx.attr.version + "/x86_64-unknown-linux-gnu/sysroot/usr/include",
    ]

    features = [
        default_compile_flags_feature,
        default_link_flags_feature,
        supports_dynamic_linker_feature,
        supports_pic_feature,
        objcopy_embed_flags_feature,
        opt_feature,
        dbg_feature,
        user_compile_flags_feature,
        sysroot_feature,
        unfiltered_compile_flags_feature,
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "gcc-" + ctx.attr.target + "-gcc_" + ctx.attr.version,
        host_system_name = ctx.attr.target,
        target_system_name = ctx.attr.target,
        target_cpu = ctx.attr.target,
        target_libc = ctx.attr.libc_version,
        compiler = "gcc-" + ctx.attr.version,
        abi_version = "gcc-" + ctx.attr.libc_version,
        abi_libc_version = ctx.attr.libc_version,
        tool_paths = tool_paths,
        features = features,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
    )

gcc_toolchain_config = rule(
    attrs = {
        "libc_version": attr.string(mandatory = True),
        "target": attr.string(mandatory = True),
        "version": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
    implementation = _impl,
)
