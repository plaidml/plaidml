COINUTIL_CFG =  ["CoinUtils/src/config.h", "CoinUtils/src/config_coinutils.h"]
COIN_COPTS = [
  "-w",
]

config_setting(
    name = "android",
    values = {
        "android_crosstool_top": "@androidndk//:toolchain-libcpp",
    },
)

config_setting(
    name = "x64_windows",
    values = {"cpu": "x64_windows"},
)

genrule(
  name = "configure_coinutils",
  srcs = glob(["CoinUtils/src/**/*"]),
  outs = COINUTIL_CFG,
  cmd = "pushd external/coinlp_archive/CoinUtils; workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); cp -a * $$workdir; pushd $$workdir; ./configure --disable-bzlib; sed -e 's/finite/std::isfinite/' src/config.h > src/config.h.new; mv -f src/config.h.new src/config.h; popd; popd; cp $$workdir/src/*.h $(@D)/CoinUtils/src; rm -rf $$workdir",
  tools = ["CoinUtils/configure"],
  message = "Configuring CoinUtils",
  local = True, 
)

cc_library(
  name = "coinutils",
  hdrs = glob(["CoinUtils/src/*.hpp", "CoinUtils/src/*.h"]) + select({
    ":x64_windows": [],
    "//conditions:default": COINUTIL_CFG,
  }),
  srcs = glob(["CoinUtils/src/*.cpp"]),
  deps = ["//external:zlib"],
  copts = ["-DCOINUTILS_BUILD"] + COIN_COPTS + select({
    ":x64_windows": [],
    "//conditions:default": ["-DHAVE_CONFIG_H"],
  }),
  includes = ["CoinUtils/src", "BuildTools/headers"]
)


OSI_CFG =  ["Osi/src/Osi/config.h", "Osi/src/Osi/config_osi.h"]

genrule(
  name = "configure_osi",
  srcs = glob(["Osi/src/**/*"]),
  outs = OSI_CFG,
  cmd = "pushd external/coinlp_archive/Osi; workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); cp -a * $$workdir; pushd $$workdir; ./configure --disable-bzlib --with-coinutils-lib=''; popd; popd; cp $$workdir/src/Osi/*.h $(@D)/Osi/src/Osi; rm -rf $$workdir;",
  tools = ["Osi/configure"],
  message = "Configuring CoinUtils",
  local = True, 
)

cc_library(
  visibility = ["//visibility:public"],
  name = "osi",
  deps = ["coinutils"],
  hdrs = glob(["Osi/src/Osi/*.hpp", "Osi/src/Osi/*.h"]) + select({
    ":x64_windows": [],
    "//conditions:default": OSI_CFG,
  }),
  srcs = glob(["Osi/src/Osi/*.cpp"]),
  copts = ["-DOSddI_BUILD"] + COIN_COPTS + select({
    ":x64_windows": [],
    "//conditions:default": ["-DHAVE_CONFIG_H"],
  }),
  includes = ["Osi/src/Osi", "BuildTools/headers"],
)

CLP_CFG = ["Clp/src/config.h", "Clp/src/config_clp.h"]

genrule(
  name = "configure_clp",
  srcs = glob(["Clp/src/**/*"]),
  outs = CLP_CFG,
  cmd = "pushd external/coinlp_archive/Clp; workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); cp -a * $$workdir; pushd $$workdir; ./configure --disable-bzlib --with-coinutils-lib=''; popd; popd; cp $$workdir/src/*.h $(@D)/Clp/src; rm -rf $$workdir;",
  tools = ["Clp/configure"],
  message = "Configuring Clp",
  local = True, 
)

cc_library(
  name = "coinlp",
  visibility = ["//visibility:public"],
  deps = ["coinutils", "osi", "//external:zlib"],
  hdrs = glob(["Clp/src/**/*.hpp", "Clp/src/*.h"]) + select({
    ":x64_windows": [],
    "//conditions:default": CLP_CFG,
  }),
  srcs = glob(["Clp/src/**/*.cpp"], exclude=["Clp/src/ClpMain.cpp", "Clp/src/*Abc*", "Clp/src/ClpCholeskyUfl.cpp", "Clp/src/ClpCholeskyWssmp*", "Clp/src/ClpCholeskyMumps*"]),
  copts = ["-DCLP_BUILD", "-DCOIN_HAS_CLP"] + COIN_COPTS + select({
    ":x64_windows": [],
    "//conditions:default": ["-DHAVE_CONFIG_H"],
  }),
  includes = ["Clp/src", "BuildTools/headers"],
)

cc_binary(
  name = "clp",
  deps = ["coinlp"],
  srcs = ["Clp/src/ClpMain.cpp"],
  copts = ["-DCLP_BUILD", "-fpic"] + COIN_COPTS + select({
    ":x64_windows": [],
    "//conditions:default": ["-DHAVE_CONFIG_H"],
  }),
  linkopts = select({
    ":android": ["-Lexternal/androidndk/ndk/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/", "-pie"],
    "//conditions:default": ["-lm"],
  }),
)
