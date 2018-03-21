# LLVM
First build the static libs:
`./t2 build @llvm_archive//:static.so`

## linux_x86_64

```
cp bazel-bin/external/llvm_archive/libbase.lo lib/linux_x86_64/libllvm_base.a
cp bazel-bin/external/llvm_archive/liblib.lo lib/linux_x86_64/libllvm_lib.a
cp bazel-bin/external/llvm_archive/libtargets.lo lib/linux_x86_64/libllvm_targets.a
```

## macos_x86_64

```
cp bazel-bin/external/llvm_archive/libbase.lo lib/macos_x86_64/libllvm_base.a
cp bazel-bin/external/llvm_archive/liblib.lo lib/macos_x86_64/libllvm_lib.a
cp bazel-bin/external/llvm_archive/libtargets.lo lib/macos_x86_64/libllvm_targets.a
```
