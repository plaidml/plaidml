# LLVM

## macos_x86_64

```
./t2 build @llvm_archive//:static.so
cp private/bazel-bin/external/llvm_archive/libbase.lo public/plaidml/lib/macos_x86_64/libllvm_base.a
cp private/bazel-bin/external/llvm_archive/liblib.lo public/plaidml/lib/macos_x86_64/libllvm_lib.a
cp private/bazel-bin/external/llvm_archive/libtargets.lo public/plaidml/lib/macos_x86_64/libllvm_targets.a
```
