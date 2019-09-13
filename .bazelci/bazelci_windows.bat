echo %PATH%
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /NoRegistry=0 /S /D=%UserProfile%\miniconda3
%UserProfile%\miniconda3\Scripts\conda.exe init cmd.exe
%UserProfile%\miniconda3\Scripts\conda.exe update -n base conda
%UserProfile%\miniconda3\Scripts\conda.exe env update --file environment-windows.yml
set PLAIDML_DEVICE_IDS=llvm_cpu.0
set PLAIDML_TARGET=llvm_cpu
set PLAIDML_EXPERIMENTAL=1
wget https://releases.bazel.build/0.28.1/release/bazel-0.28.1-windows-x86_64.exe
bazel-0.28.1-windows-x86_64.exe test //... --config=windows_x86_64
