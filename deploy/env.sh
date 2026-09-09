# chlorine-server — Strix Halo (evileye) build/run environment
# Mirrors the halogen 0.1.3 image env: ROCm 7.14 wheels in a py3.12 venv.
SP="$HOME/chlorine-venv/lib/python3.12/site-packages"
export ROCM_PATH="$SP/_rocm_sdk_core"
export HIP_PATH="$SP/_rocm_sdk_core"
export HIP_DEVICE_LIB_PATH="$SP/_rocm_sdk_core/lib/llvm/amdgcn/bitcode"
export LD_LIBRARY_PATH="$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$SP/_rocm_sdk_core/bin:$SP/_rocm_sdk_core/lib/llvm/bin:$PATH"
