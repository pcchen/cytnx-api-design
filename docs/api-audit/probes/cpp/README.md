# Raw-C++ probes (binding-fidelity verification)

These C++ programs call Cytnx's C++ methods **directly**, bypassing the pybind
layer — the one thing the Python probes cannot do. They provide the *raw-C++
side* of the **binding-fidelity** findings (where the binding layer changes
behavior versus the underlying C++ method), so those findings are runtime-
verified on **both** sides rather than source-read on the C++ side.

Unlike the Python probes (which run against the installed `cytnx==1.1.0` wheel),
these require a **local source build** of `libcytnx` with the *same* compiler
used to build the probe, because a static library's LTO bytecode is
compiler-version-specific (the PyPI wheel's `libcytnx.a` is GCC-14 LTO and will
not link with GCC 13).

## Build `libcytnx` from source (once)

The pinned `cytnx_src/` (1.1.0) builds cleanly — contrary to a note in the
sibling `pcchen/cytnx-api-analysis` repo, there is no merge conflict in this
tree.

```bash
cd cytnx_src
git submodule update --init --recursive          # morse_cmake, hptt, ...
cmake -S . -B /tmp/cytnx-build -G "Unix Makefiles" \
      -DUSE_CUDA=OFF -DUSE_MKL=OFF -DCMAKE_BUILD_TYPE=Release
make -C /tmp/cytnx-build cytnx -j"$(nproc)"       # produces /tmp/cytnx-build/libcytnx.a
```

Prerequisites (all present on a stock Ubuntu 24.04 dev box): `g++`, `cmake>=3.25`,
`make`, and system OpenBLAS/LAPACKE/arpack (`libopenblas-dev`/`liblapacke-dev`).

## Compile & run a probe

```bash
g++ -std=c++17 -fopenmp \
    -I cytnx_src/include \
    docs/api-audit/probes/cpp/UniTensor_cat01_02.cpp \
    /tmp/cytnx-build/libcytnx.a \
    -lopenblas -llapacke -larpack -lgomp \
    -o /tmp/cpp_probe && /tmp/cpp_probe
```

Expected: every line `[PASS]`, ending `cpp probe ok` (exit 0).

## Current probes

| File | Verifies (raw C++ side) |
|---|---|
| `UniTensor_cat01_02.cpp` | **UT-G5** `UniTensor& normal_(...)` returns `&self` (binding drops it) · **UT-C2** `UniTensor(Tensor)` shares memory · **UT-G11** C++ `normal_(...,-1)` treats `-1` as a literal seed (the `random_device` rule is binding-only) |
