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
| `UniTensor_04_labels.cpp` | **UT-L2** `UniTensor& relabel_(...)` returns `&*this` (the raw `c_relabel_` binding returns `None`) · **UT-L1** `relabel(...) const` returns a distinct, data-sharing object |
| `UniTensor_05_structure.cpp` | **UT-S5** `combineBond` (singular) EXISTS in C++, mutates in place, and returns `&*this` (its Python absence is a binding gap) · in-place `permute_`/`contiguous_`/`truncate_` return `&*this` |
| `UniTensor_06_element.cpp` | **UT-E1** `get_elem<T>`/`set_elem<T>` round-trip correctly across all 11 element dtypes (not just the 4 the Python lambda covers) · **UT-E2** `get(accessors)`/`set(accessors, rhs)` EXIST as C++ methods · **UT-E3** `get_block_` returns a shared-data view vs. `get_block`'s copy |
| `UniTensor_07_arithmetic.cpp` | **UT-A4** in-place `Conj_`/`Trace_`/`Pow_`/`Transpose_`/`Dagger_`/`normalize_` each return `&*this` · **UT-A5** `Inv_(clip)` EXISTS and returns `&*this` · **UT-A2** `operator%`/`linalg::Mod` work for a UniTensor⊗scalar (Dense) but `Mod(UniTensor,UniTensor)` is an unfinished `[Mod][Developing]` stub |
| `UniTensor_12_typedevice.cpp` | **UT-T1** no-op `astype(same_dtype)`/`to(same_device)` return a FRESH, distinct UniTensor wrapper (not `*this`) — contrasted with in-place `to_`, which returns `&self`; a real `astype` conversion and `clone()` return independent, non-data-sharing copies |
