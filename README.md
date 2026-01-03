# mm-ptx (Python)
PTX Inject and Stack PTX with Python bindings.

This package ships two small, header-only C libraries plus Python wrappers:
- PTX Inject: find marked sites in PTX and inject your own PTX at those sites.
- Stack PTX: generate PTX stubs you can inject at those sites.

## PTX Inject: what you write
Mark a site in CUDA with macros:
```c++
#include <ptx_inject.h>

extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    PTX_INJECT("func",
        PTX_IN (F32, x, x),
        PTX_MOD(F32, y, y),
        PTX_OUT(F32, z, z)
    );
    out[0] = z;
}
```

Compile the CUDA to PTX (nvcc or cuda.core), then build and inject a stub in Python:
```python
from mm_ptx.ptx_inject import PTXInject

annotated_ptx = "..."  # PTX from nvcc/cuda.core
inject = PTXInject(annotated_ptx)

func = inject["func"]
stub = (
    f"\tadd.ftz.f32 %{func['y'].reg}, %{func['x'].reg}, %{func['y'].reg};\n"
    f"\tadd.ftz.f32 %{func['z'].reg}, %{func['x'].reg}, %{func['y'].reg};"
)

final_ptx = inject.render_ptx({"func": stub})
```

This would be equivalent to writing this CUDA kernel directly but without the CUDA to PTX compilation overhead:
```c++
extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    y = x + y;
    z = x + y;
    out[0] = z;
}
```

If you do not want to hand-write PTX, you can use Stack PTX to generate the stub:
```python
from mm_ptx.stack_ptx import RegisterRegistry
from stack_ptx_default_types import Stack, PtxInstruction, compiler

# Setup naming associations
registry = RegisterRegistry()
registry.add(func["x"].reg, Stack.f32, name="x")
registry.add(func["y"].reg, Stack.f32, name="y")
registry.add(func["z"].reg, Stack.f32, name="z")
registry.freeze()

# Instructions to run
instructions = [
    registry.x,                     # Push 'x'
    registry.y,                     # Push 'y'
    PtxInstruction.add_ftz_f32,     # Pop 'x', Pop 'y', Push ('x' + 'y')
    registry.x,                     # Push 'x'
    PtxInstruction.add_ftz_f32      # Pop 'x', Pop ('x' + 'y'), Push ('x' + ('x' + 'y')) 
]

# Create ptx stub
ptx_stub = compiler.compile(
    registry=registry,
    instructions=instructions,
    requests=[registry.z],
    ...
)

# Inject the ptx stub in to the ptx inject site/s
final_ptx = inject.render_ptx({"func": ptx_stub})
```

This would be equivalent to writing this CUDA kernel directly but without the CUDA to PTX compilation overhead:
```c++
extern "C"
__global__
void kernel(float* out) {
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    z = x + (x + y);
    out[0] = z;
}
```

## Header access
```python
from mm_ptx import get_include_dir, get_ptx_inject_header

include_dir = get_include_dir()
header_path = get_ptx_inject_header().replace("\\", "/")
```

Include the header by absolute path if you want:
```c++
#include "<header_path>"
```

## Install
```bash
pip install mm-ptx
```

Requires Python 3.9+.

## Examples
- [PTX Inject](examples/ptx_inject/)
- [Stack PTX](examples/stack_ptx/)
- [PTX Inject + Stack PTX](examples/stack_ptx_inject/)
- [Fun](examples/fun/README.md)

## More details
For the C/C++ headers and deeper implementation notes, see the mm-ptx repo:
- https://github.com/MetaMachines/mm-ptx/blob/dev/README.md
- https://github.com/MetaMachines/mm-ptx/blob/dev/PTX_INJECT.md
- https://github.com/MetaMachines/mm-ptx/blob/dev/STACK_PTX.md

## License
MIT. See `LICENSE`.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from `CITATION.cff`):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX for Python},
  version      = {1.0.0},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx-py}
}
```
