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

## Stack PTX: stack-based instruction compiler
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

Printing `ptx_stub` gives:
```
    {
    .reg .f32 %_a<2>;
    add.ftz.f32 %_a0, %_x0, %_x2;
    add.ftz.f32 %_a1, %_x2, %_a0;
    mov.f32 %_x1, %_a1;
    }
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

### Stack PTX instruction descriptions
The instruction definitions are defined by the user and are not part of the core Stack PTX system. This allows customization of the described instructions to fit the users demands.
- Minimal example of PTX instruction and type definitions: [examples/stack_ptx_default_types.py](https://github.com/MetaMachines/mm-ptx-py/blob/master/examples/stack_ptx_default_types.py)
- More extensive example: [examples/stack_ptx_extended_types.py](https://github.com/MetaMachines/mm-ptx-py/blob/master/examples/stack_ptx_extended_types.py)

## Install
```bash
pip install mm-ptx
```

Requires Python 3.9+.

## Tests
```bash
python -m pip install -e .
python -m unittest discover -s tests
```

CUDA integration tests are skipped by default. To run them (requires `cuda.core`, `cuda.bindings`, and a CUDA-capable GPU):
```bash
 MM_PTX_RUN_CUDA_TESTS=1 python -m unittest discover -s tests
```

## Examples
- [PTX Inject](https://github.com/MetaMachines/mm-ptx-py/tree/master/examples/ptx_inject)
- [Stack PTX](https://github.com/MetaMachines/mm-ptx-py/tree/master/examples/stack_ptx)
- [PTX Inject + Stack PTX](https://github.com/MetaMachines/mm-ptx-py/tree/master/examples/stack_ptx_inject)
- [Fun](https://github.com/MetaMachines/mm-ptx-py/blob/master/examples/fun/README.md)

## More details
For the C/C++ headers and deeper implementation notes, see the mm-ptx repo:
- https://github.com/MetaMachines/mm-ptx/blob/master/README.md
- https://github.com/MetaMachines/mm-ptx/blob/master/PTX_INJECT.md
- https://github.com/MetaMachines/mm-ptx/blob/master/STACK_PTX.md

## License
MIT. See `LICENSE`.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from `CITATION.cff`):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX for Python},
  version      = {1.0.1},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx-py}
}
```
