# MetaMachines PTX for Python
> PTX Inject and Stack PTX for Python

**PTX Inject** and **Stack PTX** are lightweight, Python-friendly tools for advanced GPU kernel manipulation and generation using NVIDIA's PTX intermediate language. Designed for developers in high-performance computing, machine learning, and scientific simulations, these tools enable dynamic kernel optimizations, rapid experimentation, and automated code generation without the slowdowns of traditional compilation pipelines.

- **PTX Inject**: Dynamically inject custom PTX code into annotated CUDA kernels for ultra-fast variations and tuning.
- **Stack PTX**: Generate valid PTX code sequences using a stack-based machine, inspired by evolutionary programming languages like [Push](https://faculty.hampshire.edu/lspector/push.html), for flexible and error-resilient code construction.

Both tools are built on efficient, header-only C libraries with Python bindings for seamless integration into your workflows. They support high-throughput operations, making them ideal for algorithmic exploration or performance benchmarking on GPUs.

Explore working examples:
- [PTX Inject examples](examples/ptx_inject/)
- [Stack PTX examples](examples/stack_ptx/)
- [Combined PTX Inject + Stack PTX examples](examples/stack_ptx_inject/)
- [Fun examples](examples/fun/README.md)

The C based header files where most of the functionality is implemented is [ptx_inject.h](src/bindings/ptx_inject.h) and [stack_ptx.h](src/bindings/stack_ptx.h). If you are interested in running these with lower overhead in C/C++ or with parallel compilation see examples in [mm-ptx](https://github.com/MetaMachines/mm-ptx).

[mm-kermac-py](https://github.com/MetaMachines/mm-kermac-py) uses **PTX Inject** and **Stack PTX** to allow users to dynamically create custom [semiring](https://en.wikipedia.org/wiki/Semiring) and semiring gradient PyTorch Tensor kernels with arbitrary amounts of hyperparameters. Recompilation of it's custom [CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md) CUDA kernels can take **~3 seconds**, however recompiling the CuTe kernel from PTX to SASS with injected PTX code takes as little as **60ms**.

## Installation

### mm-ptx
---
To instal mm-ptx use:
```bash
pip install mm-ptx
```

This package has no dependency on NVIDIA CUDA toolkit or other tools beyond nanobind. `Stack PTX` and `PTX Inject` are pure header-only C libraries relying only on the C standard library.

For dependencies running the mm-ptx examples see [examples/README.md](examples/README.md)

## PTX Inject
PTX Inject is a lightweight tool that enables dynamic modification of compiled GPU kernels by injecting custom low-level code (PTX) at user-specified points in annotated CUDA source. This allows for ultra-fast kernel variations and optimizations—ideal for algorithmic tuning, performance testing, or machine-driven experiments—without the overhead of full recompilation using tools like `nvcc` or `nvrtc`.

By processing annotated kernels, it extracts register mappings and prepares templates for injection, achieving preparation in milliseconds and supporting tens of thousands of injections per second per CPU core. The result is efficient, parallelizable compilation to executable GPU code (SASS) using `ptxas` or [`nvPtxCompiler`](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html), making it suitable for high-throughput workflows in compute-intensive applications like machine learning or scientific simulations.

Key features:

* **Annotation-Based Injection**: Mark sites in CUDA kernels with simple comments 
    ```c
    extern "C"
    __global__
    void kernel() {
        // PTX Inject will give you the PTX Register name/s for these
        float x = 3.0f; 
        float y = 4.0f;
        float z;
        /* PTX_INJECT func  
            in f32 x
            in f32 y 
            out f32 z
        */
        printf("z: %f\n");
    }
    ```

* **Register Mapping Extraction**: Automatically processes annotations to map CUDA variables to PTX registers, handling multiple site inlining and loop unrolling.

* **High Performance**: Prepares templates in **~4ms** and supports **~10,000** injections per second per CPU core.

* **Parallel Compilation**: Outputs PTX ready for fast compilation to SASS using the PTX Compiler API, with loading times under 1ms.

* **Customizable Data Types**: Using Python you can describe the names and types used in the `PTX_INJECT` annotation. See [ptx_inject_default_types.py](examples/ptx_inject_default_types.py).

A simple full working example can be found [here](examples/ptx_inject/00_simple.py).

## Stack PTX
Stack PTX provides a stack-based interface for generating valid PTX code sequences, making it easy to create, modify, and evolve GPU instructions programmatically. Inspired by the [Push](https://faculty.hampshire.edu/lspector/push.html) language for genetic programming, it treats PTX operations as stack manipulations, ensuring code remains valid even after insertions, deletions, or rearrangements. Stack PTX handles register declarations, dead code elimination

Stack PTX can write 100s instruction PTX stubs in single digit microseconds.

* **Stack Machine Model**: Push constants and instructions onto a stack; operations pop operands and push results as abstract syntax trees (ASTs). 
    
    For example:
    ```python
        instructions = [
            registry.x,
            registry.y,
            PtxInstruction.add_ftz_f32
        ]
        requests = [registry.z]
    ```
    Will take the register names from the variables 'x' and 'y' and add them together with the `add.ftz.f32` PTX instruction and assign the result to the register name for 'z'. Creating a stub like:
    ```
        {
        .reg .u32 %_c<1>;
        add.u32 %_c0, %z1, %z2;
        mov.u32 %z0, %_c0;
        }
    ```

* **Dead Code Elimination**: Automatically optimizes by removing irrelevant operations from the final PTX output.

* **Customizable Instructions**: Using Python you can describe the names and types used in the **Stack PTX** compiler. See [examples/stack_ptx_default_types.py](examples/stack_ptx_default_types.py) or look at [mm-kermac-py](https://github.com/MetaMachines/mm-kermac-py) for more complete definitions.

* **No Dependencies**: Pure C99 implementation with Python bindings for easy use.

See a simple Stack PTX example [here](examples/stack_ptx/00_simple.py).

## Stack PTX Inject
Both systems are meant to be used together to dynamically create new and potentially novel CUDA kernels extremely quickly and safely. 

* A simple example combining the two systems can be found [here](examples/stack_ptx_inject/00_simple.py).

* [domain_coloring.py](examples/fun/domain_coloring/domain_coloring.py) and [domain_coloring_random.py](examples/fun/domain_coloring_random/domain_coloring_random.py) are examples of using PTX Inject and Stack PTX to randomly create and run dynamic kernels to do [domain_coloring](examples/fun/README.md) where we generate plots of function gradients.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this software in your work, please cite it using the following BibTeX entry (generated from the [CITATION.cff](CITATION.cff) file):
```bibtex
@software{Durham_mm-ptx_2025,
  author       = {Durham, Charlie},
  title        = {mm-ptx: PTX Inject and Stack PTX for Python},
  version      = {0.1.0},
  date-released = {2025-10-19},
  url          = {https://github.com/MetaMachines/mm-ptx-py}
}
```