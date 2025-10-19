
# This example is mostly the same as stack_ptx_inject 00_simple.py except we'll print
# out the value from the special register %clock using Stack PTX.

import sys
import os

import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx

from cuda.core.experimental import LaunchConfig, launch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ptx_inject_default_types import DataTypeInfo
from stack_ptx_default_types import Stack, PtxInstruction, SpecialRegister
from stack_ptx_default_types import compiler as stack_ptx_compiler
from compiler_helper import NvCompilerHelper

cuda_code = r"""
extern "C"
__global__
void
kernel() {
    unsigned int z;
    /* PTX_INJECT func
        out u32 z
    */
    printf("%u\n", z);
}
"""

processed_cuda, num_injects = ptx_inject.process_cuda(DataTypeInfo, cuda_code)
assert(num_injects == 1)

nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(processed_cuda)

inject = ptx_inject.PTXInject(DataTypeInfo, annotated_ptx)

inject.print_injects()

func = inject['func']

# Check assumptions about PTX_INJECT annotation above in cuda source.
# Notice that the data_types are all with respect to 
# our passed in data type enum.
assert( func['z'].mut_type == ptx_inject.MutType.OUT )
assert( func['z'].data_type == DataTypeInfo.U32 )

registry = stack_ptx.RegisterRegistry()
registry.add(func['z'].reg,  Stack.u32, name = 'z')
registry.freeze()

# Only 1 instruction, the special register %clock.
instructions = [
    SpecialRegister.clock
]

# The register that will be assigned to the %clock.
requests = [registry.z]

ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4
    )

print(ptx_stub)

ptx_stubs = {
    'func' : ptx_stub
}

rendered_ptx = inject.render_ptx(ptx_stubs)

print(rendered_ptx)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)
ker = mod.get_kernel("kernel")

# Launch 1 block of size 1 thread just to demonstrate the kernel.
block = int(1)
grid = int(1)
config = LaunchConfig(grid=grid, block=block)
ker_args = ()

stream = nv_compiler.dev.default_stream

launch(stream, config, ker, *ker_args)
launch(stream, config, ker, *ker_args)
launch(stream, config, ker, *ker_args)
launch(stream, config, ker, *ker_args)
launch(stream, config, ker, *ker_args)


print('Should print the result from consecutive calls of the \'%clock\' special register')
stream.sync()

