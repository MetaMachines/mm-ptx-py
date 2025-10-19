
# For this example we're now going to fuse both the Stack PTX and the PTX Inject systems in to one.
# We're going to declare a kernel with a PTX_INJECT declaration. We'll pull out the 
# register names assigned to the cuda variables and then use them with Stack PTX to
# form valid PTX code. We'll compile the PTX and run it.

import sys
import os

from enum import auto, unique
import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx

from cuda.core.experimental import LaunchConfig, launch

# We'll import some default type info from the helpers.
# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ptx_inject_default_types import DataTypeInfo

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

from compiler_helper import NvCompilerHelper

# We'll use this simple kernel to PTX Inject.
# We'll declare that "x" is input only.
# "y" is modifiable.
# "z" is output only.
# We'll throw this in a fixed size for-loop to show that inlined loops
# cause the inject to show up in multiple sites in the PTX 
# but PTX Inject will inject the stubs for each inlined site. 
cuda_code = r"""
extern "C"
__global__
void
kernel() {
    float x = 5;
    float y = 3;
    float z;
    for (int i = 0; i < 2; i++) {
        /* PTX_INJECT func
            in f32 x
            mod f32 y
            out f32 z
        */
    }
    printf("%f\n", z);
}
"""

# Process the PTX Inject CUDA to produce new CUDA code that is ready for CUDA to PTX.
# The processed CUDA is structured in a way to get the CUDA -> PTX compiler to name
# the registers for the CUDA variables we want and make it findable in the PTX code.
processed_cuda, num_injects = ptx_inject.process_cuda(DataTypeInfo, cuda_code)
assert(num_injects == 1)

nv_compiler = NvCompilerHelper()

# We now compile the CUDA to PTX.
annotated_ptx = nv_compiler.cuda_to_ptx(processed_cuda)

# Now parse out the PTX Inject structure and make it ready for
# injecting PTX stubs.
inject = ptx_inject.PTXInject(DataTypeInfo, annotated_ptx)

# Print what we found in the ptx annotated with PTX Inject.
# We should see that the number of sites for the "func" PTX_INJECT
# block is 2 due to the for-loop being unrolled.
# We should see what the PTX Inject system decided to call each
# register.
inject.print_injects()

# We'll grab the information about the PTX_INJECT block named "func"
func = inject['func']

# Check assumptions about PTX_INJECT annotation above in cuda source.
# Notice that the data_types are all with respect to 
# our passed in data type enum.
assert( func['x'].mut_type == ptx_inject.MutType.IN )
assert( func['x'].data_type == DataTypeInfo.F32 )

assert( func['y'].mut_type == ptx_inject.MutType.MOD )
assert( func['y'].data_type == DataTypeInfo.F32 )

assert( func['z'].mut_type == ptx_inject.MutType.OUT )
assert( func['z'].data_type == DataTypeInfo.F32 )

# Now for Stack PTX we'll declare the registers.
# We'll name the registers with the values we found 
# from the PTX Inject system instead of fixed names
# like the stack_ptx examples.

registry = stack_ptx.RegisterRegistry()
registry.add(func['x'].reg,  Stack.f32, name = 'x')
registry.add(func['y'].reg,  Stack.f32, name = 'y')
registry.add(func['z'].reg,  Stack.f32, name = 'z')
registry.freeze()


# We declare the Stack PTX instructions to run.
# We push the registers on the f32 stack, run the
# add ptx instruction and a meta instruction.
instructions = [
    registry.x,
    registry.y,
    PtxInstruction.add_ftz_f32,
    Stack.f32.dup,
    registry.x,
    PtxInstruction.add_ftz_f32,
]

print(instructions)

# We request two values, out_z and mod_y to be assigned
# values from their declared stack in the Register enum.
# (f32 in this case). 
requests = [registry.z, registry.y]

# Generate the PTX stub for the instructions.
ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4,
        store_size=16
    )

# Create a dict that declares that the PTX_INJECT "func"
# block is to be assigned the ptx_stub.
ptx_stubs = {
    'func' : ptx_stub
}

# Now create the full PTX. The ptx_stubs will be injected
# in to the PTX at the relevant sites.
rendered_ptx = inject.render_ptx(ptx_stubs)

# Compile the PTX to cubin.
mod = nv_compiler.ptx_to_cubin(rendered_ptx)

# Grab the kernel out of the cubin.
ker = mod.get_kernel("kernel")

# Launch 1 block of size 1 thread just to demonstrate the kernel.
block = int(1)
grid = int(1)
config = LaunchConfig(grid=grid, block=block)
ker_args = ()

stream = nv_compiler.dev.default_stream

launch(stream, config, ker, *ker_args)

print('Should print 18.0000')
stream.sync()

