
import sys
import os

from cuda.core.experimental import LaunchConfig, launch
import mm_ptx.ptx_inject as ptx_inject

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the default data types from the helper.
from ptx_inject_default_types import DataTypeInfo
from compiler_helper import NvCompilerHelper

# We'll use this simple kernel to PTX Inject.
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
processed_cuda, num_injects = ptx_inject.process_cuda(DataTypeInfo, cuda_code)
assert(num_injects == 1)

# Use the compiler helper.
nv_compiler = NvCompilerHelper()

# This PTX was annotated to be ready for CUDA to PTX.
annotated_ptx = nv_compiler.cuda_to_ptx(processed_cuda)

# Now parse out the PTX Inject structure and find the register assignments
# for the variables that were described in the PTX_INJECT block.
inject = ptx_inject.PTXInject(DataTypeInfo, annotated_ptx)

# Print what we found in the ptx annotated with PTX Inject.
inject.print_injects()

# Pull out the PTX_INJECT block named "func"
func = inject['func']

# Check assumptions about PTX_INJECT annotation above in cuda source.
# Notice that the data_types are all with respect to 
# our passed in data type enum. See "ptx_inject_default_types.py".
assert( func['x'].mut_type == ptx_inject.MutType.IN )
assert( func['x'].data_type == DataTypeInfo.F32 )

assert( func['y'].mut_type == ptx_inject.MutType.MOD )
assert( func['y'].data_type == DataTypeInfo.F32 )

assert( func['z'].mut_type == ptx_inject.MutType.OUT )
assert( func['z'].data_type == DataTypeInfo.F32 )

# Format a simple ptx stub using the register names we extracted from the PTX Inject annotations
# in the PTX code.
ptx_stubs = {
    'func': f"\tadd.ftz.f32 %{func['y'].reg}, %{func['x'].reg}, %{func['y'].reg};\n"
            f"\tadd.ftz.f32 %{func['z'].reg}, %{func['x'].reg}, %{func['y'].reg};"
}

# Create the actual PTX we will compile to SASS.
rendered_ptx = inject.render_ptx(ptx_stubs)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)

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
