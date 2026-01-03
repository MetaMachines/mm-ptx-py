# SPDX-FileCopyrightText: 2025 MetaMachines LLC
#
# SPDX-License-Identifier: MIT

# For this example we're now going to fuse both the Stack PTX and the PTX Inject systems in to one.
# We're going to declare a kernel with a PTX_INJECT declaration. We'll pull out the
# register names assigned to the cuda variables and then use them with Stack PTX to
# form valid PTX code. We'll compile the PTX and run it.

import sys
import os

import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx
from mm_ptx import get_ptx_inject_header

from cuda.core import LaunchConfig, launch

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler
from compiler_helper import NvCompilerHelper

ptx_header = get_ptx_inject_header().replace("\\", "/")

cuda_code = f"""
#include \"{ptx_header}\"

extern \"C\"
__global__
void
kernel() {{
    float x = 5;
    float y = 3;
    float z;
    for (int i = 0; i < 2; i++) {{
        PTX_INJECT(\"func\",
            PTX_IN (F32, x, x),
            PTX_MOD(F32, y, y),
            PTX_OUT(F32, z, z)
        );
    }}
    printf(\"%f\\n\", z);
}}
"""

nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func = inject["func"]

assert func["x"].mut_type == ptx_inject.MutType.IN
assert func["x"].data_type == "F32"

assert func["y"].mut_type == ptx_inject.MutType.MOD
assert func["y"].data_type == "F32"

assert func["z"].mut_type == ptx_inject.MutType.OUT
assert func["z"].data_type == "F32"

registry = stack_ptx.RegisterRegistry()
registry.add(func["x"].reg, Stack.f32, name="x")
registry.add(func["y"].reg, Stack.f32, name="y")
registry.add(func["z"].reg, Stack.f32, name="z")
registry.freeze()

instructions = [
    registry.x,
    registry.y,
    PtxInstruction.add_ftz_f32,
    Stack.f32.dup,
    registry.x,
    PtxInstruction.add_ftz_f32,
]

print(instructions)

requests = [registry.z, registry.y]

ptx_stub = stack_ptx_compiler.compile(
    registry=registry,
    instructions=instructions,
    requests=requests,
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4,
    store_size=16,
)

ptx_stubs = {
    "func": ptx_stub
}

rendered_ptx = inject.render_ptx(ptx_stubs)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)

ker = mod.get_kernel("kernel")

block = int(1)
grid = int(1)
config = LaunchConfig(grid=grid, block=block)
ker_args = ()

stream = nv_compiler.dev.default_stream

launch(stream, config, ker, *ker_args)

print("Should print 18.0000")
stream.sync()
