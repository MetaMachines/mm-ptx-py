# SPDX-FileCopyrightText: 2026 MetaMachines LLC
#
# SPDX-License-Identifier: MIT

# This example is mostly the same as stack_ptx_inject 00_simple.py except we'll print
# out the value from the special register %clock using Stack PTX.

import sys
import os

import mm_ptx.ptx_inject as ptx_inject
import mm_ptx.stack_ptx as stack_ptx
from mm_ptx import get_ptx_inject_header

from cuda.core import LaunchConfig, launch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, SpecialRegister
from stack_ptx_default_types import compiler as stack_ptx_compiler
from compiler_helper import NvCompilerHelper

ptx_header = get_ptx_inject_header().replace("\\", "/")

cuda_code = f"""
#include \"{ptx_header}\"

extern \"C\"
__global__
void
kernel() {{
    unsigned int z;
    PTX_INJECT(\"func\",
        PTX_OUT(U32, z, z)
    );
    printf(\"%u\\n\", z);
}}
"""

nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func = inject["func"]

assert func["z"].mut_type == ptx_inject.MutType.OUT
assert func["z"].data_type == "U32"

registry = stack_ptx.RegisterRegistry()
registry.add(func["z"].reg, Stack.u32, name="z")
registry.freeze()

instructions = [
    SpecialRegister.clock
]

requests = [registry.z]

ptx_stub = stack_ptx_compiler.compile(
    registry=registry,
    instructions=instructions,
    requests=requests,
    execution_limit=100,
    max_ast_size=100,
    max_ast_to_visit_stack_depth=20,
    stack_size=128,
    max_frame_depth=4,
)

print(ptx_stub)

ptx_stubs = {
    "func": ptx_stub
}

rendered_ptx = inject.render_ptx(ptx_stubs)

print(rendered_ptx)

mod = nv_compiler.ptx_to_cubin(rendered_ptx)
ker = mod.get_kernel("kernel")

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

print("Should print the result from consecutive calls of the '%clock' special register")
stream.sync()
