# SPDX-FileCopyrightText: 2026 MetaMachines LLC
#
# SPDX-License-Identifier: MIT

import sys
import os

from cuda.core import LaunchConfig, launch
import mm_ptx.ptx_inject as ptx_inject
from mm_ptx import get_ptx_inject_header

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compiler_helper import NvCompilerHelper

ptx_header = get_ptx_inject_header().replace("\\", "/")

# Inline CUDA with a PTX_INJECT site we can patch from Python.
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

# Compile CUDA to annotated PTX.
nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

# Parse inject sites and inspect their arguments.
inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func = inject["func"]

assert func["x"].mut_type == ptx_inject.MutType.IN
assert func["x"].data_type == "F32"

assert func["y"].mut_type == ptx_inject.MutType.MOD
assert func["y"].data_type == "F32"

assert func["z"].mut_type == ptx_inject.MutType.OUT
assert func["z"].data_type == "F32"

# Build a stub that adds x + y into y, then x + y into z.
ptx_stubs = {
    "func": f"\tadd.ftz.f32 %{func['y'].reg}, %{func['x'].reg}, %{func['y'].reg};\n"
            f"\tadd.ftz.f32 %{func['z'].reg}, %{func['x'].reg}, %{func['y'].reg};"
}

rendered_ptx = inject.render_ptx(ptx_stubs)

# Compile injected PTX to a cubin and launch.
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
