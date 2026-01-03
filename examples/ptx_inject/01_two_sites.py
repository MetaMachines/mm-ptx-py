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

cuda_code = f"""
#include \"{ptx_header}\"

extern \"C\"
__global__
void
kernel() {{
    float x = 5;
    float y = 3;
    float z = 0;
    float w = 0;

    PTX_INJECT(\"add\",
        PTX_IN (F32, x, x),
        PTX_IN (F32, y, y),
        PTX_OUT(F32, z, z)
    );

    PTX_INJECT(\"mul\",
        PTX_IN (F32, x, x),
        PTX_IN (F32, y, y),
        PTX_OUT(F32, w, w)
    );

    printf(\"%f %f\\n\", z, w);
}}
"""

nv_compiler = NvCompilerHelper()

annotated_ptx = nv_compiler.cuda_to_ptx(cuda_code)

inject = ptx_inject.PTXInject(annotated_ptx)

inject.print_injects()

func_add = inject["add"]
func_mul = inject["mul"]

ptx_stubs = {
    "add": f"\tadd.ftz.f32 %{func_add['z'].reg}, %{func_add['x'].reg}, %{func_add['y'].reg};",
    "mul": f"\tmul.ftz.f32 %{func_mul['w'].reg}, %{func_mul['x'].reg}, %{func_mul['y'].reg};",
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

print("Should print 8.0000 15.0000")
stream.sync()
