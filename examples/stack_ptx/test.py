import sys
import os

import mm_ptx.stack_ptx as stack_ptx

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

registry = stack_ptx.RegisterRegistry()
registry.add("z2",    Stack.u32, name='x')
registry.add("z1",    Stack.u32, name='y')
registry.add("z0",   Stack.u32, name='z')
registry.freeze()

instructions = [
    registry.x,
    registry.y,
    PtxInstruction.add_u32,
]

# Now we make our requests. 
# out_0 will demand a value from the top of the u32 due to being declared a u32 value in the Register enum.
# if there is a value in the u32 stack, it will be popped and assigned to the register "out_0" in PTX.
# out_1 will demand the next value and if present will be assigned to the register "out_1" in PTX.
requests = [registry.z]

# Now we run the Stack PTX to grab the buffer.
ptx_stub = \
    stack_ptx_compiler.compile(
        registry=registry,
        instructions=instructions, 
        requests=requests,
        # How many instructions to run before we halt.
        execution_limit=100,
        max_ast_size=100,
        max_ast_to_visit_stack_depth=20,
        stack_size=128,
        max_frame_depth=4
    )

print(ptx_stub)
