import sys
import os

from enum import IntEnum 

import mm_ptx.stack_ptx as stack_ptx

# Use the upper directory helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stack_ptx_default_types import Stack, PtxInstruction
from stack_ptx_default_types import compiler as stack_ptx_compiler

registry = stack_ptx.RegisterRegistry()
registry.add("out_0",   Stack.u32)
registry.add("out_1",   Stack.u32)
registry.add("out_2",   Stack.u32)
registry.add("out_3",   Stack.u32)
registry.freeze()

# This helps name the store locations.
class Var(IntEnum):
    add = 0

# Describe the instructions we'd like to run as a list.
instructions = [
    Stack.u32.constant(1),
    Stack.u32.constant(2),
    PtxInstruction.add_u32,
    # Store the result of 1 + 2 for later. Pops the stack and saves the value
    Stack.u32.store(Var.add),   
    Stack.u32.constant(3),  # Push 3
    Stack.u32.constant(4),  # Push 4
    # Now load the result of 1 + 2 four times to show its
    # the only value output among the four values requested.
    Stack.load(Var.add),    
    Stack.load(Var.add),
    Stack.load(Var.add),
    Stack.load(Var.add),
]

requests = [registry.out_0, registry.out_1, registry.out_2, registry.out_3]

# Now we run the Stack PTX to grab the buffer.
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
        # Can change this to increase the storage amount.
        # Each store element is 8 bytes.
        store_size=16
    )

print(ptx_stub)
