
from mm_ptx.stack_ptx import (
    StackTypeEnum, 
    ArgTypeEnum, 
    create_instruction_enum,
    create_special_register_enum,
    StackPtx
)

from enum import auto, unique

@unique
class Stack(StackTypeEnum):
    f32 =   (auto(), "f32")
    s32 =   (auto(), "s32")
    u32 =   (auto(), "u32")

@unique
class ArgType(ArgTypeEnum):
    f32 =   (auto(), Stack.f32)
    u32 =   (auto(), Stack.u32)

@unique
class PtxInstruction(create_instruction_enum(ArgType)):
    add_u32 =               (auto(),    "add.u32",              [ArgType.u32, ArgType.u32],     [ArgType.u32])
    add_ftz_f32 =           (auto(),    "add.ftz.f32",          [ArgType.f32, ArgType.f32],     [ArgType.f32])
    mul_ftz_f32 =           (auto(),    "mul.ftz.f32",          [ArgType.f32, ArgType.f32],     [ArgType.f32])
    sin_approx_ftz_f32 =    (auto(),    "sin.approx.ftz.f32",  [ArgType.f32],                   [ArgType.f32])
    cos_approx_ftz_f32 =    (auto(),    "cos.approx.ftz.f32",  [ArgType.f32],                   [ArgType.f32])

@unique
class SpecialRegister(create_special_register_enum(ArgType)):
    clock =     (auto(),    "clock",      ArgType.u32)
    tid_x =     (auto(),    "tid.x",      ArgType.u32)

compiler = \
    StackPtx(
        stack_enum=Stack,
        arg_enum=ArgType,
        ptx_instruction_enum=PtxInstruction, 
        special_register_enum=SpecialRegister
    )