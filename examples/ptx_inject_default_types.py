from mm_ptx.ptx_inject import DataTypeInfoEnum

from enum import auto, unique

# Define the types the PTX Inject system might find inside the cuda PTX_INJECT annotations
# name: the name in the PTX_INJECT cuda annotations.
# ptx_mov_name: the type PTX Inject for the mov instructions.
# ptx_register_char: The inline ptx constraint letter for the PTX register type. (see inline ptx document)
# register_cast_str: The cast string for the register in the inline PTX. For example, 'half'
# is a struct that needs to be cast.
#   half x;
#   asm(
#    ".."
#    : "=h"(*(unsigned short*)&x)
#   )
# Being able to define these types dynamically in python allows customization on the user side.
# For example, you could define a type
# B1X32 = (auto(), "b1x32", "b32", "b32", 'r')
# That specifies to you that this value contains 32 individual bits per register.
# You can then add this as a type in your CUDA code PTX Inject annotations.
# When traversing the PTX Inject structure you could prefer to dynamically
# generate PTX only relevant to this being a B1X32 data type.
# i.e. PTX "and.b32", "or.b32", "shl.b32", "shr.b32", etc...
@unique
class DataTypeInfo(DataTypeInfoEnum):
#   enum_name,  enum_idx,   name,       register_type,  ptx_mov_name,   ptx_register_char,  register_cast_str
#   F16   =     (auto(),    "f16",      "f16",          "b16",          'h',                "*(unsigned short*)&"   )
#   F16X2 =     (auto(),    "f16x2",    "f16x2",        "b32",          'r',                "*(unsigned int*)&"     )
    S32   =     (auto(),    "s32",      "s32",          "s32",          'r')
    U32   =     (auto(),    "u32",      "u32",          "u32",          'r')
    F32   =     (auto(),    "f32",      "f32",          "f32",          'f')
    B32   =     (auto(),    "b32",      "b32",          "b32",          'r')
