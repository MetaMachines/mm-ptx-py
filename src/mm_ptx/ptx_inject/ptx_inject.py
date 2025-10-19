from __future__ import annotations

from enum import IntEnum, Enum, unique, auto
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Mapping, Sequence, Union

from ._impl import (
    PtxInjectResult,
    PtxInjectMutType,
    ptx_inject_result_to_string,
    ptx_inject_process_cuda,
    ptx_inject_create,
    ptx_inject_destroy,
    ptx_inject_num_injects,
    # ptx_inject_inject_info_by_name,
    ptx_inject_inject_info_by_index,
    # ptx_inject_variable_info_by_name,
    ptx_inject_variable_info_by_index,
    ptx_inject_render_ptx,
)

# ---------------------------------------------------------------------------
# Errors & result checking
# ---------------------------------------------------------------------------

class PtxInjectError(RuntimeError):
    """Raised when a PTX inject C API call fails."""

def _check_result(ret: PtxInjectResult) -> None:
    """Raise PtxInjectError if `ret` indicates failure."""
    if ret != PtxInjectResult.PTX_INJECT_SUCCESS:
        raise PtxInjectError(ptx_inject_result_to_string(ret))


# ---------------------------------------------------------------------------
# Data type metadata enum
# ---------------------------------------------------------------------------

class DataTypeInfoEnum(IntEnum):
    """
    Base class for defining PTX data-type metadata on enum members.

    Each member carries:
      - ptx_name:         e.g., "f32" (can be anything)
      - register_type     e.g., "f32" (used for .reg declarations)
      - mov_postfix:      e.g., "f32" (used in mov.* postfix)
      - register_char:    'f' or 'r' (str) or its ASCII code (int)
      - register_cast_str: optional C cast prefix (e.g., '*(unsigned int*)&')
        # For example 'half' in cuda is a struct that needs to be cast with inline PTX.
        #   half x;
        #   asm(
        #    ".."
        #    : "=h"(*(unsigned short*)&x)
        #   )

    Example subclass:

        @unique
        class MyTypes(DataTypeInfoEnum):
            F32  = (auto(), "f32",      "f32",      "f32", 'f')
            S32  = (auto(), "s32",      "s32",      "s32", 'r')
            F16  = (auto(), "f16",      "f16",      "b16", 'h', "*(unsigned short*)&")
            F16X2= (auto(), "f16x2",    "f16x2",    "b32", 'r', "*(unsigned int*)&")
    """

    # Make auto() start at 0,1,2,...
    def _generate_next_value_(name, start, count, last_values):  # type: ignore[override]
        return count

    def __new__(
        cls,
        value: int,
        type_name: str,
        register_type: str,
        mov_postfix: str,
        register_char: int,
        register_cast_str: str = "",
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.type_name = type_name
        obj.register_type = register_type
        obj.mov_postfix = mov_postfix
        obj.register_char = register_char
        obj.register_cast_str = register_cast_str
        return obj

@unique
class DefaultPtxInjectDataType(DataTypeInfoEnum):
    F16   = (auto(), "f16",   "f16",    "b16", 'h', "*(unsigned short*)&")
    F16X2 = (auto(), "f16x2", "f16x2",  "b32", 'r', "*(unsigned int*)&")
    S32   = (auto(), "s32",   "s32",    "s32", 'r')
    U32   = (auto(), "u32",   "u32",    "u32", 'r')
    F32   = (auto(), "f32",   "f32",    "f32", 'f')
    B32   = (auto(), "b32",   "b32",    "b32", 'r')


# ---------------------------------------------------------------------------
# Mutability types (wrap C enum)
# ---------------------------------------------------------------------------

class MutType(Enum):
    """Argument mutability in an injection site (IN/MOD/OUT)."""
    OUT = PtxInjectMutType.PTX_INJECT_MUT_TYPE_OUT
    MOD = PtxInjectMutType.PTX_INJECT_MUT_TYPE_MOD
    IN  = PtxInjectMutType.PTX_INJECT_MUT_TYPE_IN


# ---------------------------------------------------------------------------
# Simple data holders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectArg:
    name: str
    mut_type: MutType
    data_type: DataTypeInfoEnum
    reg: str


@dataclass(frozen=True)
class Inject:
    name: str
    num_args: int
    num_sites: int
    args: List[InjectArg]


# ---------------------------------------------------------------------------
# User-facing helpers
# ---------------------------------------------------------------------------

def process_cuda(
    data_type_info_enum,
    cuda_code: str,
) -> Tuple[str, int]:
    """
    Preprocess annotated CUDA into PTX-inject-ready CUDA.

    Returns:
        (processed_cuda_source, num_injects)
    """

    names = [data_type.type_name for data_type in data_type_info_enum]
    reg_types = [data_type.register_type for data_type in data_type_info_enum]
    movs = [data_type.mov_postfix for data_type in data_type_info_enum]
    reg_chars = [data_type.register_char for data_type in data_type_info_enum]
    casts = [data_type.register_cast_str for data_type in data_type_info_enum]

    ret, required_size, num_injects = ptx_inject_process_cuda(
        names, reg_types, movs, reg_chars, casts, cuda_code, None
    )
    _check_result(ret)

    # +1 for NUL safety, though the C side may not need it.
    buffer = bytearray(required_size + 1)

    ret, written_length, num_injects = ptx_inject_process_cuda(
        names, reg_types, movs, reg_chars, casts, cuda_code, buffer
    )
    _check_result(ret)

    return buffer[:written_length].decode("utf-8"), num_injects


class PTXInject:
    """
    High-level wrapper around a PTX-inject handle.

    Typical usage:

        # 1) Process CUDA (once per translation unit)
        processed_cuda, num_injects = process_cuda(DefaultPtxInjectDataType, cuda_src)

        # 2) Build injector from annotated PTX (after compiling CUDA->PTX)
        injector = PTXInject(DefaultPtxInjectDataType, annotated_ptx)

        # 3) Inspect sites
        injector.print_injects()

        # 4) Render final PTX by supplying PTX text for each inject (dict by inject name)
        final_ptx = injector.render_ptx({"my_inject": "...", ...})

    This object is also a mapping of inject name -> {arg_name -> InjectArg} for convenience:
        injector["my_inject"]["x"].data_type
    """

    def __init__(
            self, 
            data_type_info_enum,
            annotated_ptx: str
        ):
        self._enum = data_type_info_enum

        names = [data_type.type_name for data_type in data_type_info_enum]
        reg_types = [data_type.register_type for data_type in data_type_info_enum]
        movs = [data_type.mov_postfix for data_type in data_type_info_enum]
        reg_chars = [data_type.register_char for data_type in data_type_info_enum]
        casts = [data_type.register_cast_str for data_type in data_type_info_enum]

        ret, self._handle = ptx_inject_create(names, reg_types, movs, reg_chars, casts, annotated_ptx)
        _check_result(ret)

        self.injects: List[Inject] = []
        ret, num_injects = ptx_inject_num_injects(self._handle)
        _check_result(ret)

        for inject_idx in range(num_injects):
            ret, inject_name, inject_num_args, num_sites = \
                ptx_inject_inject_info_by_index(
                    self._handle, inject_idx
                )
            _check_result(ret)

            args: List[InjectArg] = []
            for arg_idx in range(inject_num_args):
                ret, arg_name, mut_type, data_type, reg_name = \
                    ptx_inject_variable_info_by_index(
                        self._handle, inject_idx, arg_idx
                    )
                _check_result(ret)
                args.append(
                    InjectArg(
                        name=arg_name,
                        mut_type=MutType(mut_type),
                        data_type=self._enum(data_type),
                        reg=reg_name,
                    )
                )
            self.injects.append(Inject(inject_name, inject_num_args, num_sites, args))

        # Fast lookup: inject_name -> (arg_name -> InjectArg)
        self._inject_lookup: Dict[str, Dict[str, InjectArg]] = {
            inj.name: {arg.name: arg for arg in inj.args} for inj in self.injects
        }

    # --- Mapping-like sugar ---
    def __getitem__(self, inject_name: str) -> Dict[str, InjectArg]:
        return self._inject_lookup[inject_name]

    # --- Nice printing for users ---
    def print_injects(self) -> None:
        if not self.injects:
            print("No inject sites found.")
            return

        for i, inject in enumerate(self.injects, start=1):
            print(f"Inject #{i}:")
            print(f"  Name:            {inject.name!r}")
            print(f"  Number of Args:  {inject.num_args}")
            print(f"  Number of Sites: {inject.num_sites}")
            if inject.args:
                print("  Arguments:")
                for j, arg in enumerate(inject.args, start=1):
                    print(f"    Arg #{j}:")
                    print(f"      CUDA Name:     {arg.name!r}")
                    print(f"      Mutation Type: {arg.mut_type.name}")
                    print(f"      Data Type:     {arg.data_type.name}")
                    print(f"      Register Name: {arg.reg!r}")
            else:
                print("  (No arguments.)")
            print()

    # --- Core render ---
    def render_ptx(self, ptx_stubs: Mapping[str, str]) -> str:
        """
        Render final PTX by supplying PTX text for each inject (keyed by inject name).

        Args:
            ptx_stubs: dict mapping inject_name -> PTX stub source

        Returns:
            Combined PTX string with stubs injected in the correct order.
        """
        # Ensure deterministic order aligned with discovered injects
        ordered_stubs = [ptx_stubs[inject.name] for inject in self.injects]

        ret, required_size = ptx_inject_render_ptx(self._handle, ordered_stubs, None)
        _check_result(ret)

        buffer = bytearray(required_size + 1)
        ret, written_size = ptx_inject_render_ptx(self._handle, ordered_stubs, buffer)
        _check_result(ret)

        return buffer[:written_size].decode("utf-8")

    # --- Context manager & cleanup ---
    def close(self) -> None:
        if getattr(self, "_handle", None):
            ret = ptx_inject_destroy(self._handle)
            _check_result(ret)
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> "PTXInject":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        # Best-effort cleanup; avoid raising in destructor.
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "PtxInjectError",
    "MutType",
    "DataTypeInfoEnum",
    "DefaultPtxInjectDataType",
    "InjectArg",
    "Inject",
    "process_cuda",
    "PTXInject",
]
