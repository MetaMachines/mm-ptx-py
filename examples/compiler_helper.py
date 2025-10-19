from cuda.core.experimental import Device, Program, ProgramOptions

import sys

class NvCompilerHelper:
    def __init__(self):
        self.dev = Device()
        self.dev.set_current()
        capability = self.dev.compute_capability
        self.arch=f"sm_{capability.major}{capability.minor}"

    def cuda_to_ptx(
        self,
        cuda: str
    ):
        program_options = ProgramOptions(std="c++11", arch=self.arch)
        prog = Program(cuda, code_type="c++", options=program_options)
        # Compile the CUDA to PTX.
        mod = prog.compile("ptx", logs=sys.stdout,) 
        return mod.code.decode('utf-8')

    def ptx_to_cubin(
        self,
        ptx: str
    ):
        program_options = ProgramOptions(arch=self.arch)
        prog = Program(ptx, code_type="ptx", options=program_options)

        mod = prog.compile("cubin", logs=sys.stdout,)
        return mod