cmake_minimum_required(VERSION 3.15...3.30)
project(mm_ptx)

# Option to enable/disable Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings with Nanobind" OFF)

# Check if Python bindings are enabled
if(BUILD_PYTHON_BINDINGS)
    # Find Python (required for Nanobind)
    find_package(Python QUIET COMPONENTS Interpreter Development.Module)
    if(Python_FOUND)
        # Find Nanobind
        find_package(nanobind CONFIG QUIET)
        if(nanobind_FOUND)
            message(STATUS "Python and Nanobind found, building Python bindings")
            nanobind_add_module(
                _ptx_inject
                src/bindings/ptx_inject_bindings.cpp
            )
            target_include_directories(_ptx_inject PRIVATE src/bindings)
            install(TARGETS _ptx_inject LIBRARY DESTINATION mm_ptx/ptx_inject/_impl)
            nanobind_add_module(
                _stack_ptx
                src/bindings/stack_ptx_bindings.cpp
            )
            target_include_directories(_stack_ptx PRIVATE src/bindings)
            install(TARGETS _stack_ptx LIBRARY DESTINATION mm_ptx/stack_ptx/_impl)
        else()
            message(WARNING "Nanobind not found. Disabling Python bindings.")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    else()
        message(WARNING "Python not found. Disabling Python bindings.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()
