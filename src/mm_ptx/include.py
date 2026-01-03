# SPDX-FileCopyrightText: 2026 MetaMachines LLC
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os


def get_include_dir() -> str:
    """Return the directory that contains the bundled PTX headers."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "include"))


def get_ptx_inject_header() -> str:
    """Return the full path to the bundled ptx_inject.h."""
    return os.path.join(get_include_dir(), "ptx_inject.h")


def get_stack_ptx_header() -> str:
    """Return the full path to the bundled stack_ptx.h."""
    return os.path.join(get_include_dir(), "stack_ptx.h")
