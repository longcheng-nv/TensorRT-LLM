#!/usr/bin/env python3
"""Pure-Python envsubst replacement.

Reads a template from stdin and substitutes ${VAR} / $VAR with current
environment values, writing to stdout. Unset variables expand to "".

Required because some hosts (e.g., umb-b300 internal VMs) lack the
gettext-base package that provides /usr/bin/envsubst.
"""
import os
import re
import sys

_PAT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def sub(m: re.Match) -> str:
    name = m.group(1) or m.group(2)
    return os.environ.get(name, "")


sys.stdout.write(_PAT.sub(sub, sys.stdin.read()))
