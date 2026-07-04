"""Extract the live pybind parameter signatures for every public callable of
every in-scope unit, as machine-readable JSON on stdout.

This is the empirical ground truth for the N4 (argument naming/ordering/default
consistency) sweep in docs/api-audit/parameter-consistency.md. Signatures come
from the installed cytnx==1.1.0 wheel's pybind docstrings, so what this prints
is exactly what a Python caller sees at runtime.

Usage: source tools/env.sh && $PY tools/param_inventory.py > /tmp/params.json

Each record:
  {"unit","member","overload","params":[{"pos","name","type","default"}],
   "opaque": bool}
`opaque` is True when the signature is `*args, **kwargs` (pybind hides the real
parameters) or could not be parsed. `self` is dropped. A parameter whose name
is `argN` is an erased name (pybind lost the C++ argument name) -- itself an N4
finding.
"""
import sys
import re
import json
import inspect

try:
    from unit_registry import UNIT_REGISTRY
except ImportError:
    from tools.unit_registry import UNIT_REGISTRY


def split_top_level(s):
    """Split a comma-separated arg list, ignoring commas inside [] () | brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out if a.strip()]


def parse_one_param(tok):
    """Parse `name: type = default` / `name: type` / `name` / `*args`."""
    tok = tok.strip()
    if tok in ("*args", "**kwargs", "*", "/"):
        return None
    name, typ, default = tok, None, None
    # peel default (only split on the FIRST top-level '='; defaults have no nested '=')
    if "=" in tok:
        # find '=' at bracket depth 0
        depth = 0
        for i, ch in enumerate(tok):
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            elif ch == "=" and depth == 0:
                default = tok[i + 1:].strip()
                tok = tok[:i].strip()
                break
    if ":" in tok:
        name, typ = tok.split(":", 1)
        name, typ = name.strip(), typ.strip()
    else:
        name = tok.strip()
    return {"name": name, "type": typ, "default": default}


SIG_RE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*->", re.DOTALL)


def parse_signature_line(line, member):
    """Return (params_list, opaque_bool) for one signature line, or None if no match."""
    m = SIG_RE.match(line)
    if not m or m.group(1) != member:
        return None
    argstr = m.group(2)
    if "*args" in argstr and "**kwargs" in argstr:
        return ([], True)
    params = []
    for tok in split_top_level(argstr):
        p = parse_one_param(tok)
        if p is None:
            continue
        if p["name"] in ("self",):
            continue
        params.append(p)
    return (params, False)


def signature_lines(member_name, doc):
    """Yield each concrete signature line from a pybind docstring."""
    lines = doc.splitlines()
    if any("Overloaded function" in ln for ln in lines[:3]):
        for ln in lines:
            if re.match(r"^\s*\d+\.\s", ln):
                yield ln
    else:
        # plain: the signature is (usually) the first line
        for ln in lines[:2]:
            if SIG_RE.match(ln):
                yield ln
                break


def main():
    records = []
    for unit, obj in UNIT_REGISTRY.items():
        for name in sorted(m for m in dir(obj) if not m.startswith("_")):
            member = getattr(obj, name)
            if not callable(member):
                continue
            doc = inspect.getdoc(member) or ""
            got_any = False
            for oi, line in enumerate(signature_lines(name, doc)):
                parsed = parse_signature_line(line.strip(), name)
                if parsed is None:
                    continue
                params, opaque = parsed
                records.append({
                    "unit": unit, "member": name, "overload": oi,
                    "params": [{"pos": i, **p} for i, p in enumerate(params)],
                    "opaque": opaque,
                })
                got_any = True
            if not got_any:
                # callable but no parseable signature (rare); record as opaque
                records.append({
                    "unit": unit, "member": name, "overload": 0,
                    "params": [], "opaque": True,
                })
    json.dump(records, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
