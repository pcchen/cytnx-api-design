"""Behavioral probe for the five enum/config units (Cytnx 1.1.0):
Type, Device, SymType, bondType, fermionParity.

Every behavioral claim made in docs/api-audit/per-class/enums.md's Parity
and Consistency findings sections is backed by a report() assertion here.
Run with: source tools/env.sh && $PY docs/api-audit/probes/enums.py
"""
import sys, os, io, contextlib, enum as pyenum
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

Type = cytnx.Type
Device = cytnx.Device
SymType = cytnx.SymType
bondType = cytnx.bondType
fermionParity = cytnx.fermionParity

# =========================================================================
# Type
# =========================================================================

expected_type_values = {
    "Void": 0, "ComplexDouble": 1, "ComplexFloat": 2, "Double": 3, "Float": 4,
    "Int64": 5, "Uint64": 6, "Int32": 7, "Uint32": 8, "Int16": 9, "Uint16": 10,
    "Bool": 11,
}
report("Type's 12 members have the exact integer codes from Type_class::Typeinfos "
       "(cy_typeid_v, the Type_list variant index order in Type.hpp)",
       all(int(getattr(Type, name)) == val for name, val in expected_type_values.items()))

report("Type.Double is a distinct enum-wrapper object, NOT a plain Python int "
       "(isinstance(Type.Double, int) is False)",
       isinstance(Type.Double, int) is False)

report("Type members are singletons: Type.Double is Type.Double (repeated attribute "
       "access returns the identical object)",
       Type.Double is Type.Double)

d = Type.Double
report("An enum instance exposes .name/.value (standard pybind11 enum descriptors)",
       d.name == "Double" and d.value == 3)

# --- headline: Type_class's large C++ static-method surface is entirely ---
# unbound in Python; only the bare 12 enum values + name/value are exposed.
# (py::enum_<Type_class::Type> is bound directly -- Type_class itself, with
# getname/typeSize/is_unsigned/is_complex/is_float/is_int/type_promote/
# check_type, has no py::class_ binding at all.)

report("None of Type_class's C++ static utility methods (getname, typeSize, "
       "is_unsigned, is_complex, is_float, is_int, type_promote, check_type) "
       "are reachable from Python on cytnx.Type",
       not any(hasattr(Type, m) for m in
               ("getname", "typeSize", "is_unsigned", "is_complex", "is_float",
                "is_int", "type_promote", "check_type")))

t = cytnx.Tensor([2, 2], dtype=cytnx.Type.Double)
report("The only Python-reachable equivalent of Type_class::getname() is via "
       "Tensor.dtype_str(), NOT via any method on Type itself",
       t.dtype() == 3 and t.dtype_str() == "Double (Float64)")

try:
    Type.Double + Type.Float
    type_arithmetic_worked = True
except TypeError:
    type_arithmetic_worked = False
report("Type is bound WITHOUT py::arithmetic(): Type.Double + Type.Float raises "
       "TypeError (no implicit int-like arithmetic on the enum wrapper itself)",
       type_arithmetic_worked is False)

# =========================================================================
# Device
# =========================================================================

report("cytnx.Device is a plain Python module object (py::module_::def_submodule), "
       "NOT a class/enum -- unlike Type/SymType/bondType/fermionParity, each a "
       "real py::enum_-bound type",
       type(Device).__name__ == "module")

report("Device.cpu/Device.cuda are plain Python int (isinstance(..., int) is True), "
       "unlike Type.Double which is a distinct enum-wrapper type",
       isinstance(Device.cpu, int) is True and isinstance(Device.cuda, int) is True
       and Device.cpu == -1 and Device.cuda == 0)

report("Device has no __members__ (it is not an enum type at all), while the four "
       "real enums (Type/SymType/bondType/fermionParity) all do",
       not hasattr(Device, "__members__")
       and hasattr(Type, "__members__") and hasattr(SymType, "__members__")
       and hasattr(bondType, "__members__") and hasattr(fermionParity, "__members__"))

report("Device.getname(-1) (cpu) returns a descriptive string",
       Device.getname(-1) == "cytnx device: CPU")

try:
    Device.getname(99)
    getname_raised = False
except RuntimeError:
    getname_raised = True
report("Device.getname(99) (an invalid device id) raises a catchable RuntimeError "
       "(B4: errors are exceptions, not sentinels)", getname_raised)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    Device.Print_Property()
report("Device.Print_Property()'s output is NOT capturable via Python's "
       "contextlib.redirect_stdout (the binding at cytnx.cpp's mdev.def(\"Print_Property\", "
       "...) has no py::scoped_ostream_redirect guard, unlike Symmetry's "
       "print-via-__repr__ pattern documented in Symmetry.md's P5) -- it writes "
       "directly to the process's real stdout, bypassing Python-level capture",
       buf.getvalue() == "")

report("Device.Ncpus/Device.Ngpus are non-negative ints, baked into the submodule "
       "once at import time (mdev.attr(\"Ncpus\") = ... is a value assignment, not "
       "a property/getter) -- same 'computed once at construction' semantics as "
       "the global C++ cytnx::Device singleton, so this is not itself a parity gap",
       isinstance(Device.Ncpus, int) and Device.Ncpus >= 0
       and isinstance(Device.Ngpus, int) and Device.Ngpus >= 0)

# =========================================================================
# SymType
# =========================================================================

expected_symtype_values = {"U": -1, "Z": 0, "fPar": -2, "fNum": -3}
report("SymType's 4 bound members have the exact integer codes from SymmetryType "
       "(Symmetry.hpp)",
       all(int(getattr(SymType, name)) == val for name, val in expected_symtype_values.items()))

report("SymmetryType::Void (-99, the C++ 'uninitialized' sentinel) has NO Python "
       "binding at all: cytnx.SymType has no Void member",
       not hasattr(SymType, "Void"))

# =========================================================================
# bondType
# =========================================================================

expected_bondtype_values = {"BD_KET": -1, "BD_BRA": 1, "BD_REG": 0, "BD_IN": -1, "BD_OUT": 1}
report("bondType's 5 bound members have the exact integer codes from Bond.hpp",
       all(int(getattr(bondType, name)) == val for name, val in expected_bondtype_values.items()))

report("BD_IN is a value-equal alias for BD_KET, and BD_OUT is a value-equal alias "
       "for BD_BRA (both sides bind two names to the same underlying int)",
       (bondType.BD_IN == bondType.BD_KET) is True
       and (bondType.BD_OUT == bondType.BD_BRA) is True)

report("...but BD_IN and BD_KET are NOT the same Python object (value-equal, "
       "identity-distinct): pybind11 creates two separate enumerator wrapper "
       "objects for the two .value() registrations that share one C++ int",
       (bondType.BD_IN is bondType.BD_KET) is False)

report("bondType.BD_NONE (C++ Bond.hpp: 'BD_NONE = 0, alias for BD_REG') has NO "
       "Python binding: bond_py.cpp's py::enum_<bondType> registers BD_BRA/BD_KET/"
       "BD_REG/BD_IN/BD_OUT but never BD_NONE -- a real C++-source-vs-Python-wheel "
       "member gap",
       not hasattr(bondType, "BD_NONE"))

# =========================================================================
# fermionParity
# =========================================================================

report("fermionParity's 2 members have the exact integer codes from Symmetry.hpp "
       "(enum fermionParity : bool { EVEN = false, ODD = true })",
       int(fermionParity.EVEN) == 0 and int(fermionParity.ODD) == 1)

# --- headline: bool()-truthiness diverges from the declared C++ value for EVEN.
# In C++, `enum fermionParity : bool { EVEN = false, ... }` has EVEN implicitly
# convertible to bool `false` (verified independently by compiling and running
# a standalone C++17 snippet with this exact enum declaration: `(bool)EVEN`
# prints 0, `!EVEN` prints 1). The pybind11 enum wrapper object, by contrast,
# defines no __bool__, so Python's default object truthiness applies: EVERY
# enum instance -- including EVEN -- is truthy.

report("bool(fermionParity.EVEN) is True in Python, even though EVEN's own "
       "declared/underlying value is 0/false (int(EVEN) == 0) -- the pybind11 "
       "enum wrapper has no __bool__, so Python's default 'objects are truthy' "
       "rule applies uniformly to EVEN and ODD alike",
       bool(fermionParity.EVEN) is True and bool(fermionParity.ODD) is True)

report("...confirmed via a fresh, purpose-built truthiness check ('if not EVEN') "
       "matching how C++ code would naturally test a bool-backed enum: Python's "
       "'not EVEN' is False (i.e. EVEN reads as truthy), the opposite of what "
       "the enum's own declared value (0/false) would suggest",
       (not fermionParity.EVEN) is False)

# =========================================================================
# Cross-enum: export_values() namespace pollution + cross-type equality
# =========================================================================

report("export_values() (called on Type/SymType/bondType/fermionParity's "
       "py::enum_ bindings, cytnx.cpp:64 and symmetry_py.cpp:41,45) re-exports "
       "every member as a cytnx-module-level name: cytnx.Double, cytnx.Z, "
       "cytnx.U, cytnx.BD_BRA, cytnx.EVEN all exist at the top level, in "
       "addition to their qualified cytnx.Type.Double / cytnx.SymType.Z / etc. "
       "forms",
       hasattr(cytnx, "Double") and hasattr(cytnx, "Z") and hasattr(cytnx, "U")
       and hasattr(cytnx, "BD_BRA") and hasattr(cytnx, "EVEN")
       and cytnx.Double == Type.Double and cytnx.Z == SymType.Z
       and cytnx.BD_BRA == bondType.BD_BRA and cytnx.EVEN == fermionParity.EVEN)

report("Device does NOT get this treatment (it is a submodule, not a py::enum_, "
       "so nothing calls export_values() on it): cytnx.cpu and cytnx.cuda do "
       "NOT exist at the top level",
       not hasattr(cytnx, "cpu") and not hasattr(cytnx, "cuda"))

# --- cross-enum-type equality: members of four DIFFERENT enum classes that
# happen to share the same underlying int (0) compare equal to one another,
# and hash equal, causing a real dict-key collision across unrelated types.

report("Members of four DIFFERENT enum classes that share the underlying int "
       "value 0 compare equal to one another: SymType.Z == Type.Void == "
       "bondType.BD_REG == fermionParity.EVEN, all True pairwise",
       (SymType.Z == Type.Void) is True
       and (bondType.BD_REG == Type.Void) is True
       and (fermionParity.EVEN == Type.Void) is True)

report("...and they hash equal too, so a dict keyed on one enum type suffers a "
       "silent key collision from an unrelated enum type's same-valued member: "
       "bondType.BD_REG (0) collides with a dict keyed by Type.Void (0)",
       hash(Type.Void) == hash(bondType.BD_REG)
       and (bondType.BD_REG in {Type.Void: "type-void"}) is True)

print("enums probe ok")
