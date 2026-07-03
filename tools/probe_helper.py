"""Helpers shared by every behavioral probe. Import from probes/<Unit>.py."""
import cytnx


def report(claim: str, ok: bool) -> None:
    """Print a PASS/FAIL line and raise AssertionError on failure."""
    print(f"[{'PASS' if ok else 'FAIL'}] {claim}")
    assert ok, claim


def mutates_alias(make, mutate, read) -> bool:
    """make()->obj, mutate(obj) in place, read(obj)->value. Returns True if an
    aliased handle observes the mutation (view), False if independent (copy)."""
    obj = make()
    alias = obj
    mutate(alias)
    return read(obj) == read(alias)


def is_view(make, mutate) -> bool:
    """True if mutating a value obtained via make() through mutate() is visible
    on an alias of the same handle (i.e. make()/mutate() operate by reference).
    Convenience wrapper around mutates_alias with an identity read."""
    return mutates_alias(make, mutate, lambda obj: obj)


def assert_inplace(obj, method_name: str) -> None:
    """Call obj.<method_name>() and assert it returns obj itself (in-place op)."""
    method = getattr(obj, method_name)
    result = method()
    assert result is obj, f"{method_name} did not return self (not in-place)"
