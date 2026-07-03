"""Helpers shared by every behavioral probe. Import from probes/<Unit>.py."""
import cytnx


def report(claim: str, ok: bool) -> None:
    """Print a PASS/FAIL line and raise AssertionError on failure."""
    print(f"[{'PASS' if ok else 'FAIL'}] {claim}")
    assert ok, claim


def returns_view(make, derive, mutate, read) -> bool:
    """Detect view vs. copy semantics of a derivation.

    make()          -> a fresh source object
    derive(source)  -> a new handle produced by the method under test
    mutate(handle)  -> mutate the derived handle in place
    read(source)    -> a comparable snapshot of the source's data

    Returns True if the mutation is visible in the source (view), False if
    the source is unchanged (copy).
    """
    src = make()
    before = read(src)
    handle = derive(src)
    mutate(handle)
    after = read(src)
    return after != before
