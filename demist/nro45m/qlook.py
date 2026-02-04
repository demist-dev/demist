__all__ = ["otf", "psw"]

# standard library
from collections.abc import Sequence

# dependencies
from .utils import Array, StrPath


def otf(
    log: StrPath,
    /,
    *,
    array: Sequence[Array] | Array = "A1",
) -> None:
    raise NotImplementedError("Quick-look for OTF observations is not yet implemented.")


def psw(
    log: StrPath,
    /,
    *,
    array: Sequence[Array] | Array = "A1",
) -> None:
    raise NotImplementedError("Quick-look for PSW observations is not yet implemented.")
