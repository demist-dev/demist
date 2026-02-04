__all__ = ["cli", "nro45m", "stats"]
__version__ = "0.1.0"

# dependencies
from fire import Fire
from . import nro45m, stats


def cli() -> None:
    """Run the command line interface."""
    Fire(
        {
            "nro45m": {
                "qlook": {
                    "otf": nro45m.qlook.otf,
                    "psw": nro45m.qlook.psw,
                },
            }
        }
    )
