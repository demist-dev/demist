__all__ = ["cli", "nro45m", "stats"]
__version__ = "0.4.0"

# dependencies
from logging import basicConfig
from fire import Fire
from . import nro45m, stats


def cli() -> None:
    """Run the command line interface."""
    basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s %(name)s %(funcName)s %(levelname)s] %(message)s",
    )

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
