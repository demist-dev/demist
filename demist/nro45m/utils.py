__all__ = ["DIMS", "Array", "StrPath", "read"]

# standard library
from os import PathLike
from typing import Literal, get_args

# dependencies
import sam45
import numpy as np
import pandas as pd
import xarray as xr
from ..stats import mean

# type hints
Array = Literal[
    # fmt: off
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16",
    "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24",
    "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32",
    # fmt: on
]
StrPath = PathLike[str] | str

# constants
DIMS = "time", "chan"
DATETIME_FORMAT = "%Y%m%d%H%M%S.%f"
TIMEZONE_JST = pd.Timedelta(9, "h")


def read(
    log: StrPath,
    /,
    *,
    array: Array,
    chan_binning: int = 1,
    time_binning: int = 1,
) -> xr.DataArray:
    """Read a SAM45 log to create a DataArray of an array.

    Args:
        log: Path to the SAM45 log.
        array: Array name to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.

    Returns:
        DataArray of the array.

    """
    if array not in get_args(Array):
        raise ValueError(f"Invalid array name: {array!r}")

    # read OBS information
    obs = sam45.read.obs(log)[0]
    obs_index = int(array[1:]) - 1

    # read DAT information
    dat = sam45.read.dat(log)
    dat_index = dat["cary_name"] == array.encode()

    # create DataArray dimensions
    time = pd.Series(dat[dat_index]["cint_sttm"]).astype(np.str_)
    time = time.mask(~time.str.contains(r"\."), time + ".0000")
    time = pd.to_datetime(time, format=DATETIME_FORMAT) - TIMEZONE_JST  # UTC
    chan = np.arange(obs["ichannel"][obs_index])  # zero-based index

    # create DataArray coordinates
    x_0, x_1 = obs["dfqdat_ch"][obs_index] - 1  # zero-based index
    y_0, y_1 = obs["dfqdat_fq"][obs_index] * 1e-9  # Hz to GHz

    array_ = np.full(len(time), array)
    beam = np.full(len(time), obs["imlt_no"][obs_index].astype(np.int64))
    exposure = np.full(len(time), obs["diptim"])  # s
    frequency = (y_0 - y_1) / (x_0 - x_1) * (chan - x_0) + y_0  # GHz
    observation = np.full(len(time), obs["clog_id"].astype(np.str_))
    scan = dat[dat_index]["iline_no"].astype(np.int64)
    state = pd.Series(dat[dat_index]["cscan_type"].astype(np.str_))
    state = state.replace(r"^([A-Z]+).*$", r"\1", regex=True)
    state = state.to_numpy().astype(np.str_)

    # create DataArray data
    att = np.where(state == "R", obs["iary_ifatt"][obs_index], 0)
    data = dat[dat_index]["fary_data"].astype(np.float64)
    data = (data - data[state == "ZERO"]) * (10 ** (att / 10))[:, np.newaxis]

    # create DataArray attributes
    object = str(obs["cobj_name"].astype(np.str_))
    observer = str(obs["cgroup"].astype(np.str_))
    project = str(obs["cproject"].astype(np.str_))

    # create DataArray with time and channel binning
    da = xr.DataArray(
        data=data,
        dims=DIMS,
        coords={
            "array": (DIMS[0], array_),
            "beam": (DIMS[0], beam),
            "chan": (DIMS[1], chan),
            "exposure": (DIMS[0], exposure, {"units": "s"}),
            "frequency": (DIMS[1], frequency, {"units": "GHz"}),
            "observation": (DIMS[0], observation),
            "scan": (DIMS[0], scan),
            "state": (DIMS[0], state),
            "time": (DIMS[0], time),
        },
        attrs={
            "object": object,
            "observer": observer,
            "project": project,
        },
    )
    da = (
        da.sel(time=da.state != "ZERO")
        .groupby("state")
        .apply(
            mean,
            dim={
                DIMS[0]: time_binning,
                DIMS[1]: chan_binning,
            },
        )
    )
    return (
        da.assign_coords(
            beam=da.beam.astype(np.int64),
            chan=da.chan.astype(np.int64),
            scan=da.scan.astype(np.int64),
        )
        .sortby(DIMS[0])
        .sortby(DIMS[1])
    )
