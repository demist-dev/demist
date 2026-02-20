__all__ = ["Array", "read"]

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

# constants
DATETIME_FORMAT = "%Y%m%d%H%M%S.%f"
TIMEZONE_JST = pd.Timedelta(9, "h")


def read(
    log: PathLike[str] | str,
    array: Array,
    /,
    *,
    chan_binning: int = 1,
    time_binning: int = 1,
    chan_flip: bool = False,
) -> xr.DataArray:
    """Read a SAM45 log to create a DataArray of an array.

    Args:
        log: Path to the SAM45 log.
        array: Array name to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.
        chan_flip: Whether to flip the channel order.
            Note that this is a temporary workaround for reading SAM45 logs
            with flipped channel order and will be removed in future versions.

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

    # calculate DataArray dimensions
    time = pd.Series(dat[dat_index]["cint_sttm"]).astype(np.str_)
    time = time.mask(~time.str.contains(r"\."), time + ".0000")
    time = pd.to_datetime(time, format=DATETIME_FORMAT) - TIMEZONE_JST  # UTC
    chan = np.arange(obs["ichannel"][obs_index])  # zero-based index

    # calculate DataArray coordinates
    x_0, x_1 = obs["dfqdat_ch"][obs_index] - 1  # zero-based index
    y_0, y_1 = obs["dfqdat_fq"][obs_index] * 1e-9  # GHz
    frequency = (y_0 - y_1) / (x_0 - x_1) * (chan - x_0) + y_0  # GHz
    state = pd.Series(dat[dat_index]["cscan_type"].astype(np.str_))
    state = state.replace(r"^([A-Z]+).*$", r"\1", regex=True)
    state = state.to_numpy().astype(np.str_)

    # calculate DataArray data
    att = np.where(state == "R", obs["iary_ifatt"][obs_index], 0)  # dB
    data = dat[dat_index]["fary_data"].astype(np.float64)
    data = (data - data[state == "ZERO"]) * (10 ** (att / 10))[:, np.newaxis]

    if chan_flip:
        data = data[:, ::-1]

    # create DataArray
    dataarray = xr.DataArray(
        data=data,
        dims=("time", "chan"),
        coords={
            "array": (
                "time",
                np.full(len(time), array).astype(np.str_),
                {"long_name": "Array name"},
            ),
            "beam": (
                "time",
                np.full(len(time), obs["imlt_no"][obs_index].astype(np.int64)),
                {"long_name": "Beam number"},
            ),
            "chan": (
                "chan",
                chan,
                {"long_name": "Channel number"},
            ),
            "exposure": (
                "time",
                np.full(len(time), obs["diptim"]).astype(np.float64),
                {"long_name": "Exposure time", "units": "s"},
            ),
            "frequency": (
                "chan",
                frequency.astype(np.float64),
                {"long_name": "Rest frequency", "units": "GHz"},
            ),
            "observation": (
                "time",
                np.full(len(time), obs["clog_id"].astype(np.str_)),
                {"long_name": "Observation ID"},
            ),
            "pressure": (
                "time",
                dat[dat_index]["dweather"][:, 1].astype(np.float64),
                {"long_name": "Atmospheric pressure", "units": "hPa"},
            ),
            "scan": (
                "time",
                dat[dat_index]["iline_no"].astype(np.int64),
                {"long_name": "Scan number"},
            ),
            "state": (
                "time",
                state,
                {"long_name": "Scan state"},
            ),
            "temperature": (
                "time",
                dat[dat_index]["dweather"][:, 0].astype(np.float64) + 273.15,
                {"long_name": "Atmospheric temperature", "units": "K"},
            ),
            "time": (
                "time",
                time,
                {"long_name": "Observed time in UTC"},
            ),
            "vapor_pressure": (
                "time",
                dat[dat_index]["dweather"][:, 2].astype(np.float64),
                {"long_name": "Vapor pressure", "units": "hPa"},
            ),
            "width": (
                "chan",
                np.full(len(chan), obs["dbechwid"][obs_index]) * 1e-9,
                {"long_name": "Channel width", "units": "GHz"},
            ),
            "wind_direction": (
                "time",
                dat[dat_index]["dweather"][:, 4].astype(np.float64) % 360,
                {"long_name": "Wind direction", "units": "deg"},
            ),
            "wind_speed": (
                "time",
                dat[dat_index]["dweather"][:, 3].astype(np.float64),
                {"long_name": "Wind speed", "units": "m/s"},
            ),
        },
        attrs={
            "object": str(obs["cobj_name"].astype(np.str_)),
            "observer": str(obs["cgroup"].astype(np.str_)),
            "project": str(obs["cproject"].astype(np.str_)),
        },
    )

    # perform time and channel binning
    dataarray = (
        dataarray.sel(time=dataarray.state != "ZERO")
        .groupby("state")
        .apply(
            mean,
            dim={
                "time": time_binning,
                "chan": chan_binning,
            },
        )
    )
    return (
        dataarray.assign_coords(
            beam=dataarray.beam.astype(np.int64),
            chan=dataarray.chan.astype(np.int64),
            exposure=dataarray.exposure * time_binning,
            scan=dataarray.scan.astype(np.int64),
            width=dataarray.width * chan_binning,
        )
        .sortby("time")
        .sortby("chan")
    )
