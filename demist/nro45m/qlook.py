__all__ = ["otf", "psw"]

# standard library
from collections.abc import Sequence
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Any

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm
from .models import fit_lowrank, fit_poly, fit_sky, fit_sparse
from .io import DIMS, Array, read
from ..stats import mean

# type hints
FreqRange = tuple[float | None, float | None]  # (GHz, GHz)


def otf(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    arrays: Sequence[Array] = ("A1",),
    chan_binning: int = 8,
    time_binning: int = 5,
    # options for PolyFit analysis
    polyfit_degree: int = 1,
    polyfit_ranges: Sequence[FreqRange] = ((None, None),),
    # options for DE:MIST analysis
    demist_iterations: int = 10,
    lowrank_components: int = 5,
    lowrank_off_only: bool = False,
    lowrank_per_array: bool = True,
    lowrank_per_observation: bool = True,
    sparse_per_array: bool = False,
    sparse_per_observation: bool = False,
    sparse_prefilter: int = 3,
    sparse_threshold: float = 3.0,
    # options for displaying
    progress: bool = True,
    # options for figure saving
    figsize: tuple[float, float] = (10, 5),
    format: str = "pdf",
    **options: Any,
) -> Path:
    """Quick-look at a DE:MIST on-the-fly (OTF) mapping observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        arrays: Array names to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.
        polyfit_degree: Degree of polynomial for PoltFit analysis.
        polyfit_ranges: Frequency ranges to use for PolyFit analysis.
        demist_iterations: Number of iterations for DE:MIST analysis.
        lowrank_components: Number of components for low-rank model fitting.
        lowrank_off_only: Whether to use only OFF samples for low-rank model fitting.
        lowrank_per_array: Whether to fit low-rank model per array.
        lowrank_per_observation: Whether to fit low-rank model per observation.
        sparse_per_array: Whether to fit sparse model per array.
        sparse_per_observation: Whether to fit sparse model per observation.
        sparse_prefilter: Size of median filter for sparse model fitting.
        sparse_threshold: Absolute S/N threshold for sparse model fitting.
        progress: Whether to display progress bar.
        figsize: Size of the saved figure.
        format: File format of the saved figure.
        **options: Additional options for figure saving.

    Returns:
        Absolute path to the saved quick-look figure.
        If multiple logs are given, the last log's name will be used for saving.

    """
    raise NotImplementedError("This command is not yet implemented.")


def psw(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    arrays: Sequence[Array] = ("A1",),
    chan_binning: int = 8,
    time_binning: int = 5,
    # options for PolyFit analysis
    polyfit_degree: int = 1,
    polyfit_ranges: Sequence[FreqRange] = ((None, None),),
    # options for DE:MIST analysis
    demist_iterations: int = 10,
    lowrank_components: int = 5,
    lowrank_off_only: bool = False,
    lowrank_per_array: bool = True,
    lowrank_per_observation: bool = True,
    sparse_per_array: bool = False,
    sparse_per_observation: bool = False,
    sparse_prefilter: int = 3,
    sparse_threshold: float = 3.0,
    # options for displaying
    progress: bool = True,
    # options for figure saving
    figsize: tuple[float, float] = (10, 5),
    format: str = "pdf",
    **options: Any,
) -> Path:
    """Quick-look at a DE:MIST position-switching (PSW) observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        arrays: Array names to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.
        polyfit_degree: Degree of polynomial for PoltFit analysis.
        polyfit_ranges: Frequency ranges to use for PolyFit analysis.
        demist_iterations: Number of iterations for DE:MIST analysis.
        lowrank_components: Number of components for low-rank model fitting.
        lowrank_off_only: Whether to use only OFF samples for low-rank model fitting.
        lowrank_per_array: Whether to fit low-rank model per array.
        lowrank_per_observation: Whether to fit low-rank model per observation.
        sparse_per_array: Whether to fit sparse model per array.
        sparse_per_observation: Whether to fit sparse model per observation.
        sparse_prefilter: Size of median filter for sparse model fitting.
        sparse_threshold: Absolute S/N threshold for sparse model fitting.
        progress: Whether to display progress bar.
        figsize: Size of the saved figure.
        format: File format of the saved figure.
        **options: Additional options for figure saving.

    Returns:
        Absolute path to the saved quick-look figure.
        If multiple logs are given, the last log's name will be used for saving.

    """
    # create concatenated DataArray ready for analysis
    Ps: list[xr.DataArray] = []

    with tqdm(
        desc="Reading SAM45 logs/arrays",
        disable=not progress,
        total=len(logs) * len(arrays),
    ) as bar:
        for log, array in product(logs, arrays):
            P = read(
                log,
                array,
                time_binning=time_binning,
                chan_binning=chan_binning,
            )

            # swap ON/OFF for non-central beams
            is_on = (P.state == "ON") & (P.beam != 1)
            is_off = (P.state == "OFF") & (P.beam != 1)
            P.state[is_on] = "OFF"
            P.state[is_off] = "ON"

            # assign calibrator (R) as a coordinate
            calibrator = (
                P.sel(time=P.state == "R")
                .groupby("scan")
                .apply(mean, dim=DIMS[0])
                .swap_dims({"scan": DIMS[0]})
                .interp_like(P, kwargs={"fill_value": "extrapolate"})
                .reset_coords(drop=True)
                .drop_attrs()
            )
            P = P.assign_coords(calibrator=calibrator)
            Ps.append(P.sel(time=P.state.isin(["ON", "OFF"])))
            bar.update()

    P = xr.concat(Ps, dim=DIMS[0]).sortby(DIMS[0])

    # run PolyFit (conventional) analysis
    P_sky = fit_sky(P)
    T = P.temperature * (P - P_sky.data) / (P.calibrator - P_sky.data)
    T_poly = fit_poly(
        T,
        fit_degree=polyfit_degree,
        fit_ranges=polyfit_ranges,
        fit_per_array=True,
        fit_per_observation=True,
    )
    T_polyfit = T - T_poly.data

    # run DE:MIST (proposed) analysis
    X: xr.DataArray = np.log(-(P - P.calibrator.data) / P.temperature)
    X_sparse = xr.zeros_like(X)

    with tqdm(
        desc="Running DE:MIST analysis",
        disable=not progress,
        total=demist_iterations,
    ) as bar:
        for _ in range(demist_iterations):
            X_lowrank = fit_lowrank(
                X - X_sparse,
                fit_components=lowrank_components,
                fit_off_only=lowrank_off_only,
                fit_per_array=lowrank_per_array,
                fit_per_observation=lowrank_per_observation,
            )
            X_sparse = fit_sparse(
                X - X_lowrank,
                fit_per_array=sparse_per_array,
                fit_per_observation=sparse_per_observation,
                fit_prefilter=sparse_prefilter,
                fit_threshold=sparse_threshold,
            )
            bar.update()

    T_demist = P.temperature * (1 - np.exp(X - X_lowrank))

    # calculate integrate spectra
    T_sys = P.temperature / (P.calibrator / P_sky - 1)
    spec_polyfit = to_spectrum(T_polyfit, T_sys, alpha=np.sqrt(2))
    spec_demist = to_spectrum(T_demist, T_sys, alpha=1)

    # calculate achieved scaling factor
    a_polyfit = spec_polyfit.sel(chan=T_poly.fit_ranges).std() / spec_demist.noise
    a_demist = spec_demist.sel(chan=T_poly.fit_ranges).std() / spec_demist.noise

    # plot quick-look result
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    ax = axes[0]
    spec_polyfit.swap_dims(chan="frequency").plot.step(
        ax=ax,
        label=rf"PolyFit ($\alpha$ = {a_polyfit.mean():.2f} $\pm$ {a_polyfit.std():.2f})",
    )
    ax.fill_between(
        spec_polyfit.frequency,
        -spec_polyfit.noise,
        +spec_polyfit.noise,
        color="gray",
        alpha=0.25,
        ec="none",
        label=r"Expected noise level ($\alpha = \sqrt{2}$)",
    )

    ax = axes[1]
    spec_demist.swap_dims(chan="frequency").plot.step(
        ax=ax,
        label=rf"DE:MIST ($\alpha$ = {a_demist.mean():.2f} $\pm$ {a_demist.std():.2f})",
    )
    ax.fill_between(
        spec_demist.frequency,
        -spec_demist.noise,
        +spec_demist.noise,
        color="gray",
        alpha=0.25,
        ec="none",
        label=r"Expected noise level ($\alpha = 1$)",
    )

    for ax in axes:
        ax.margins(x=0.0)
        ax.legend()
        ax.grid()

    fig.tight_layout()

    # save quick-look result
    fig.savefig(name := Path(log).with_suffix(f".qlook.psw.{format}").name, **options)
    return Path(name).resolve()


def to_spectrum(
    T: xr.DataArray,
    T_sys: xr.DataArray,
    /,
    *,
    alpha: float = 1.0,
    cumulative: bool = False,
) -> xr.DataArray:
    """Create integrated spectrum of calibrated temperature.

    Args:
        T: DataArray of calibrated temperature.
        T_sys: DataArray of system noise temperature.
        alpha: Scaling factor for noise level.
        cumulative: Whether to calculate cumulative spectrum.

    Returns:
        Integrated spectrum with expected noise level and effective number of samples.

    """
    T = T.sel(time=T.state == "ON")
    T_sys = T_sys.sel(time=T_sys.state == "ON")

    # calculate expected noise level per time sample
    sigma = alpha * T_sys * (T.exposure * T.width * 1e9) ** -0.5
    weight = sigma**-2

    # calculate expected noise level of integrated spectrum
    if cumulative:
        signal = (T * weight).cumsum(DIMS[0]) / weight.cumsum(DIMS[0])
        noise = (sigma**-2).cumsum(DIMS[0]) ** -0.5
        n_eff = weight.cumsum(DIMS[0]) ** 2 / (weight**2).cumsum(DIMS[0])
    else:
        signal = (T * weight).sum(DIMS[0]) / weight.sum(DIMS[0])
        noise = (sigma**-2).sum(DIMS[0]) ** -0.5
        n_eff = weight.sum(DIMS[0]) ** 2 / (weight**2).sum(DIMS[0])

    return signal.assign_coords(
        noise=noise.assign_attrs(long_name="Expected noise level", units=T.units),
        n_eff=n_eff.assign_attrs(long_name="Effective number of samples"),
    ).assign_attrs(
        long_name=r"$T_{\mathrm{A}}^{\ast}$",
        units=T.units,
    )
