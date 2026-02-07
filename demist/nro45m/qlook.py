__all__ = ["otf", "psw"]

# standard library
from collections.abc import Sequence
from itertools import product
from os import PathLike
from pathlib import Path
from warnings import catch_warnings, simplefilter

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from .models import fit_background, fit_lowrank, fit_polynomial, fit_sparse
from .io import Array, read
from ..stats import mean

# type hints
Range = tuple[float | None, float | None]


def otf(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    arrays: Sequence[Array] = ("A1",),
    chan_binning: int = 8,
    time_binning: int = 5,
    # options for the PolyFit analysis
    polyfit_degree: int = 1,
    polyfit_ranges: Sequence[Range] = ((None, None),),
    # options for the DE:MIST analysis
    demist_iterations: int = 100,
    demist_threshold: float = 1e-4,
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
    # options for saving the quick-look results
    figsize: tuple[float, float] = (10, 5),
    simple: bool = True,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
) -> Path:
    """Quick-look at a DE:MIST on-the-fly (OTF) mapping observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        arrays: Array names to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.
        polyfit_degree: Degree of polynomial for the PolyFit analysis.
        polyfit_ranges: Frequency ranges in GHz to use for the PolyFit analysis.
        demist_iterations: Number of maximum iterations for the DE:MIST analysis.
        demist_threshold: Convergence threshold for the DE:MIST analysis.
        lowrank_components: Number of components for low-rank model fitting.
        lowrank_off_only: Whether to use only OFF samples for low-rank model fitting.
        lowrank_per_array: Whether to fit low-rank model per array.
        lowrank_per_observation: Whether to fit low-rank model per observation.
        sparse_per_array: Whether to fit sparse model per array.
        sparse_per_observation: Whether to fit sparse model per observation.
        sparse_prefilter: Size of median filter for sparse model fitting.
        sparse_threshold: Absolute S/N threshold for sparse model fitting.
        progress: Whether to display progress bar.
        figsize: Size of the saved quick-look results.
        simple: Whether not to save miscellaneous information.
        xlim: X-axis limits for the saved quick-look results.
        ylim: Y-axis limits for the saved quick-look results.

    Returns:
        Absolute path to the saved quick-look results.
        If multiple logs are given, the last log's name will be used for saving.

    """
    raise NotImplementedError("This command is not yet implemented.")


def psw(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    arrays: Sequence[Array] = ("A1",),
    chan_binning: int = 8,
    time_binning: int = 5,
    # options for the PolyFit analysis
    polyfit_degree: int = 1,
    polyfit_ranges: Sequence[Range] = ((None, None),),
    # options for the DE:MIST analysis
    demist_iterations: int = 100,
    demist_threshold: float = 1e-4,
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
    # options for saving the quick-look results
    figsize: tuple[float, float] = (10, 5),
    simple: bool = True,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
) -> Path:
    """Quick-look at a DE:MIST position-switching (PSW) observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        arrays: Array names to read (A1-A32).
        chan_binning: Number of channels to bin together.
        time_binning: Number of time samples to bin together.
        polyfit_degree: Degree of polynomial for the PolyFit analysis.
        polyfit_ranges: Frequency ranges in GHz to use for the PolyFit analysis.
        demist_iterations: Number of maximum iterations for the DE:MIST analysis.
        demist_threshold: Convergence threshold for the DE:MIST analysis.
        lowrank_components: Number of components for low-rank model fitting.
        lowrank_off_only: Whether to use only OFF samples for low-rank model fitting.
        lowrank_per_array: Whether to fit low-rank model per array.
        lowrank_per_observation: Whether to fit low-rank model per observation.
        sparse_per_array: Whether to fit sparse model per array.
        sparse_per_observation: Whether to fit sparse model per observation.
        sparse_prefilter: Size of median filter for sparse model fitting.
        sparse_threshold: Absolute S/N threshold for sparse model fitting.
        progress: Whether to display progress bar.
        figsize: Size of the saved quick-look results.
        simple: Whether not to save miscellaneous information.
        xlim: X-axis limits for the saved quick-look results.
        ylim: Y-axis limits for the saved quick-look results.

    Returns:
        Absolute path to the saved quick-look results.
        If multiple logs are given, the last log's name will be used for saving.

    """
    # Read SAM45 logs and arrays
    with tqdm(
        desc="Reading SAM45 logs/arrays",
        disable=not progress,
        total=len(logs) * len(arrays),
    ) as bar:
        Ps: list[xr.DataArray] = []

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

            Ps.append(P)
            bar.update(1)

        P = xr.concat(Ps, dim="time").sortby("time")

    # run PolyFit (conventional) analysis
    with tqdm(
        desc="Running PolyFit analysis",
        disable=not progress,
        total=1,
    ) as bar:
        P_cal = fit_background(P, "R").sel(time=P.state.isin(["ON", "OFF"]))
        P_sky = fit_background(P, "OFF").sel(time=P.state.isin(["ON", "OFF"]))
        P = P.sel(time=P.state.isin(["ON", "OFF"]))

        T: xr.DataArray = P.temperature * (P - P_sky) / (P_cal - P_sky)
        T_poly = fit_polynomial(
            T,
            fit_degree=polyfit_degree,
            fit_ranges=polyfit_ranges,
            fit_per_array=True,
            fit_per_observation=True,
        )
        bar.update(1)

    # run DE:MIST (proposed) analysis
    with tqdm(
        desc="Running DE:MIST analysis",
        disable=not progress,
        total=demist_iterations,
    ) as bar:
        X: xr.DataArray = np.log(-(P - P_cal) / P.temperature)
        X_sparse = xr.zeros_like(X)

        for n in range(demist_iterations):
            X_lowrank = fit_lowrank(
                X - X_sparse,
                fit_components=lowrank_components,
                fit_off_only=lowrank_off_only,
                fit_per_array=lowrank_per_array,
                fit_per_observation=lowrank_per_observation,
            )

            X_sparse_ = X_sparse
            X_sparse = fit_sparse(
                X - X_lowrank,
                fit_per_array=sparse_per_array,
                fit_per_observation=sparse_per_observation,
                fit_prefilter=sparse_prefilter,
                fit_threshold=sparse_threshold,
            )

            with catch_warnings():
                simplefilter("ignore", category=RuntimeWarning)
                eps = float(((X_sparse - X_sparse_) / X_sparse_).mean())

            if abs(eps) < demist_threshold:
                bar.update(demist_iterations - n)
                break
            else:
                bar.update(1)

    # calculate final spectra
    with tqdm(
        desc="Calculating final spectra",
        disable=not progress,
        total=1,
    ) as bar:
        T_polyfit: xr.DataArray = T - T_poly
        T_polyfit = T_polyfit.assign_attrs(long_name=r"$T_{\mathrm{A}}^{\ast}$")
        T_demist: xr.DataArray = P.temperature * (1 - np.exp(X - X_lowrank))
        T_demist = T_demist.assign_attrs(long_name=r"$T_{\mathrm{A}}^{\ast}$")
        T_sys: xr.DataArray = P.temperature / (P_cal / P_sky - 1)
        T_sys = T_sys.assign_attrs(long_name=r"$T_{\mathrm{sys}}$")

        spec_polyfit = to_spectrum(
            T_polyfit,
            T_sys,
            alpha=np.sqrt(2),
            cumulative=False,
            shuffle=False,
        )
        spec_demist = to_spectrum(
            T_demist,
            T_sys,
            alpha=1.0,
            cumulative=False,
            shuffle=False,
        )
        cumspec_polyfit = to_spectrum(
            T_polyfit,
            T_sys,
            alpha=np.sqrt(2),
            cumulative=True,
            shuffle=True,
        )
        cumspec_demist = to_spectrum(
            T_demist,
            T_sys,
            alpha=1.0,
            cumulative=True,
            shuffle=True,
        )
        bar.update(1)

    # plot and save results
    with (
        tqdm(
            desc="Plotting/saving results",
            disable=not progress,
            total=1 if simple else 2,
        ) as bar,
        PdfPages(name := Path(log).name + f".qlook.psw.pdf") as pdf,
    ):
        pdf.savefig(
            plot_integrated_info(
                figsize=figsize,
                horizontal=True,
                spec_demist=spec_demist,
                spec_polyfit=spec_polyfit,
                xlim=xlim,
                ylim=ylim,
            )
        )
        pdf.savefig(
            plot_integrated_info(
                figsize=figsize,
                horizontal=False,
                spec_demist=spec_demist,
                spec_polyfit=spec_polyfit,
                xlim=xlim,
                ylim=ylim,
            )
        )
        bar.update(1)

        if simple:
            return Path(name).resolve()

        pdf.savefig(
            plot_cumulative_info(
                cumspec_demist=cumspec_demist,
                cumspec_polyfit=cumspec_polyfit,
                figsize=figsize,
            )
        )

        for fig in plot_timewise_info(
            figsize=figsize,
            T_demist=T_demist,
            T_polyfit=T_polyfit,
            T_sys=T_sys,
        ):
            pdf.savefig(fig)

        pdf.savefig(
            plot_chanwise_info(
                figsize=figsize,
                T_demist=T_demist,
                T_polyfit=T_polyfit,
            )
        )
        bar.update(1)
        return Path(name).resolve()


def cov(da: xr.DataArray, /) -> xr.DataArray:
    """Calculate the normalized covariance matrix."""
    da = da.swap_dims(chan="frequency")
    cov = np.cov(((da - da.mean("time")) / da.std("time")).T)
    cov[np.diag_indices_from(cov)] = np.nan

    return xr.DataArray(
        cov,
        dims=("frequency_0", "frequency_1"),
        coords={
            "frequency_0": da.frequency.rename(frequency="frequency_0"),
            "frequency_1": da.frequency.rename(frequency="frequency_1"),
        },
        attrs={
            "long_name": "Normalized covariance",
        },
    )


def edgena(da: xr.DataArray, /) -> xr.DataArray:
    """Set the edge time samples to N/A (for plotting)."""
    da.loc[{"time": da.time[[0, -1]]}] = np.nan
    return da


def plot_chanwise_info(
    *,
    T_demist: xr.DataArray,
    T_polyfit: xr.DataArray,
    figsize: tuple[float, float],
) -> Figure:
    """Plot channel-wise information (covariance matrices)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    ax = axes[0]
    cov_polyfit = cov(T_polyfit)
    cov_polyfit.plot.pcolormesh(
        ax=ax,
        cmap="coolwarm",
        rasterized=True,
        vmin=-3.0 * (sigma := float(cov_polyfit.std())),
        vmax=+3.0 * sigma,
    )
    ax.set_title("Normalized covariance (PolyFit)")

    ax = axes[1]
    cov_demist = cov(T_demist)
    cov_demist.plot.pcolormesh(
        ax=ax,
        cmap="coolwarm",
        rasterized=True,
        vmin=-3.0 * sigma,
        vmax=+3.0 * sigma,
    )
    ax.set_title("Normalized covariance (DE:MIST)")

    fig.tight_layout()
    return fig


def plot_cumulative_info(
    *,
    cumspec_demist: xr.DataArray,
    cumspec_polyfit: xr.DataArray,
    figsize: tuple[float, float],
) -> Figure:
    """Plot cumulative information (cumulative noise level and maximum S/N)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fit_ranges = cumspec_polyfit.fit_ranges

    ax = axes[0]
    ax.plot(
        cumspec_polyfit.sel(chan=fit_ranges).exposure.mean("chan"),
        cumspec_polyfit.sel(chan=fit_ranges).std("chan"),
        label="PolyFit",
    )
    ax.plot(
        cumspec_demist.sel(chan=fit_ranges).exposure.mean("chan"),
        cumspec_demist.sel(chan=fit_ranges).std("chan"),
        label="DE:MIST",
    )
    ax.set_title("Cumulative noise level")
    ax.set_xlabel("Effective exposure time [s]")
    ax.set_ylabel("Achieved noise level [K]")

    ax = axes[1]
    ax.plot(
        cumspec_polyfit.sel(chan=fit_ranges).exposure.mean("chan"),
        cumspec_polyfit.max("chan") / cumspec_polyfit.sel(chan=fit_ranges).std("chan"),
        label="PolyFit",
    )
    ax.plot(
        cumspec_demist.sel(chan=fit_ranges).exposure.mean("chan"),
        cumspec_demist.max("chan") / cumspec_demist.sel(chan=fit_ranges).std("chan"),
        label="DE:MIST",
    )
    ax.set_title("Maximum signal-to-noise ratio")
    ax.set_xlabel("Effective exposure time [s]")
    ax.set_ylabel("Maximum signal-to-noise ratio [1]")
    ax.set_ylim(1.0, None)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.margins(x=0.0)
        ax.legend()
        ax.grid()

    fig.tight_layout()
    return fig


def plot_integrated_info(
    *,
    figsize: tuple[float, float],
    horizontal: bool = True,
    spec_demist: xr.DataArray,
    spec_polyfit: xr.DataArray,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
) -> Figure:
    """Plot integrated information (spectra with noise levels)."""
    if horizontal:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    fit_ranges = spec_polyfit.fit_ranges
    a_polyfit = spec_polyfit.sel(chan=fit_ranges).std() / spec_demist.noise
    a_demist = spec_demist.sel(chan=fit_ranges).std() / spec_demist.noise

    ax = axes[0]
    spec_polyfit.swap_dims(chan="frequency").plot.step(
        ax=ax,
        label=(
            # fmt: off
            "PolyFit "
            rf"($\alpha$ = {a_polyfit.mean():.2f} $\pm$ {a_polyfit.std():.2f})",
            # fmt: on
        ),
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
        label=(
            # fmt: off
            "DE:MIST "
            rf"($\alpha$ = {a_demist.mean():.2f} $\pm$ {a_demist.std():.2f})"
            # fmt: on
        ),
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
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        ax.grid()

    fig.tight_layout()
    return fig


def plot_timewise_info(
    *,
    figsize: tuple[float, float],
    T_demist: xr.DataArray,
    T_polyfit: xr.DataArray,
    T_sys: xr.DataArray,
) -> list[Figure]:
    """Plot time-wise information (time-frequency plots)."""
    figs: list[Figure] = []

    for group in T_polyfit.groupby("array"):
        fig, ax = plt.subplots(figsize=figsize)

        array, T_polyfit_ = group
        T_polyfit_ = T_polyfit_.groupby("observation").apply(edgena)
        T_polyfit_.swap_dims(chan="frequency").T.plot.pcolormesh(
            ax=ax,
            cmap="coolwarm",
            rasterized=True,
            vmin=-5.0 * (sigma := float(T_polyfit.std())),
            vmax=+5.0 * sigma,
        )
        ax.set_title(r"$T_{\mathrm{A}}^{\ast}$ " f"(PolyFit, {array})")
        fig.tight_layout()
        figs.append(fig)

    for group in T_demist.groupby("array"):
        fig, ax = plt.subplots(figsize=figsize)

        array, T_demist_ = group
        T_demist_ = T_demist_.groupby("observation").apply(edgena)
        T_demist_.swap_dims(chan="frequency").T.plot.pcolormesh(
            ax=ax,
            cmap="coolwarm",
            rasterized=True,
            vmin=-5.0 * sigma,
            vmax=+5.0 * sigma,
        )
        ax.set_title(r"$T_{\mathrm{A}}^{\ast}$ " f"(DE:MIST, {array})")
        fig.tight_layout()
        figs.append(fig)

    for group in T_sys.groupby("array"):
        fig, ax = plt.subplots(figsize=figsize)

        array, T_sys_ = group
        T_sys_ = T_sys_.groupby("observation").apply(edgena)
        T_sys_.swap_dims(chan="frequency").T.plot.pcolormesh(
            ax=ax,
            cmap="inferno",
            rasterized=True,
            vmin=float(T_sys.min()),
            vmax=float(T_sys.max()),
        )
        ax.set_title(r"$T_{\mathrm{sys}}$ " f"({array})")
        fig.tight_layout()
        figs.append(fig)

    return figs


def to_spectrum(
    T: xr.DataArray,
    T_sys: xr.DataArray,
    /,
    *,
    alpha: float = 1.0,
    cumulative: bool = False,
    shuffle: bool = True,
) -> xr.DataArray:
    """Create integrated spectrum of calibrated temperature.

    Args:
        T: DataArray of calibrated antenna temperature.
        T_sys: DataArray of system noise temperature.
        alpha: Scaling factor for noise level.
        cumulative: Whether to calculate cumulative spectrum.
        shuffle: Whether to shuffle time samples before integration.

    Returns:
        Integrated spectrum with expected noise level and effective exposure time.

    """
    T = T.sel(time=T.state == "ON")
    T_sys = T_sys.sel(time=T_sys.state == "ON")

    if shuffle:
        np.random.seed(0)
        index = np.random.permutation(np.arange(T.sizes["time"]))
        T = T.isel(time=index)
        T_sys = T_sys.isel(time=index)

    # calculate expected noise level per time sample
    sigma = alpha * T_sys * (T.exposure * T.width * 1e9) ** -0.5
    weight = sigma**-2

    # calculate expected noise level of integrated spectrum
    if cumulative:
        exposure = weight.cumsum("time") ** 2 / (weight**2 / T.exposure).cumsum("time")
        signal = (T * weight).cumsum("time") / weight.cumsum("time")
        noise = (sigma**-2).cumsum("time") ** -0.5
    else:
        exposure = weight.sum("time") ** 2 / (weight**2 / T.exposure).sum("time")
        signal = (T * weight).sum("time") / weight.sum("time")
        noise = (sigma**-2).sum("time") ** -0.5

    return signal.assign_coords(
        exposure=exposure.assign_attrs(long_name="Effective exposure time"),
        noise=noise.assign_attrs(long_name="Expected noise level", units=T.units),
    ).assign_attrs(
        long_name=r"$T_{\mathrm{A}}^{\ast}$",
        units=T.units,
    )
