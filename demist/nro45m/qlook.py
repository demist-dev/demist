__all__ = ["otf", "psw"]

# standard library
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import product
from logging import DEBUG, getLogger
from os import PathLike
from pathlib import Path
from warnings import catch_warnings, simplefilter

# dependencies
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from ndtools import Any, Range as NDRange
from tqdm import tqdm
from .models import (
    fit_background,
    fit_integration,
    fit_lowrank,
    fit_polynomial,
    fit_sparse,
)
from .io import Array, read
from .. import __version__ as demist_version

# type hints
Range = tuple[float | None, float | None]

# constants
COLOR_DEMIST = "#1e50a2"  # https://www.colordic.org/colorsample/2069
COLOR_POLYFIT = "#a22041"  # https://www.colordic.org/colorsample/2005
LOGGER = getLogger(__name__)


@contextmanager
def set_logger(debug: bool, /) -> Iterator[None]:
    """Temporarily set the level of the module logger."""
    level = LOGGER.level

    if debug:
        LOGGER.setLevel(DEBUG)

    try:
        yield
    finally:
        LOGGER.setLevel(level)


def otf(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    analysis_ranges: Sequence[Range] = ((None, None),),
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
    # options for saving the quick-look results
    figsize: tuple[float, float] = (10, 5),
    overwrite: bool = False,
    detailed: bool = False,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
    # options for displaying
    debug: bool = False,
    progress: bool = True,
) -> Path:
    """Quick-look at a DE:MIST on-the-fly (OTF) mapping observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        analysis_ranges: Frequency ranges in GHz to use for the whole analysis.
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
        figsize: Size of the saved quick-look results.
        detailed: Whether to save detailed information.
        overwrite: Whether to overwrite existing quick-look results.
        xlim: X-axis limits for the saved quick-look results.
        ylim: Y-axis limits for the saved quick-look results.
        debug: Whether to display debug information.
        progress: Whether to display progress bar.

    Returns:
        Absolute path to the saved quick-look results.
        If multiple logs are given, the first log's name will be used for saving.

    """
    with set_logger(debug):
        for key, val in (params := locals().copy()).items():
            LOGGER.debug(f"{key}: {val!r}")

    raise NotImplementedError("This command is not yet implemented.")


def psw(
    *logs: PathLike[str] | str,
    # options for reading SAM45 logs
    analysis_ranges: Sequence[Range] = ((None, None),),
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
    # options for saving the quick-look results
    figsize: tuple[float, float] = (10, 5),
    detailed: bool = False,
    overwrite: bool = False,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
    # options for displaying
    debug: bool = False,
    progress: bool = True,
    # options for workarounds
    chan_flip: bool = False,
) -> Path:
    """Quick-look at a DE:MIST position-switching (PSW) observation.

    Args:
        *logs: Path(s) to SAM45 log(s).
        analysis_ranges: Frequency ranges in GHz to use for the whole analysis.
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
        figsize: Size of the saved quick-look results.
        detailed: Whether to save detailed information.
        overwrite: Whether to overwrite existing quick-look results.
        xlim: X-axis limits for the saved quick-look results.
        ylim: Y-axis limits for the saved quick-look results.
        debug: Whether to display debug information.
        progress: Whether to display progress bar.
        chan_flip: Whether to flip the channel order.
            Note that this is a temporary workaround for reading SAM45 logs
            with flipped channel order and will be removed in future versions.
    Returns:
        Absolute path to the saved quick-look results.
        If multiple logs are given, the first log name will be used for saving.

    """
    with set_logger(debug):
        for key, val in (params := locals().copy()).items():
            LOGGER.debug(f"{key}: {val!r}")

    if len(logs) == 0:
        raise ValueError("At least one SAM45 log must be given.")
    if len(logs) == 1:
        prefix = f"{Path(logs[0]).name}.{'+'.join(arrays)}.qlook.psw"
    else:
        prefix = f"{Path(logs[0]).name}+.{'+'.join(arrays)}.qlook.psw"

    if (results := Path(f"{prefix}.pdf")).exists() and not overwrite:
        raise FileExistsError(f"{results} already exists.")

    # Read SAM45 logs and arrays
    with tqdm(
        desc="Reading SAM45 logs/arrays",
        disable=not progress,
        total=len(logs) * len(arrays),
    ) as bar:
        analysis_ranges = Any(NDRange(*args) for args in analysis_ranges)
        Ps: list[xr.DataArray] = []

        for log, array in product(logs, arrays):
            P = read(
                log,
                array,
                time_binning=time_binning,
                chan_binning=chan_binning,
                chan_flip=chan_flip,
            )

            # swap ON/OFF for non-central beams
            is_on = (P.state == "ON") & (P.beam != 1)
            is_off = (P.state == "OFF") & (P.beam != 1)
            P.state[is_on] = "OFF"
            P.state[is_off] = "ON"

            Ps.append(P)
            bar.update(1)

        P = xr.concat(Ps, dim="time").sortby("time")
        P = P.sel(chan=P.frequency == analysis_ranges)

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
        T_sigma: xr.DataArray = T_sys * (T.exposure * T.width * 1e9) ** -0.5
        T_sigma = T_sigma.assign_attrs(long_name="Expected noise level")

        S_polyfit = fit_integration(
            T_polyfit.sel(time=T_polyfit.state == "ON"),
            T_sigma.sel(time=T_sigma.state == "ON") * np.sqrt(2),
        )
        S_demist = fit_integration(
            T_demist.sel(time=T_demist.state == "ON"),
            T_sigma.sel(time=T_sigma.state == "ON"),
        )
        bar.update(1)

    # plot and save results
    with (
        tqdm(
            desc="Plotting/saving results",
            disable=not progress,
            total=2 if detailed else 1,
        ) as bar,
        PdfPages(results) as pdf,
    ):
        keywords = [
            f"{key}={value}".replace(" ", "")
            for key, value in {
                "demist_version": demist_version,
                "demist_function": "demist.nro45m.qlook.psw",
                **params,
            }.items()
        ]

        pdfinfo = pdf.infodict()
        pdfinfo["Title"] = "DE:MIST Quick-Look Results (NRO 45m PSW)"
        pdfinfo["Keywords"] = ", ".join(keywords)

        pdf.savefig(
            plot_integrated_info(
                figsize=figsize,
                horizontal=True,
                S_demist=S_demist,
                S_polyfit=S_polyfit,
                xlim=xlim,
                ylim=ylim,
            )
        )
        pdf.savefig(
            plot_integrated_info(
                figsize=figsize,
                horizontal=False,
                S_demist=S_demist,
                S_polyfit=S_polyfit,
                xlim=xlim,
                ylim=ylim,
            )
        )
        bar.update(1)

        if not detailed:
            return results.resolve()

        pdf.savefig(
            plot_cumulative_info(
                S_demist=S_demist,
                S_polyfit=S_polyfit,
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
        return results.resolve()


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
    S_demist: xr.DataArray,
    S_polyfit: xr.DataArray,
    figsize: tuple[float, float],
) -> Figure:
    """Plot cumulative information (cumulative noise level and maximum S/N)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fit_ranges = S_polyfit.fit_ranges

    ax = axes[0]
    ax.plot(
        S_polyfit.sel(chan=fit_ranges).exposure.mean("chan"),
        S_polyfit.sel(chan=fit_ranges).std("chan"),
        label="PolyFit (achieved)",
        color=COLOR_POLYFIT,
    )
    ax.plot(
        S_polyfit.sel(chan=fit_ranges).exposure.mean("chan"),
        S_polyfit.sel(chan=fit_ranges).noise.mean("chan"),
        label="PolyFit (expected)",
        alpha=0.5,
        color=COLOR_POLYFIT,
        linestyle="dotted",
    )
    ax.plot(
        S_demist.sel(chan=fit_ranges).exposure.mean("chan"),
        S_demist.sel(chan=fit_ranges).std("chan"),
        label="DE:MIST (achieved)",
        color=COLOR_DEMIST,
    )
    ax.plot(
        S_demist.sel(chan=fit_ranges).exposure.mean("chan"),
        S_demist.sel(chan=fit_ranges).noise.mean("chan"),
        label="DE:MIST (expected)",
        alpha=0.5,
        color=COLOR_DEMIST,
        linestyle="dotted",
    )
    ax.set_title("Noise level (averaged over PolyFit ranges)")
    ax.set_xlabel(f"{S_demist.exposure.long_name} [{S_demist.exposure.units}]")
    ax.set_ylabel(f"Standard deviation [{S_demist.units}]")

    ax = axes[1]
    ax.plot(
        S_polyfit.sel(chan=fit_ranges).exposure.mean("chan"),
        S_polyfit.max("chan") / S_polyfit.sel(chan=fit_ranges).std("chan"),
        label="PolyFit",
        color=COLOR_POLYFIT,
    )
    ax.plot(
        S_demist.sel(chan=fit_ranges).exposure.mean("chan"),
        S_demist.max("chan") / S_demist.sel(chan=fit_ranges).std("chan"),
        label="DE:MIST",
        color=COLOR_DEMIST,
    )
    ax.set_title("Maximum S/N (among analysis ranges)")
    ax.set_xlabel(f"{S_demist.exposure.long_name} [{S_demist.exposure.units}]")
    ax.set_ylabel(f"Maximum signal-to-noise ratio [1]")
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
    S_demist: xr.DataArray,
    S_polyfit: xr.DataArray,
    xlim: Range = (None, None),
    ylim: Range = (None, None),
) -> Figure:
    """Plot integrated information (spectra with noise levels)."""
    if horizontal:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    alpha_polyfit = (
        # fmt: off
        S_polyfit.isel(time=-1).sel(chan=S_polyfit.fit_ranges).std()
        / S_demist.noise.isel(time=-1)
        # fmt: on
    )
    alpha_demist = (
        # fmt: off
        S_demist.isel(time=-1).sel(chan=S_polyfit.fit_ranges).std()
        / S_demist.noise.isel(time=-1)
        # fmt: on
    )
    spec_polyfit = S_polyfit.isel(time=-1).swap_dims(chan="frequency")
    spec_demist = S_demist.isel(time=-1).swap_dims(chan="frequency")

    ax = axes[0]
    ax.fill_between(
        spec_polyfit.frequency,
        spec_polyfit,
        color=COLOR_POLYFIT,
        ec="none",
        label=(
            # fmt: off
            "Achieved "
            rf"($\alpha$ = {alpha_polyfit.mean():.2f} $\pm$ {alpha_polyfit.std():.2f})"
            # fmt: on
        ),
        step="mid",
    )
    ax.fill_between(
        spec_polyfit.frequency,
        -spec_polyfit.noise,
        +spec_polyfit.noise,
        color=COLOR_POLYFIT,
        alpha=0.25,
        ec="none",
        label=r"Expected noise level ($\alpha = \sqrt{2}$)",
    )
    ax.set_title("Integrated spectrum (PolyFit)")
    ax.set_xlabel(
        f"{spec_polyfit.frequency.long_name} [{spec_polyfit.frequency.units}]"
    )
    ax.set_ylabel(f"{spec_polyfit.long_name} [{spec_polyfit.units}]")

    ax = axes[1]
    ax.fill_between(
        spec_demist.frequency,
        spec_demist,
        color=COLOR_DEMIST,
        ec="none",
        label=(
            # fmt: off
            "Achieved "
            rf"($\alpha$ = {alpha_demist.mean():.2f} $\pm$ {alpha_demist.std():.2f})"
            # fmt: on
        ),
        step="mid",
    )
    ax.fill_between(
        spec_demist.frequency,
        -spec_demist.noise,
        +spec_demist.noise,
        color=COLOR_DEMIST,
        alpha=0.25,
        ec="none",
        label=r"Expected noise level ($\alpha = 1$)",
    )
    ax.set_title("Integrated spectrum (DE:MIST)")
    ax.set_xlabel(f"{spec_demist.frequency.long_name} [{spec_demist.frequency.units}]")
    ax.set_ylabel(f"{spec_demist.long_name} [{spec_demist.units}]")

    for ax in axes:
        ax.grid()
        ax.legend()
        ax.margins(x=0.0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

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
