__all__ = [
    "fit_background",
    "fit_integration",
    "fit_lowrank",
    "fit_polynomial",
    "fit_outliers",
    "fit_sparse",
]

# standard library
from collections.abc import Sequence

# dependencies
import numpy as np
import xarray as xr
from ndtools import Any, Range as NDRange
from scipy.ndimage import median_filter
from sklearn.decomposition import TruncatedSVD
from ..stats import mad, mean

# type hints
Range = tuple[float | None, float | None]

# constants
SIGMA_OVER_MAD = 1.4826


def fit_background(
    da: xr.DataArray,
    target: str,
    /,
    *,
    fit_per_array: bool = True,
    fit_per_observation: bool = True,
    fit_per_scans: int | None = None,
) -> xr.DataArray:
    """Fit background model to a DataArray.

    Args:
        da: DataArray to fit.
        target: Target state to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_per_scans: Number of consecutive scans to fit independently.

    Returns:
        Modeled background DataArray.
    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if fit_per_scans is not None:
        if fit_per_observation:
            phase = da.scan.groupby("observation").map(phase_of)
        else:
            phase = phase_of(da.scan)

        da = da.assign_coords(scanblock=phase // fit_per_scans)
        groups.append("scanblock")

    if groups:
        return (
            da.groupby(groups)
            .apply(
                fit_background,
                args=(target,),
                fit_per_array=False,
                fit_per_observation=False,
                fit_per_scans=None,
            )
            .drop_vars("scanblock", errors="ignore")
        )

    model = (
        da.sel(time=da.state == target)
        .groupby("scan")
        .apply(mean, dim="time", keep_dims=True)
    )

    if model.sizes["time"] > 1:
        return (
            # fmt: off
            model
            .interp_like(da, kwargs={"fill_value": "extrapolate"})
            .assign_coords(da.coords)
            # fmt: on
        )
    else:
        return (
            # fmt: off
            model
            .reindex_like(da, method="nearest")
            .assign_coords(da.coords)
            # fmt: on
        )


def fit_integration(
    da: xr.DataArray,
    sigma: xr.DataArray,
    /,
    fit_seed: int = 0,
) -> xr.DataArray:
    """Fit cumulative-integration model to a DataArray.

    Args:
        da: DataArray to fit.
        sigma: DataArray of noise level.
        fit_seed: Random seed for shuffling time samples.

    Returns:
        Modeled cumulative-integration DataArray.
    """
    # shuffle time samples
    np.random.seed(fit_seed)
    index = np.random.permutation(np.arange(da.sizes["time"]))
    da = da.isel(time=index)
    sigma = sigma.isel(time=index)

    # calculate cumulative information
    weight = sigma**-2
    signal = (da * weight).cumsum("time") / weight.cumsum("time")
    noise = weight.cumsum("time") ** -0.5
    exposure = weight.cumsum("time") ** 2 / (weight**2 / da.exposure).cumsum("time")

    return signal.assign_coords(
        exposure=exposure.assign_attrs(
            long_name="Effective exposure time",
            units=da.exposure.units,
        ),
        noise=noise.assign_attrs(
            long_name="Expected noise level",
            units=sigma.units,
        ),
    ).assign_attrs(
        long_name=r"$T_{\mathrm{A}}^{\ast}$",
        units=da.units,
    )


def fit_lowrank(
    da: xr.DataArray,
    /,
    *,
    fit_components: int = 5,
    fit_off_only: bool = False,
    fit_per_array: bool = True,
    fit_per_observation: bool = True,
    fit_per_scans: int | None = None,
    fit_seed: int = 0,
) -> xr.DataArray:
    """Fit low-rank model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_components: Number of low-rank components to fit.
        fit_off_only: Whether to use only OFF samples for fitting.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_per_scans: Number of consecutive scans to fit independently.
        fit_seed: Random seed for truncated SVD.

    Returns:
        Modeled low-rank DataArray.
    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if fit_per_scans is not None:
        if fit_per_observation:
            phase = da.scan.groupby("observation").map(phase_of)
        else:
            phase = phase_of(da.scan)

        da = da.assign_coords(scanblock=phase // fit_per_scans)
        groups.append("scanblock")

    if groups:
        return (
            da.groupby(groups)
            .apply(
                fit_lowrank,
                fit_components=fit_components,
                fit_off_only=fit_off_only,
                fit_per_array=False,
                fit_per_observation=False,
                fit_per_scans=None,
            )
            .drop_vars("scanblock", errors="ignore")
        )

    model = TruncatedSVD(fit_components, random_state=fit_seed)
    model.fit(da.sel(time=da.state == "OFF") if fit_off_only else da)
    coefs = model.transform(da)

    return (
        # fmt: off
        (xr.zeros_like(da) + model.inverse_transform(coefs))
        .assign_coords(
            {
                f"lowrank_{i}": ("time", coefs[:, i])
                for i in range(fit_components)
            }
        )
        # fmt: on
    )


def fit_polynomial(
    da: xr.DataArray,
    /,
    *,
    fit_degree: int = 1,
    fit_per_array: bool = True,
    fit_per_observation: bool = True,
    fit_per_scans: int | None = None,
    fit_ranges: Sequence[Range] = ((None, None),),
) -> xr.DataArray:
    """Fit polynomial model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_degree: Degree of the polynomial to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_per_scans: Number of consecutive scans to fit independently.
        fit_ranges: Frequency ranges in GHz to use for fitting.

    Returns:
        Modeled polynomial DataArray.
    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if fit_per_scans is not None:
        if fit_per_observation:
            phase = da.scan.groupby("observation").map(phase_of)
        else:
            phase = phase_of(da.scan)

        da = da.assign_coords(scanblock=phase // fit_per_scans)
        groups.append("scanblock")

    if groups:
        return (
            da.groupby(groups)
            .apply(
                fit_polynomial,
                fit_degree=fit_degree,
                fit_per_array=False,
                fit_per_observation=False,
                fit_per_scans=None,
                fit_ranges=fit_ranges,
            )
            .drop_vars("scanblock", errors="ignore")
        )

    fit_ranges = Any(NDRange(*args) for args in fit_ranges)
    da_fit = da.sel(chan=da.frequency == fit_ranges)
    model = da_fit.polyfit("chan", fit_degree)
    coefs = model.polyfit_coefficients

    return (
        xr.polyval(da_fit.chan, coefs)
        .interp_like(da, kwargs={"fill_value": "extrapolate"})
        .assign_coords(fit_ranges=da.frequency == fit_ranges)
        .assign_coords(
            {
                # fmt: off
                f"polyfit_{i}": ("time", coefs.sel(degree=i).data)
                for i in range(fit_degree + 1)
                # fmt: on
            }
        )
    )


def fit_outliers(
    da: xr.DataArray,
    /,
    *,
    fit_per_array: bool = True,
    fit_per_observation: bool = True,
    fit_per_scans: int | None = None,
    method: str = "lowrank",
    threshold: float = 3.0,
) -> xr.DataArray:
    """Detect outlier scans based on low-rank or polynomial coefficients.

    Args:
        da: DataArray to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_per_scans: Number of consecutive scans to fit independently.
        method: Prefix of the coefficient coordinates (lowrank or polyfit).
        threshold: Z-score threshold for outlier detection.

    Returns:
        1D boolean DataArray indicating outlier scans.
    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if fit_per_scans is not None:
        if fit_per_observation:
            phase = da.scan.groupby("observation").map(phase_of)
        else:
            phase = phase_of(da.scan)

        da = da.assign_coords(scanblock=phase // fit_per_scans)
        groups.append("scanblock")

    if groups:
        return (
            da.groupby(groups)
            .apply(
                fit_outliers,
                fit_per_array=False,
                fit_per_observation=False,
                fit_per_scans=None,
                method=method,
                threshold=threshold,
            )
            .drop_vars("scanblock", errors="ignore")
        )

    coefs = xr.concat(
        [
            da.coords[key]
            for key in da.coords.keys()
            if isinstance(key, str) and key.startswith(f"{method}_")
        ],
        "component",
    )
    score = (coefs - coefs.mean("time")) / coefs.std("time")

    return (
        (abs(score) > threshold)
        .any("component")
        .groupby("scan")
        .map(lambda da: da.any() | xr.zeros_like(da, dtype=bool))
    )


def fit_sparse(
    da: xr.DataArray,
    /,
    *,
    fit_per_array: bool = False,
    fit_per_observation: bool = False,
    fit_per_scans: int | None = None,
    fit_prefilter: int = 3,
    fit_threshold: float = 3.0,
) -> xr.DataArray:
    """Fit sparse model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_per_scans: Number of consecutive scans to fit independently.
        fit_prefilter: Size of the median filter before signal detection.
        fit_threshold: Absolute S/N threshold for signal detection.

    Returns:
        Modeled sparse DataArray.
    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if fit_per_scans is not None:
        if fit_per_observation:
            phase = da.scan.groupby("observation").map(phase_of)
        else:
            phase = phase_of(da.scan)

        da = da.assign_coords(scanblock=phase // fit_per_scans)
        groups.append("scanblock")

    if groups:
        return (
            da.groupby(groups)
            .apply(
                fit_sparse,
                fit_per_array=False,
                fit_per_observation=False,
                fit_per_scans=None,
                fit_prefilter=fit_prefilter,
                fit_threshold=fit_threshold,
            )
            .drop_vars("scanblock", errors="ignore")
        )

    signal = da.sel(time=da.state == "ON").mean("time")
    signal.data = median_filter(signal.data, fit_prefilter)
    noise = SIGMA_OVER_MAD * mad(signal)
    signal[abs(signal / noise) < fit_threshold] = 0
    signal = signal.assign_coords(signal_ranges=signal != 0)
    return (da.state == "ON") * signal


def phase_of(da: xr.DataArray, /) -> xr.DataArray:
    """Assign a phase to each value in a 1D DataArray.

    The function assigns a unique phase (int64) to consecutive
    identical values in the DataArray. The phase increases
    sequentially whenever the value in the DataArray changes.

    Args:
        da: Input 1D DataArray.

    Returns:
        1D int64 DataArray of phases.
    """
    if da.ndim != 1:
        raise ValueError("Input DataArray must be 1D.")

    is_transision = xr.zeros_like(da, bool)
    is_transision.data[1:] = da.data[1:] != da.data[:-1]
    return is_transision.cumsum().rename(None).reset_coords(drop=True)
