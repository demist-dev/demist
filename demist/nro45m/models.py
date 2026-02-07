__all__ = [
    "fit_background",
    "fit_integration",
    "fit_lowrank",
    "fit_polynomial",
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
) -> xr.DataArray:
    """Fit background model to a DataArray.

    Args:
        da: DataArray to fit.
        target: Target state to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.

    Returns:
        Modeled background DataArray.

    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if groups:
        return da.groupby(groups).apply(
            fit_background,
            args=(target,),
            fit_per_array=False,
            fit_per_observation=False,
        )

    model = (
        da.sel(time=da.state == target)
        .groupby("scan")
        .apply(mean, dim="time", keep_dims=True)
    )

    if model.sizes["time"] > 1:
        return model.interp_like(da, kwargs={"fill_value": "extrapolate"})
    else:
        return model.reindex_like(da, method="nearest")


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
        exposure=exposure.assign_attrs(long_name="Effective exposure time"),
        noise=noise.assign_attrs(long_name="Expected noise level"),
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
) -> xr.DataArray:
    """Fit low-rank model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_components: Number of low-rank components to fit.
        fit_off_only: Whether to use only OFF samples for fitting.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.

    Returns:
        Modeled low-rank DataArray.

    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if groups:
        return da.groupby(groups).apply(
            fit_lowrank,
            fit_components=fit_components,
            fit_off_only=fit_off_only,
            fit_per_array=False,
            fit_per_observation=False,
        )

    model = TruncatedSVD(fit_components)
    model.fit(da.sel(time=da.state == "OFF") if fit_off_only else da)
    return xr.zeros_like(da) + model.inverse_transform(model.transform(da))


def fit_polynomial(
    da: xr.DataArray,
    /,
    *,
    fit_degree: int = 1,
    fit_per_array: bool = True,
    fit_per_observation: bool = True,
    fit_ranges: Sequence[Range] = ((None, None),),
) -> xr.DataArray:
    """Fit polynomial model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_degree: Degree of the polynomial to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
        fit_ranges: Frequency ranges in GHz to use for fitting.

    Returns:
        Modeled polynomial DataArray.

    """
    groups: list[str] = []

    if fit_per_array:
        groups.append("array")

    if fit_per_observation:
        groups.append("observation")

    if groups:
        return da.groupby(groups).apply(
            fit_polynomial,
            fit_degree=fit_degree,
            fit_per_array=False,
            fit_per_observation=False,
            fit_ranges=fit_ranges,
        )

    fit_ranges = Any(NDRange(*args) for args in fit_ranges)
    da_fit = da.sel(chan=da.frequency == fit_ranges)
    model = da_fit.polyfit("chan", fit_degree)

    return (
        xr.polyval(da_fit.chan, model.polyfit_coefficients)
        .interp_like(da, kwargs={"fill_value": "extrapolate"})
        .assign_coords(fit_ranges=da.frequency == fit_ranges)
    )


def fit_sparse(
    da: xr.DataArray,
    /,
    *,
    fit_per_array: bool = False,
    fit_per_observation: bool = False,
    fit_prefilter: int = 3,
    fit_threshold: float = 3.0,
) -> xr.DataArray:
    """Fit sparse model to a DataArray.

    Args:
        da: DataArray to fit.
        fit_per_array: Whether to fit per array.
        fit_per_observation: Whether to fit per observation.
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

    if groups:
        return da.groupby(groups).apply(
            fit_sparse,
            fit_per_array=False,
            fit_per_observation=False,
            fit_prefilter=fit_prefilter,
            fit_threshold=fit_threshold,
        )

    signal = da.sel(time=da.state == "ON").mean("time")
    signal.data = median_filter(signal.data, fit_prefilter)
    noise = SIGMA_OVER_MAD * mad(signal)
    signal[abs(signal / noise) < fit_threshold] = 0
    return (da.state == "ON") * signal
