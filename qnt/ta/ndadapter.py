import numpy as np
import pandas as pd
import xarray as xr
import typing as tp

NdType = tp.Union[np.ndarray, pd.DataFrame, xr.DataArray, pd.Series]
NdTupleType = tp.Union[
    tp.Tuple[NdType],
    tp.Tuple[NdType, NdType],
    tp.Tuple[NdType, NdType, NdType],
    tp.Tuple[NdType, NdType, NdType, NdType],
]

XR_TIME_DIMENSION = "time"


def nd_universal_adapter(d1_function, nd_args: NdTupleType, plain_args: tuple) -> NdType:
    if isinstance(nd_args[0], np.ndarray):
        return nd_np_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], pd.DataFrame):
        return nd_pd_df_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], pd.Series):
        return nd_pd_s_adapter(d1_function, nd_args, plain_args)
    if isinstance(nd_args[0], xr.DataArray):
        return nd_xr_da_adapter(d1_function, nd_args, plain_args)
    raise Exception("unsupported")


def nd_np_adapter(d1_function, nd_args: tp.Tuple[np.ndarray], plain_args: tuple) -> np.ndarray:
    shape = nd_args[0].shape
    if len(shape) == 1:
        args = nd_args + plain_args
        return d1_function(*args)
    nd_args_2d = tuple(a.reshape(-1, shape[-1]) for a in nd_args)
    result2d = np.array([d1_function(*(a[i] for a in nd_args_2d), *plain_args) for i in range(nd_args_2d[0].shape[0])])
    return result2d.reshape(shape)


def nd_pd_df_adapter(d1_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple) -> pd.DataFrame:
    np_nd_args = tuple(a.to_numpy().transpose() for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.DataFrame(np_result, columns=nd_args[0].columns, index=nd_args[0].index)


def nd_pd_s_adapter(d1_function, nd_args: tp.Tuple[pd.Series], plain_args: tuple) -> pd.Series:
    np_nd_args = tuple(a.to_numpy() for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.Series(np_result, nd_args[0].index)


def nd_xr_da_adapter(d1_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple) -> xr.DataArray:
    origin_dims = nd_args[0].dims
    transpose_dims = tuple(i for i in origin_dims if i != XR_TIME_DIMENSION) + (XR_TIME_DIMENSION,)
    np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    return xr.DataArray(np_result, dims=transpose_dims, coords=nd_args[0].coords).transpose(*origin_dims)


def nd_to_1d_universal_adapter(np_function, nd_args: NdTupleType, plain_args: tuple) -> NdType:
    if isinstance(nd_args[0], np.ndarray):
        return nd_to_1d_np_adapter(nd_args, plain_args)
    if isinstance(nd_args[0], pd.DataFrame):
        return nd_to_1d_pd_df_adapter(np_function, nd_args, plain_args)
    if isinstance(nd_args[0], xr.DataArray):
        return nd_to_1d_xr_da_adapter(np_function, nd_args, plain_args)
    raise Exception("unsupported")


def nd_to_1d_np_adapter(np_function, nd_args: tp.Tuple[np.ndarray], plain_args: tuple) -> np.ndarray:
    args = nd_args + plain_args
    return np_function(*args)


def nd_to_1d_pd_df_adapter(np_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple) -> pd.Series:
    # This operation is memory intensive, but can be made more efficient by operating on the minimal ndarray
    # conversion only for required columns (if relevant); however, signature and generality must be preserved.
    # Flatten all nd_args at once to a single ndarray for vectorization if >1 argument
    if len(nd_args) == 1:
        arr = nd_args[0].to_numpy().T
        np_result = np_function(arr, *plain_args)
        np_result = np_result.T
    else:
        # Use generator expression for potentially reduced memory pressure
        np_nd_args = tuple(a.to_numpy().T for a in nd_args)
        np_result = np_function(*np_nd_args, *plain_args)
        np_result = np_result.T
    # Use pd.Series constructor efficiently
    return pd.Series(np_result, index=nd_args[0].index)


def nd_to_1d_xr_da_adapter(np_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple) -> xr.DataArray:
    first_da = nd_args[0]
    origin_dims = first_da.dims
    # Avoid repeated attribute lookup in tight loop
    time_dim = XR_TIME_DIMENSION

    # Check for common case of ('asset', 'time') or ('instrument', 'time'), avoid unnecessary tuple constructs
    if origin_dims[-1] == time_dim and all(a.dims == origin_dims for a in nd_args):
        # Fast-path: already last dim is time and all have same order, just .values
        np_nd_args = tuple(a.values for a in nd_args)
    else:
        # General path: must transpose
        transpose_dims = tuple(i for i in origin_dims if i != time_dim) + (time_dim,)
        np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)

    np_result = np_function(*np_nd_args, *plain_args)

    # Place time as the only dimension, with appropriate coordinates
    # Reuse the coord object directly for no-copy
    time_coord = first_da.coords[time_dim]

    return xr.DataArray(
        np_result,
        dims=[time_dim],
        coords={time_dim: time_coord}
    )
