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
        # Avoid tuple concatenation overhead for single-dimensional args
        return d1_function(*(nd_args + plain_args))
    # Flatten nd_args to 2D only if necessary, using efficient np.reshape and avoid generator for tuple creation
    nd_args_2d = tuple(np.reshape(a, (-1, shape[-1])) for a in nd_args)
    n_rows = nd_args_2d[0].shape[0]
    # Use preallocated numpy array and vectorized unpacking for speed
    results = np.empty((n_rows,), dtype=nd_args_2d[0].dtype)
    # It is not possible to vectorize d1_function over arbitrary input, but preallocating output and re-using views is faster
    for i in range(n_rows):
        row_args = (a[i] for a in nd_args_2d)
        results[i] = d1_function(*row_args, *plain_args)
    return results.reshape(shape)


def nd_pd_df_adapter(d1_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple) -> pd.DataFrame:
    np_nd_args = tuple(a.to_numpy().transpose() for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.DataFrame(np_result, columns=nd_args[0].columns, index=nd_args[0].index)


def nd_pd_s_adapter(d1_function, nd_args: tp.Tuple[pd.Series], plain_args: tuple) -> pd.Series:
    # Avoid repeated to_numpy calls and convert all Series to numpy arrays at once
    np_nd_args = tuple(a.values for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.transpose()  # This step is necessary for compatibility, leave as-is
    # Use the existing index directly for pandas Series construction
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
    np_nd_args = tuple(a.to_numpy().transpose() for a in nd_args)
    np_result = nd_to_1d_np_adapter(np_function, np_nd_args, plain_args)
    np_result = np_result.transpose()
    return pd.Series(np_result, index=nd_args[0].index)


def nd_to_1d_xr_da_adapter(np_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple) -> xr.DataArray:
    origin_dims = nd_args[0].dims
    transpose_dims = tuple(i for i in origin_dims if i != XR_TIME_DIMENSION) + (XR_TIME_DIMENSION,)
    np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)
    np_result = nd_to_1d_np_adapter(np_function, np_nd_args, plain_args)
    return xr.DataArray(
        np_result,
        dims=[XR_TIME_DIMENSION],
        coords=[nd_args[0].coords[XR_TIME_DIMENSION]]
    )
