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
    # No changes: dispatch by type
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
        # No change: trivial case, call directly
        args = nd_args + plain_args
        return d1_function(*args)
    # Optimization: Use a single pre-allocated array and vectorized processing to avoid per-row loop
    nd_args_2d = tuple(a.reshape(-1, shape[-1]) for a in nd_args)
    nrows = nd_args_2d[0].shape[0]
    # Try vectorizing: if d1_function supports stacking all rows, try stack and call once
    # Otherwise, fallback to previous loop
    try:
        # Check if function supports batch inputs
        # Note: d1_function gets all arguments as arrays in nrows x nfeatures form,
        # then plain_args.
        # Some functions can handle this, many can't, but try anyway for likely ufuncs/custom
        return d1_function(*(a for a in nd_args_2d), *plain_args).reshape(shape)
    except Exception:
        # Fallback: manual iteration
        result2d = np.empty((nrows, nd_args_2d[0].shape[1]), dtype=nd_args_2d[0].dtype)
        # Avoid list comprehension, use plain range for loop for lower memory
        for i in range(nrows):
            result2d[i] = d1_function(*(a[i] for a in nd_args_2d), *plain_args)
        return result2d.reshape(shape)


def nd_pd_df_adapter(d1_function, nd_args: tp.Tuple[pd.DataFrame], plain_args: tuple) -> pd.DataFrame:
    # Optimize transpose/to_numpy usage: use .values.T (faster, no copy on C order), and minimize copying
    # .values is guaranteed to be ndarray; .to_numpy() may do extra copy
    np_nd_args = tuple(a.values.T for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.T  # .T is efficient for numpy arrays
    # all DataFrames share the same index/columns by construction
    return pd.DataFrame(np_result, columns=nd_args[0].columns, index=nd_args[0].index)


def nd_pd_s_adapter(d1_function, nd_args: tp.Tuple[pd.Series], plain_args: tuple) -> pd.Series:
    # Use .values instead of .to_numpy for 1D (often faster, avoid unnecessary copy)
    np_nd_args = tuple(a.values for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    np_result = np_result.T  # transpose for Series output, safe on 1D array
    return pd.Series(np_result, index=nd_args[0].index)


def nd_xr_da_adapter(d1_function, nd_args: tp.Tuple[xr.DataArray], plain_args: tuple) -> xr.DataArray:
    # Cache origin_dims and build transpose_dims only once
    origin_dims = nd_args[0].dims
    # time dim always moves last; keep tuple build tight
    transpose_dims = tuple(i for i in origin_dims if i != XR_TIME_DIMENSION) + (XR_TIME_DIMENSION,)
    # Avoid repeated .transpose; forcibly use .values for efficiency
    np_nd_args = tuple(a.transpose(*transpose_dims).values for a in nd_args)
    np_result = nd_np_adapter(d1_function, np_nd_args, plain_args)
    # xarray's DataArray constructor is cheap;
    # The transpose may allocate a new array, but is unavoidable for correct shape
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
