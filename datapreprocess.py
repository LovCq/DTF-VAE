import numpy as np

__all__ = ["complete_timestamp", "standardize_kpi"]


def complete_timestamp(timestamp, arrays=None):
    """
    Complete `timestamp` such that the time interval is homogeneous.
    Zeros will be inserted into each array in `arrays`, at missing points.
    Also, an indicator array will be returned to indicate whether each
    point is missing or not.

    Args:
        timestamp (np.ndarray): 1-D int64 array, the timestamp values.
            It can be unsorted.
        arrays (Iterable[np.ndarray]): The arrays to be filled with zeros
            according to `timestamp`. Can be 1-D or multi-dimensional.

    Returns:
        np.ndarray: A 1-D int64 array, the completed timestamp.
        np.ndarray: A 1-D int32 array, indicating whether each point is missing.
        list[np.ndarray]: The arrays, missing points filled with zeros.
            (optional, return only if `arrays` is specified)
    """
    # 将输入转为 int64 数组并检查维度
    timestamp = np.asarray(timestamp, np.int64)
    if len(timestamp.shape) != 1:
        raise ValueError("`timestamp` must be a 1-D array")

    has_arrays = arrays is not None
    # 处理输入的 arrays，转为数组集合
    arrays = [np.asarray(array) for array in (arrays or ())]
    # 检查每个数组的第一维是否与 timestamp 的长度一致
    for i, array in enumerate(arrays):
        if array.shape[0] != timestamp.shape[0]:
            raise ValueError(
                "The first dimension of ``arrays[{}]`` does not agree with "
                "the shape of `timestamp` ({} vs {})".format(
                    i, array.shape[0], timestamp.shape[0]
                )
            )

    # 对时间戳排序
    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    # 计算时间间隔，确保间隔均匀
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError("Duplicated values in `timestamp`")
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError(
                "Not all intervals in `timestamp` are multiples "
                "of the minimum interval"
            )

    # 生成补全的时间戳
    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(
        timestamp_sorted[0], timestamp_sorted[-1] + interval, interval, dtype=np.int64
    )
    ret_missing = np.ones([length], dtype=np.int32)

    # 为每个数组创建补全后的结果数组，支持多维
    ret_arrays = []
    for array in arrays:
        if array.ndim == 1:
            ret_array = np.zeros([length], dtype=array.dtype)
        else:
            ret_array = np.zeros([length] + list(array.shape[1:]), dtype=array.dtype)
        ret_arrays.append(ret_array)

    # 计算目标索引并填充数据
    dst_index = np.asarray(
        (timestamp_sorted - timestamp_sorted[0]) // interval, dtype=np.int32
    )
    ret_missing[dst_index] = 0
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if has_arrays:
        return ret_timestamp, ret_missing, ret_arrays
    else:
        return ret_timestamp, ret_missing


# standardize_kpi 函数保持不变
def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardize a KPI observation array to have mean 0 and standard deviation 1.

    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not None, use this mean to standardize values.
        std (float): If not None, use this std to standardize values.
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array indicating exclusions.

    Returns:
        np.ndarray: The standardized values.
        float: The computed or given mean.
        float: The computed or given std.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError("`values` must be a 1-D array")
    if (mean is None) != (std is None):
        raise ValueError("`mean` and `std` must be both None or not None")
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool_)
        if excludes.shape != values.shape:
            raise ValueError(
                "The shape of `excludes` does not agree with "
                "the shape of `values` ({} vs {})".format(excludes.shape, values.shape)
            )

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()

    return (values - mean) / std, mean, std