from datetime import datetime
from typing import Optional, Union
import pandas as pd


class TimeSeriesWrapper:
    """
    A wrapper class for time series data that provides preprocessing capabilities.

    This class handles various transformations like resampling, normalization, and moving average calculations.
    """

    def __init__(
        self,
        time_series: Union[
            pd.DataFrame,
            tuple[list[datetime], list[float]],
            tuple[list[datetime], list[list[float]]],
            list[tuple[list[datetime], list[float]]]
        ],
    ):
        """
        Args:
            time_series: DataFrame with index as timestamps and one or more value columns,
                        or tuple containing timestamps and values in various formats:

                        - Univariate: tuple[timestamps, values_list]
                        - Multivariate with shared timestamps: tuple[timestamps, [[series1_vals], [series2_vals], ...]]
                        - Multivariate with individual timestamps: [tuple[timestamps1, values1], tuple[timestamps2, values2], ...]
        """
        processed_time_series = self._build_time_series(time_series)

        self._original_time_series = processed_time_series
        self._dim = len(processed_time_series.columns)
        self._is_multivariate = self._dim > 1

        processed_time_series = processed_time_series.dropna(how='all')
        self._time_series, self.granularity = self.temporal_resample(processed_time_series)

    @staticmethod
    def _build_time_series(
        time_series: Union[
            pd.DataFrame,
            tuple[list[datetime], list[float]],
            tuple[list[datetime], list[list[float]]],
            list[tuple[list[datetime], list[float]]]
        ],
    ) -> pd.DataFrame:
        """
        Build a standardized DataFrame from various time series input formats.

        Args:
            time_series: Input time series in various formats

        Returns:
            Standardized pandas DataFrame with datetime index and value columns
        """
        if isinstance(time_series, pd.DataFrame):
            df = TimeSeriesWrapper._build_from_dataframe(time_series)
        elif isinstance(time_series, tuple):
            df = TimeSeriesWrapper._build_from_tuple(time_series)
        elif isinstance(time_series, list) and len(time_series) > 0 and isinstance(time_series[0], tuple):
            df = TimeSeriesWrapper._build_from_list_of_tuples(time_series)
        else:
            raise ValueError(f"Unsupported time series format: {type(time_series)}")

        if not df.index.is_monotonic_increasing:
            raise ValueError("Time series index must be sorted in ascending order without NaN values.")

        return df

    @staticmethod
    def _build_from_dataframe(time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Builder for DataFrame input format.

        Args:
            time_series: Input DataFrame

        Returns:
            Processed DataFrame
        """
        if time_series.shape[1] == 0:
            raise ValueError(f'Time series DataFrame must contain values, got Dataframe of shape {time_series.shape}')
        return time_series.copy()

    @staticmethod
    def _build_from_tuple(time_series: tuple) -> pd.DataFrame:
        """
        Builder for tuple input format (univariate or multivariate with shared timestamps).

        Args:
            time_series: Tuple of (timestamps, values)

        Returns:
            Processed DataFrame
        """
        timestamps, values = time_series

        def is_iterable_collection(obj):
            return hasattr(obj, '__getitem__') and hasattr(obj, '__iter__') and hasattr(obj, '__len__')

        # Check if timestamps and values are iterable
        if not is_iterable_collection(timestamps):
            raise ValueError(f"Timestamps must have __iter__ and __getitem__ methods, got {type(timestamps)} object")
        if not is_iterable_collection(values):
            raise ValueError(f"Values must have __iter__ and __getitem__ methods, got {type(values)} object")

        # Validate that values is not empty
        if len(values) == 0:
            raise ValueError("Values must not be empty")

        # Validate that timestamps and values have matching lengths
        if not is_iterable_collection(values[0]):
            # Univariate case - check length match
            if len(timestamps) != len(values):
                raise ValueError(
                    f"Length mismatch: timestamps ({len(timestamps)}) and values ({len(values)}) must have the same length"
                )
        else:
            # Multivariate case - check length match for each series
            for i, series_values in enumerate(values):
                if len(timestamps) != len(series_values):
                    raise ValueError(
                        f"Length mismatch: timestamps ({len(timestamps)}) and values[{i}] ({len(series_values)}) must have the same length"
                    )

        # Normalize to list of lists format for unified processing
        if is_iterable_collection(values[0]):
            # Multivariate with shared timestamps
            values_list = values
        else:
            # Univariate - wrap in list to use same format
            values_list = [values]

        # Create DataFrame with unified value_0, value_1, ... format
        df = pd.DataFrame({
            f"value_{i}": values_series
            for i, values_series in enumerate(values_list)
        }, index=timestamps)
        return df

    @staticmethod
    def _build_from_list_of_tuples(time_series: list[tuple]) -> pd.DataFrame:
        """
        Builder for list of tuples input format (multivariate with individual timestamps).

        Args:
            time_series: List of (timestamps, values) tuples

        Returns:
            Processed DataFrame
        """
        all_data = []
        for i, tuple_data in enumerate(time_series):
            # Use _build_from_tuple for each individual tuple
            series_df = TimeSeriesWrapper._build_from_tuple(tuple_data)
            series_name = f"value_{i}"
            # Rename the column to maintain consistent naming
            series_df.columns = [series_name]
            all_data.append(series_df)

        df = pd.concat(all_data, axis=1)
        df = df.sort_index()
        return df

    @staticmethod
    def temporal_resample(time_series: pd.DataFrame, granularity: Optional[str] = None) -> tuple[pd.DataFrame, str]:
        """
        Resample time series to a regular frequency.

        Args:
            time_series: Input DataFrame
            granularity: Resampling frequency (e.g. '1h', '30min', '1D')

        Returns:
            Tuple of (resampled DataFrame, granularity)
        """
        if granularity is None:
            granularity = pd.infer_freq(time_series.index)
        if granularity is None:
            if len(time_series.index) < 2:
                granularity = "D"
            else:
                diffs = time_series.index.to_series().diff().dropna()
                non_zero_diffs = diffs[diffs > pd.Timedelta(0)]

                if non_zero_diffs.empty:
                    granularity = "D"
                else:
                    inference_diff = non_zero_diffs.value_counts().index[0]
                    total_seconds = inference_diff.total_seconds()

                    if total_seconds % 86400 == 0:  # days
                        days = total_seconds // 86400
                        granularity = f"{int(days)}D"
                    elif total_seconds % 3600 == 0:  # hours
                        hours = total_seconds // 3600
                        granularity = f"{int(hours)}h"
                    elif total_seconds % 60 == 0:  # minutes
                        minutes = total_seconds // 60
                        granularity = f"{int(minutes)}min"
                    else:  # seconds
                        granularity = f"{int(total_seconds)}s"

        resampled = time_series.resample(granularity).mean().interpolate()
        if resampled.shape[1] > 1:
            resampled = resampled.ffill().bfill().fillna(0)
        return resampled, granularity

    @staticmethod
    def mean_var_normalize(time_series: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mean-variance normalization.

        Args:
            time_series: Input time series
            **kwargs: Parameters for MeanVarNormalize

        Returns:
            Normalized TimeSeries
        """
        return (time_series - time_series.mean()) / (time_series.std() + 1e-8)

    @staticmethod
    def moving_average(time_series: pd.DataFrame, n_steps: int = 1):
        """
        Apply moving average smoothing.

        Args:
            time_series: Input time series
            n_steps: Window size for moving average

        Returns:
            Smoothed TimeSeries
        """
        return time_series.rolling(window=n_steps, min_periods=1).mean()

    def apply_transforms(
        self,
        apply_normalization: bool = False,
        apply_moving_average: bool = False,
        apply_spectral_residual: bool = False,
        apply_stl_decomposition: bool = False,
        spectral_residual_window: int = 3,
        spectral_residual_padding: int = 10,
        spectral_residual_padding_mode: str = "reflect",
        moving_average_n_steps: int = 1,
        stl_decomposition_n_steps: int = 1,
        granularity: Optional[str] = None,
    ) -> "TimeSeriesWrapper":
        """
        Apply a sequence of transformations to the time series.

        Args:
            resample_params: Parameters for temporal resampling
            normalize_params: Parameters for normalization
            moving_average_params: Parameters for moving average

        Returns:
            New TimeSeriesWrapper instance with transformed data
        """
        ts = self._original_time_series.copy()

        self._time_series, self.granularity = TimeSeriesWrapper.temporal_resample(ts, granularity=granularity)

        if apply_normalization:
            self._time_series = self.mean_var_normalize(self._time_series)

        if apply_moving_average:
            self._time_series = self.moving_average(self._time_series, n_steps=moving_average_n_steps)

        return self

    def copy(self) -> "TimeSeriesWrapper":
        return TimeSeriesWrapper(self._original_time_series.copy())

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.time_series_pd.values.flatten()),
                tuple(self.time_series_pd.index),
                tuple(self.time_series_pd.columns),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._original_time_series.equals(other._original_time_series)

    @property
    def original_time_series(self) -> pd.DataFrame:
        return self._original_time_series.copy()

    @property
    def time_series_pd(self) -> pd.DataFrame:
        return self._time_series.copy()

    @property
    def dates(self) -> list:
        return self._time_series.index.tolist()

    @property
    def values(self) -> list:
        if self._is_multivariate:
            # Return 2D array for multivariate data
            return self._time_series.values.tolist()
        else:
            # Return list for univariate data (backward compatible)
            return self._time_series.iloc[:, 0].tolist()

    @property
    def is_multivariate(self) -> bool:
        """Returns True if more than one column."""
        return self._is_multivariate

    @property
    def n_series(self) -> int:
        """Returns number of series (columns)."""
        return self._dim

    @property
    def duration(self) -> pd.Timedelta:
        """Returns duration of series."""
        return self._time_series.index[-1] - self._time_series.index[0]
