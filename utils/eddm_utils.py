import math


class EDDMDetector:

    def __init__(
        self,
        warning_threshold=0.95,
        drift_threshold=0.90,
        min_num_errors=50
    ):
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.min_num_errors = min_num_errors

        self.reset()

    def reset(self):

        # Total processed predictions
        self.num_instances = 0

        # Total classification errors
        self.num_errors = 0

        # Position (prediction number) where previous error occurred
        self.last_error_position = 0

        # Running statistics of distance between consecutive errors
        self.mean_distance = 0.0
        self.m2 = 0.0

        # Best value observed so far
        self.max_mean_std = 0.0

    def update(self, is_correct: bool):

        self.num_instances += 1

        # No new error
        if is_correct:

            return self._result(
                status="normal",
                warning=False,
                drift=False,
                ratio=None
            )

        # New classification error
        self.num_errors += 1

        distance = (
            self.num_instances -
            self.last_error_position
        )

        self.last_error_position = self.num_instances

        # First error
        if self.num_errors == 1:

            self.mean_distance = distance

            return self._result(
                status="normal",
                warning=False,
                drift=False,
                ratio=None
            )

        # Welford online mean / variance
        delta = distance - self.mean_distance

        self.mean_distance += delta / self.num_errors

        delta2 = distance - self.mean_distance

        self.m2 += delta * delta2

        variance = self.m2 / (self.num_errors - 1)

        std_distance = math.sqrt(variance)

        current_mean_std = (
            self.mean_distance +
            2 * std_distance
        )

        if current_mean_std > self.max_mean_std:

            self.max_mean_std = current_mean_std

        # Ignore EDDM decisions until enough errors
        if self.num_errors < self.min_num_errors:

            return self._result(
                status="normal",
                warning=False,
                drift=False,
                ratio=None
            )

        ratio = (
            current_mean_std /
            self.max_mean_std
        )

        if ratio < self.drift_threshold:

            return self._result(
                status="drift",
                warning=False,
                drift=True,
                ratio=ratio
            )

        elif ratio < self.warning_threshold:

            return self._result(
                status="warning",
                warning=True,
                drift=False,
                ratio=ratio
            )

        return self._result(
            status="normal",
            warning=False,
            drift=False,
            ratio=ratio
        )

    def _result(
        self,
        status,
        warning,
        drift,
        ratio
    ):

        return {

            "status": status,

            "warning": warning,

            "drift": drift,

            "ratio": ratio,

            "num_instances": self.num_instances,

            "num_errors": self.num_errors,

            "mean_distance": self.mean_distance,

            "max_mean_std": self.max_mean_std
        }