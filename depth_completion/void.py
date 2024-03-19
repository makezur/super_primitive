# mixture from https://github.com/isl-org/VI-Depth
# and https://github.com/nianticlabs/simplerecon/blob/main/utils/metrics_utils.py

import numpy as np
import torch

def rmse(estimate, target):
    return np.sqrt(np.mean((estimate - target) ** 2))

def mae(estimate, target):
    return np.mean(np.abs(estimate - target))

def absrel(estimate, target):
    return np.mean(np.abs(estimate - target) / target)

def inv_rmse(estimate, target):
    return np.sqrt(np.mean((1.0/estimate - 1.0/target) ** 2))

def inv_mae(estimate, target):
    return np.mean(np.abs(1.0/estimate - 1.0/target))

def inv_absrel(estimate, target):
    return np.mean((np.abs(1.0/estimate - 1.0/target)) / (1.0/target))


def delta_metric(estimate, target, mult_a=False):
    thresh = np.maximum((target / estimate), 
                       (estimate / target))
    a_dict = {}
    a_dict["a5"] = (thresh < 1.05     ).astype(np.float32).mean()
    a_dict["a10"] = (thresh < 1.10     ).astype(np.float32).mean()
    a_dict["a25"] = (thresh < 1.25     ).astype(np.float32).mean()

    a_dict["a0"] = (thresh < 1.10     ).astype(np.float32).mean()
    a_dict["a1"] = (thresh < 1.25     ).astype(np.float32).mean()
    a_dict["a2"] = (thresh < 1.25 ** 2).astype(np.float32).mean()
    a_dict["a3"] = (thresh < 1.25 ** 3).astype(np.float32).mean()


    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100
    return a_dict


class ErrorMetrics(object):
    def __init__(self):
        # initialize by setting to worst values
        self.rmse, self.mae, self.absrel = np.inf, np.inf, np.inf
        self.inv_rmse, self.inv_mae, self.inv_absrel = np.inf, np.inf, np.inf

    def compute(self, estimate, target, valid):
        # apply valid masks
        estimate = estimate[valid]
        target = target[valid]

        # depth error, estimate in meters, convert units to mm
        self.rmse = rmse(1000.0*estimate, 1000.0*target)
        self.mae = mae(1000.0*estimate, 1000.0*target)
        self.absrel = absrel(1000.0*estimate, 1000.0*target)

        # inverse depth error, estimate in meters, convert units to 1/km
        self.inv_rmse = inv_rmse(0.001*estimate, 0.001*target)
        self.inv_mae = inv_mae(0.001*estimate, 0.001*target)
        self.inv_absrel = inv_absrel(0.001*estimate, 0.001*target)

class ErrorMetricsDeltas(object):
    def __init__(self):
        # initialize by setting to worst values
        self.rmse, self.mae, self.absrel = np.inf, np.inf, np.inf
        self.inv_rmse, self.inv_mae, self.inv_absrel = np.inf, np.inf, np.inf

        self.delta0, self.delta1, self.delta2, self.delta3 = 0, 0, 0, 0

    def compute(self, estimate, target, valid):
        # apply valid masks
        estimate = estimate[valid]
        target = target[valid]

        # depth error, estimate in meters, convert units to mm
        self.rmse = rmse(1000.0*estimate, 1000.0*target)
        self.mae = mae(1000.0*estimate, 1000.0*target)
        self.absrel = absrel(1000.0*estimate, 1000.0*target)

        # inverse depth error, estimate in meters, convert units to 1/km
        self.inv_rmse = inv_rmse(0.001*estimate, 0.001*target)
        self.inv_mae = inv_mae(0.001*estimate, 0.001*target)
        self.inv_absrel = inv_absrel(0.001*estimate, 0.001*target)

        deltas = delta_metric(estimate, target)
        self.delta0 = deltas['a0']
        self.delta1 = deltas['a1']
        self.delta2 = deltas['a2']
        self.delta3 = deltas['a3']

        self.delta105 = deltas['a5']
        self.delta110 = deltas['a10']


class ErrorMetricsAverager(object):
    def __init__(self):
        # initialize avg accumulators to zero
        self.rmse_avg, self.mae_avg, self.absrel_avg = 0, 0, 0
        self.inv_rmse_avg, self.inv_mae_avg, self.inv_absrel_avg = 0, 0, 0
        self.total_count = 0

    def accumulate(self, error_metrics):
        # adds to accumulators from ErrorMetrics object
        assert isinstance(error_metrics, ErrorMetrics)

        self.rmse_avg += error_metrics.rmse
        self.mae_avg += error_metrics.mae
        self.absrel_avg += error_metrics.absrel

        self.inv_rmse_avg += error_metrics.inv_rmse
        self.inv_mae_avg += error_metrics.inv_mae
        self.inv_absrel_avg += error_metrics.inv_absrel

        self.total_count += 1

    def average(self):
        # print(f"Averaging depth metrics over {self.total_count} samples")
        self.rmse_avg = self.rmse_avg / self.total_count
        self.mae_avg = self.mae_avg / self.total_count
        self.absrel_avg = self.absrel_avg / self.total_count
        # print(f"Averaging inv depth metrics over {self.total_count} samples")
        self.inv_rmse_avg = self.inv_rmse_avg / self.total_count
        self.inv_mae_avg = self.inv_mae_avg / self.total_count
        self.inv_absrel_avg = self.inv_absrel_avg / self.total_count




class ErrorMetricsDeltasAverager(object):
    def __init__(self):
        # initialize avg accumulators to zero
        self.rmse_avg, self.mae_avg, self.absrel_avg = 0, 0, 0
        self.inv_rmse_avg, self.inv_mae_avg, self.inv_absrel_avg = 0, 0, 0
        self.delta0_avg, self.delta1_avg, self.delta2_avg, self.delta3_avg = 0, 0, 0, 0
        self.delta105_avg, self.delta110_avg = 0, 0

        self.total_count = 0

    def accumulate(self, error_metrics):
        # adds to accumulators from ErrorMetrics object
        assert isinstance(error_metrics, ErrorMetricsDeltas)

        self.rmse_avg += error_metrics.rmse
        self.mae_avg += error_metrics.mae
        self.absrel_avg += error_metrics.absrel

        self.inv_rmse_avg += error_metrics.inv_rmse
        self.inv_mae_avg += error_metrics.inv_mae
        self.inv_absrel_avg += error_metrics.inv_absrel

        self.delta0_avg += error_metrics.delta0
        self.delta1_avg += error_metrics.delta1
        self.delta2_avg += error_metrics.delta2
        self.delta3_avg += error_metrics.delta3

        self.delta105_avg += error_metrics.delta105
        self.delta110_avg += error_metrics.delta110

        self.total_count += 1

    def average(self):
        # print(f"Averaging depth metrics over {self.total_count} samples")
        self.rmse_avg = self.rmse_avg / self.total_count
        self.mae_avg = self.mae_avg / self.total_count
        self.absrel_avg = self.absrel_avg / self.total_count
        # print(f"Averaging inv depth metrics over {self.total_count} samples")
        self.inv_rmse_avg = self.inv_rmse_avg / self.total_count
        self.inv_mae_avg = self.inv_mae_avg / self.total_count
        self.inv_absrel_avg = self.inv_absrel_avg / self.total_count

        self.delta0_avg = self.delta0_avg / self.total_count
        self.delta1_avg = self.delta1_avg / self.total_count
        self.delta2_avg = self.delta2_avg / self.total_count
        self.delta3_avg = self.delta3_avg / self.total_count

        self.delta105_avg = self.delta105_avg / self.total_count
        self.delta110_avg = self.delta110_avg / self.total_count


    