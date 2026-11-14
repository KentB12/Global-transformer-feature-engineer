import numpy as np


def mae(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return np.mean(np.abs(y - yh))


def rmse(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return np.sqrt(np.mean((y - yh) ** 2))


def wmape(y, yh):
    """sum|y - yh| / sum|y| * 100. Undefined (nan) if all actuals are zero."""
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    den = np.sum(np.abs(y))
    return np.sum(np.abs(y - yh)) / den * 100 if den > 0 else np.nan