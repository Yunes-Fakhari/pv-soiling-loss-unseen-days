import numpy as np


def regression_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, float), np.asarray(predictions, float)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError('Expected equally sized, nonempty one-dimensional arrays')
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError('Metrics require finite labels and predictions')
    errors = predictions-labels
    mse = float(np.mean(errors**2))
    variance = float(np.sum((labels-labels.mean())**2))
    return {'n': len(labels), 'MSE': mse, 'RMSE': float(np.sqrt(mse)),
            'MAE': float(np.mean(abs(errors))),
            'R2': float(1-np.sum(errors**2)/variance) if variance > 0 else None}
