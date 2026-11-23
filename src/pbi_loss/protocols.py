import numpy as np


def make_folds(frame, protocol, seed=42, test_days=None):
    dates = frame.date.to_numpy()
    days = np.array(sorted(set(dates)))
    if protocol == 'lodo':
        masks = [(day, dates == day) for day in days]
    elif protocol == 'random_days':
        if len(days) < 4:
            raise ValueError('random_days needs at least four acquisition days')
        chosen = days[np.random.default_rng(seed).permutation(len(days))[:3]]
        masks = [(f'seed_{seed}', np.isin(dates, chosen))]
    elif protocol == 'images':
        if len(frame) < 5:
            raise ValueError('images needs at least five samples')
        mask = np.zeros(len(frame), dtype=bool)
        mask[np.random.default_rng(seed).permutation(len(frame))[:round(0.2*len(frame))]] = True
        masks = [(f'seed_{seed}', mask)]
    elif protocol == 'fixed_days':
        if not test_days or set(test_days)-set(days):
            raise ValueError('Supply test dates that all occur in the metadata')
        masks = [('specified_dates', np.isin(dates, test_days))]
    else:
        raise ValueError(f'Unknown protocol: {protocol}')
    for name, test in masks:
        train = ~test
        if not train.any() or not test.any():
            raise ValueError('Each split needs nonempty training and test sets')
        if protocol != 'images' and set(dates[train]) & set(dates[test]):
            raise ValueError('Training and test days overlap')
        yield name, np.flatnonzero(train), np.flatnonzero(test)
