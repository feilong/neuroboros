import operator

import numpy as np


def kfold_bagging(n, n_folds=5, n_reps=20, seed=0):
    """Generate train and test indices for k-fold bagging.

    Similar to bagging, but ensures that each sample is used as a test sample
    at least once in each repetition by incorporating k-fold cross-validation.

    In each repetition, the samples are randomly shuffled and divided into
    k folds. Each time, samples in one of the k folds is withheld as test
    candidates, and the remaining samples are used as training candidates.
    The training candidates are then resampled with replacement to generate
    the final training set. The remaining samples are used as the test set.
    In other words, the test set is the union of the withheld fold and the
    training candidates that were not selected for.

    The total number of train-test pairs is n_reps * n_folds.

    Parameters
    ----------
    n : int
        Number of samples.
    n_folds : int, default=5
        Number of folds.
    n_reps : int, default=20
        Number of repetitions of the k-fold procedure.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    indices_li : list of tuples
        List of tuples of training and test indices.
    """
    rng = np.random.default_rng(seed=seed)
    arng = np.arange(n)

    indices_li = []
    for i in range(n_reps):
        folds = np.array_split(rng.permutation(n), n_folds)
        for test_idx in folds:
            train_idx = np.setdiff1d(arng, test_idx)
            train_idx = rng.choice(train_idx, size=n, replace=True)
            test_idx = np.setdiff1d(arng, train_idx)
            indices_li.append((train_idx, test_idx))

    return indices_li


def kfold_bagging_groups(groups, n_folds=5, n_reps=20, seed=0):
    """Generate train and test indices for k-fold bagging with group constraints.

    Like kfold_bagging, but operates at the group level so that members of the
    same group are never split across training and test sets. Groups are assigned
    to folds while balancing by sample count: larger groups are placed first,
    and each group is assigned to a randomly chosen fold that still has room.

    The total number of train-test pairs is n_reps * n_folds.

    Notes
    -----
    The fold assignment assumes that there are enough small groups (e.g.,
    single-member groups) to fill folds to their expected size. If all
    groups are large relative to the expected fold size, a group may not
    fit into any fold, causing an infinite loop. For example, with two
    groups of sizes 1 and 5 and n_folds=2, the expected fold size is 3,
    but the group of size 5 cannot be placed.

    Parameters
    ----------
    groups : list of arrays
        Each array contains the sample indices of members of one group.
    n_folds : int, default=5
        Number of folds.
    n_reps : int, default=20
        Number of repetitions of the k-fold procedure.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    indices_li : list of tuples
        List of ``(train_idx, test_groups)`` tuples, where ``train_idx`` is an
        array of observation indices used for training and ``test_groups`` is an
        array of group indices that have no overlap with ``train_idx``.
    """
    rng = np.random.default_rng(seed=seed)
    groups = [np.asarray(g) for g in groups]
    n_groups = len(groups)
    all_indices = np.concatenate(groups)
    expected_sizes = [len(_) for _ in np.array_split(all_indices, n_folds)]

    size_to_idxs = {}
    for i, g in enumerate(groups):
        size_to_idxs.setdefault(len(g), []).append(i)
    lengths = sorted(size_to_idxs.keys())[::-1]

    indices_li = []
    for _ in range(n_reps):
        d = {
            l: list(idxs) for l, idxs in size_to_idxs.items()
        }  # copy because pop() mutates it
        folds = [[] for _ in range(n_folds)]
        current_sizes = np.zeros(n_folds, dtype=int)
        for l in lengths:
            while d[l]:
                idx = d[l].pop(rng.choice(len(d[l])))
                choices = [
                    k
                    for k in range(n_folds)
                    if current_sizes[k] + l <= expected_sizes[k]
                ]
                if not choices:
                    raise ValueError(
                        f"Group of size {l} cannot fit in any fold. "
                        f"Ensure there are enough small groups to balance the folds."
                    )
                k = rng.choice(choices)
                folds[k].append(idx)
                current_sizes[k] += l

        for withheld_idxs in folds:
            candidate_group_idxs = np.setdiff1d(np.arange(n_groups), withheld_idxs)
            candidates = np.concatenate([groups[i] for i in candidate_group_idxs])
            train_idx = rng.choice(candidates, size=len(all_indices), replace=True)
            train_idx = np.unique(train_idx)
            test_groups = np.sort(
                [gi for gi, g in enumerate(groups) if not np.isin(g, train_idx).any()]
            )
            indices_li.append((train_idx, test_groups))

    return indices_li


def _check_int(name, value, minimum):
    """Return ``value`` as an int, which must be at least ``minimum``."""
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"`{name}` must be an integer, got {value!r}.") from None
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}, got {value}.")
    return value


def _with_neighbors(idx, n, buffersize):
    """Mark the samples ``idx`` and their ``buffersize`` neighbors on each side."""
    marked = np.zeros(n, dtype=bool)
    for offset in range(-buffersize, buffersize + 1):
        shifted = idx + offset
        marked[shifted[(shifted >= 0) & (shifted < n)]] = True
    return marked


def kfold_bagging_blocks(
    n, n_folds=5, n_reps=20, blocksize=4, buffersize=4, seed=0  # , mask=None
):
    """Generate train and test indices for k-fold bagging of a time series.

    Like `kfold_bagging`, but for samples ordered in time, such as fMRI
    time points, whose neighbors are correlated. The folds are formed from
    blocks of consecutive samples rather than from single samples, and a
    buffer of ``buffersize`` samples separates the training and test sets
    on both sides, so that no test sample is adjacent to a training sample
    when ``buffersize > 0``.

    In each repetition, the samples are divided into consecutive blocks of
    ``blocksize`` samples, and the blocks are randomly divided into k
    folds. When ``n`` is not a multiple of ``blocksize``, the remaining
    samples are randomly divided between the first and the last block, so
    that every sample belongs to a block. Each time, the samples of one
    fold are withheld, and the remaining samples, except those within the
    buffer of a withheld sample, are the training candidates, which are
    resampled with replacement to generate the training set. The test set
    contains the samples whose distance from every selected training
    sample exceeds ``buffersize``. As in `kfold_bagging`, every sample is
    withheld once per repetition, so it is a test sample at least once in
    each repetition.

    The total number of train-test pairs is n_reps * n_folds.

    Parameters
    ----------
    n : int
        Number of samples in the time series.
    n_folds : int, default=5
        Number of folds.
    n_reps : int, default=20
        Number of repetitions of the k-fold procedure.
    blocksize : int, default=4
        Number of consecutive samples in a block.
    buffersize : int, default=4
        Number of samples on each side of a test sample that are excluded
        from training, and vice versa. Zero disables the buffer.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    indices_li : list of tuples
        List of tuples of training and test indices. The training indices
        contain repeats, because the training set is a bootstrap sample
        of the training candidates with ``n`` samples.
    """
    # Parameter kept for later:
    # mask : ndarray of bool of shape (n,) or None, default=None
    #     Samples to use, e.g., to exclude censored time points. Excluded
    #     samples are never in the training or test sets, but they still
    #     count as neighbors when the buffers are applied. The returned
    #     indices refer to the ``mask.sum()`` samples that are kept, i.e.,
    #     they index ``data[mask]`` rather than ``data``.
    n = _check_int("n", n, 1)
    n_folds = _check_int("n_folds", n_folds, 2)
    n_reps = _check_int("n_reps", n_reps, 1)
    blocksize = _check_int("blocksize", blocksize, 1)
    buffersize = _check_int("buffersize", buffersize, 0)
    n_blocks, remainder = divmod(n, blocksize)
    if n_blocks < n_folds:
        raise ValueError(
            f"{n} samples form {n_blocks} blocks of {blocksize}, fewer than "
            f"the {n_folds} folds."
        )

    # if mask is None:
    #     mask = np.ones(n, dtype=bool)
    # else:
    #     mask = np.asarray(mask, dtype=bool)
    #     if mask.shape != (n,):
    #         raise ValueError(f"`mask` must have shape ({n},), got {mask.shape}.")
    # # Index of each sample among the kept samples.
    # kept = np.cumsum(mask) - 1
    # n_kept = int(mask.sum())

    rng = np.random.default_rng(seed=seed)
    arng = np.arange(n)
    indices_li = []
    for _ in range(n_reps):
        # The first block starts at 0 and the last ends at n, so the block
        # boundaries absorb the remaining samples at a random split.
        shift = rng.integers(remainder + 1)
        boundaries = shift + np.arange(1, n_blocks) * blocksize
        blocks = np.split(arng, boundaries)
        folds = np.array_split(rng.permutation(n_blocks), n_folds)
        for test_blocks in folds:
            withheld = np.concatenate([blocks[b] for b in test_blocks])
            candidates = np.where(~_with_neighbors(withheld, n, buffersize))[0]
            # candidates = np.where(mask & ~_with_neighbors(withheld, n, buffersize))[0]
            if not len(candidates):
                raise ValueError(
                    "No training candidates remain; reduce `buffersize` or "
                    "the fraction of withheld samples."
                )
            train_idx = rng.choice(candidates, size=n, replace=True)
            # train_idx = rng.choice(candidates, size=n_kept, replace=True)
            excluded = _with_neighbors(np.unique(train_idx), n, buffersize)
            test_idx = np.where(~excluded)[0]
            # test_idx = np.where(mask & ~excluded)[0]
            indices_li.append((train_idx, test_idx))
            # indices_li.append((kept[train_idx], kept[test_idx]))

    return indices_li


def permute_groups(groups, n_perms, seed=0):
    """Generate permutations of group members.

    In each permutation, groups of the same size are randomly reordered among
    themselves, and members within each placed group are shuffled. Members
    never cross group boundaries.

    Parameters
    ----------
    groups : list of arrays
        Each array contains the sample indices of members of one group.
        ``np.sort(np.concatenate(groups))`` must equal ``np.arange(n)``.
    n_perms : int
        Number of permutations to generate.
    seed : int, default=0
        Random seed for the random number generator.

    Returns
    -------
    perms : ndarray of int, shape (n_perms, n)
        Each row is a permutation array. ``data[perms[p]]`` gives the
        permuted data for permutation ``p``.
    """
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g) for g in groups]
    n = sum(len(g) for g in groups)

    size_to_group_idxs = {}
    for i, g in enumerate(groups):
        size_to_group_idxs.setdefault(len(g), []).append(i)

    perms = np.empty((n_perms, n), dtype=int)
    for p in range(n_perms):
        perm = perms[p]
        for size, group_idxs in size_to_group_idxs.items():
            shuffled_idxs = rng.permutation(group_idxs)
            for new_pos, old_idx in zip(group_idxs, shuffled_idxs):
                perm[groups[new_pos]] = rng.permutation(groups[old_idx])

    return perms
