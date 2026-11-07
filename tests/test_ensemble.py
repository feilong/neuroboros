import numpy as np
import pytest
from joblib import cpu_count

import neuroboros as nb
from neuroboros.ensemble import (
    kfold_bagging_blocks,
    kfold_bagging_groups,
    permute_groups,
)


class TestEnsemble:
    def test_kfold_bagging(self):
        n = 500
        indices_li = nb.ensemble.kfold_bagging(n, n_folds=20, n_reps=20, seed=0)

        test_idx = np.concatenate([_[1] for _ in indices_li])
        uu, cc = np.unique(test_idx, return_counts=True)
        counts_test = np.zeros((n,))
        counts_test[uu] = cc
        assert np.all(counts_test >= 20)

        for train_idx, test_idx in indices_li:
            assert np.all(np.intersect1d(train_idx, test_idx) == [])

    def test_kfold_bagging_groups_no_group_overlap(self):
        rng = np.random.default_rng(42)
        sizes = rng.choice([1, 2, 3, 4], size=30)
        groups, idx = [], 0
        for s in sizes:
            groups.append(np.arange(idx, idx + s))
            idx += s
        n_groups = len(groups)
        n_perms, n_folds = 10, 5
        splits = kfold_bagging_groups(groups, n_folds=n_folds, n_reps=n_perms, seed=0)
        assert len(splits) == n_perms * n_folds
        for train_idx, test_groups in splits:
            # test groups have no observation overlap with train_idx
            for gi in test_groups:
                assert not np.isin(groups[gi], train_idx).any()
            # train groups have at least one member in train_idx
            train_groups = np.setdiff1d(np.arange(n_groups), test_groups)
            for gi in train_groups:
                assert np.isin(groups[gi], train_idx).any()

    def test_kfold_bagging_groups_each_sample_tested_n_perms_times(self):
        rng = np.random.default_rng(42)
        sizes = rng.choice([1, 2, 3, 4], size=30)
        groups, idx = [], 0
        for s in sizes:
            groups.append(np.arange(idx, idx + s))
            idx += s
        n = idx
        n_perms, n_folds = 10, 5
        splits = kfold_bagging_groups(groups, n_folds=n_folds, n_reps=n_perms, seed=0)
        all_test = np.concatenate(
            [
                np.concatenate([groups[gi] for gi in test_groups])
                for _, test_groups in splits
            ]
        )
        counts = np.bincount(all_test, minlength=n)
        assert np.all(counts >= n_perms)

    def test_kfold_bagging_blocks_buffer(self):
        n, blocksize, buffersize = 503, 4, 3
        splits = kfold_bagging_blocks(
            n, n_folds=5, n_reps=4, blocksize=blocksize, buffersize=buffersize
        )
        assert len(splits) == 20
        for train_idx, test_idx in splits:
            assert len(train_idx) == n
            assert len(test_idx) > 0
            assert np.all(np.diff(test_idx) > 0), "test indices are sorted"
            # Every test sample is more than buffersize away from every
            # training sample, and nothing else is kept out of the test set.
            train_unique = np.unique(train_idx)
            gap = np.abs(test_idx[:, None] - train_unique[None, :]).min(axis=1)
            assert gap.min() > buffersize
            rest = np.setdiff1d(np.arange(n), np.union1d(test_idx, train_unique))
            gap_rest = np.abs(rest[:, None] - train_unique[None, :]).min(axis=1)
            assert gap_rest.max() <= buffersize

    @pytest.mark.parametrize("n", [200, 203])
    def test_kfold_bagging_blocks_withheld_fold(self, n):
        # Without a buffer, the test set is exactly the samples not trained
        # on, and the withheld sets of one repetition are unions of whole
        # blocks that tile the series.
        blocksize, n_folds, n_reps = 5, 4, 3
        splits = kfold_bagging_blocks(
            n, n_folds=n_folds, n_reps=n_reps, blocksize=blocksize, buffersize=0
        )
        n_blocks, remainder = divmod(n, blocksize)
        for rep in range(n_reps):
            withheld_sets = []
            for train_idx, test_idx in splits[rep * n_folds : (rep + 1) * n_folds]:
                assert not np.intersect1d(train_idx, test_idx).size
                untrained = np.setdiff1d(np.arange(n), train_idx)
                np.testing.assert_array_equal(test_idx, untrained)
                withheld_sets.append(set(untrained))
            # The block boundaries of the repetition are not known, so try
            # every possible split of the remaining samples. For one of
            # them, every block is absent from training and present in
            # testing as a whole in at least one fold. (A block can also
            # be left out of a bootstrap by chance, so not exactly one.)
            matches = 0
            for shift in range(remainder + 1):
                boundaries = shift + np.arange(1, n_blocks) * blocksize
                blocks = np.split(np.arange(n), boundaries)
                assert len(blocks) == n_blocks
                matches += all(
                    any(set(block) <= withheld for withheld in withheld_sets)
                    for block in blocks
                )
            assert matches >= 1

    @pytest.mark.parametrize("n", [400, 23])
    @pytest.mark.parametrize("buffersize", [0, 4])
    def test_kfold_bagging_blocks_each_sample_tested(self, n, buffersize):
        n_folds, n_reps = 5, 10
        splits = kfold_bagging_blocks(
            n, n_folds=n_folds, n_reps=n_reps, buffersize=buffersize, seed=4
        )
        for rep in range(n_reps):
            tested = np.concatenate(
                [t for _, t in splits[rep * n_folds : (rep + 1) * n_folds]]
            )
            assert np.all(np.bincount(tested, minlength=n) >= 1)

    # def test_kfold_bagging_blocks_mask(self):
    #     n = 120
    #     mask = np.ones(n, dtype=bool)
    #     mask[40:50] = False
    #     mask[77] = False
    #     kept = np.where(mask)[0]
    #     splits = kfold_bagging_blocks(
    #         n, n_folds=4, n_reps=3, buffersize=2, mask=mask
    #     )
    #     for train_idx, test_idx in splits:
    #         assert len(train_idx) == mask.sum()
    #         assert train_idx.max() < mask.sum() and test_idx.max() < mask.sum()
    #         # Map back to the original time points and check the buffer.
    #         train_t, test_t = kept[np.unique(train_idx)], kept[test_idx]
    #         gap = np.abs(test_t[:, None] - train_t[None, :]).min(axis=1)
    #         assert gap.min() > 2

    def test_kfold_bagging_blocks_seed_and_errors(self):
        a = kfold_bagging_blocks(100, n_reps=2, seed=3)
        b = kfold_bagging_blocks(100, n_reps=2, seed=3)
        for (ta, sa), (tb, sb) in zip(a, b):
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(sa, sb)
        with pytest.raises(ValueError, match="fewer than"):
            kfold_bagging_blocks(16, n_folds=5, blocksize=4)
        with pytest.raises(ValueError, match="blocksize"):
            kfold_bagging_blocks(100, blocksize=0)
        with pytest.raises(ValueError, match="buffersize"):
            kfold_bagging_blocks(100, buffersize=-1)
        with pytest.raises(ValueError, match="n_reps"):
            kfold_bagging_blocks(100, n_reps=-1)
        with pytest.raises(ValueError, match="n_folds"):
            kfold_bagging_blocks(100, n_folds=1)
        with pytest.raises(TypeError, match="blocksize"):
            kfold_bagging_blocks(100, blocksize=4.0)
        with pytest.raises(ValueError, match="No training candidates"):
            kfold_bagging_blocks(40, n_folds=5, blocksize=4, buffersize=40)

    def test_permute_groups_shape(self):
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6])]
        perms = permute_groups(groups, n_perms=10, seed=0)
        assert perms.shape == (10, 7)
        assert perms.dtype == int

    def test_permute_groups_valid_permutation(self):
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6])]
        perms = permute_groups(groups, n_perms=20, seed=0)
        for perm in perms:
            np.testing.assert_array_equal(np.sort(perm), np.arange(7))

    def test_permute_groups_no_cross_group_mixing(self):
        # perm[g] must always be exactly the members of one original same-size group
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6])]
        size2_members = {0, 1, 2, 3}
        size3_members = {4, 5, 6}
        original_size2_groups = [frozenset([0, 1]), frozenset([2, 3])]
        perms = permute_groups(groups, n_perms=50, seed=0)
        for perm in perms:
            for g in groups:
                pg = set(perm[g])
                if len(g) == 2:
                    assert pg in original_size2_groups
                else:
                    assert pg == size3_members

    def test_permute_groups_same_group_members_stay_together(self):
        # members that belong to the same original group must always
        # appear together in the same group slot after permutation
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7])]
        original_groups = [frozenset(g) for g in groups]
        perms = permute_groups(groups, n_perms=100, seed=0)
        for perm in perms:
            for g in groups:
                # all members of g map to the same original group
                assert frozenset(perm[g]) in original_groups

    def test_permute_groups_perm_g_gives_permuted_group(self):
        # [perm[g] for g in groups] should reconstruct a valid permuted partition
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6])]
        perms = permute_groups(groups, n_perms=20, seed=0)
        for perm in perms:
            permuted = [perm[g] for g in groups]
            # permuted is a valid partition of arange(7)
            np.testing.assert_array_equal(
                np.sort(np.concatenate(permuted)), np.arange(7)
            )
            # each permuted group has the same size as the original
            for g, pg in zip(groups, permuted):
                assert len(pg) == len(g)

    def test_permute_groups_groups_swap(self):
        # over many permutations, same-size groups should swap
        groups = [np.array([0, 1]), np.array([2, 3])]
        perms = permute_groups(groups, n_perms=200, seed=0)
        swapped = sum(1 for p in perms if set(p[[0, 1]]) == {2, 3})
        assert 30 < swapped < 170  # roughly 50% swap rate

    def test_permute_groups_members_shuffled_within_group(self):
        # within-group order should vary across permutations
        groups = [np.array([0, 1, 2, 3])]
        perms = permute_groups(groups, n_perms=200, seed=0)
        unique_orderings = {tuple(p) for p in perms}
        assert len(unique_orderings) > 1
        for ordering in unique_orderings:
            np.testing.assert_array_equal(np.sort(ordering), [0, 1, 2, 3])

    def test_permute_groups_seed_reproducibility(self):
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5, 6])]
        perms1 = permute_groups(groups, n_perms=10, seed=42)
        perms2 = permute_groups(groups, n_perms=10, seed=42)
        np.testing.assert_array_equal(perms1, perms2)

    def test_permute_groups_several_singletons(self):
        # several size-1 groups: no within-group shuffle possible,
        # but the groups themselves are shuffled across positions
        groups = [np.array([i]) for i in range(5)]
        perms = permute_groups(groups, n_perms=200, seed=0)
        for perm in perms:
            np.testing.assert_array_equal(np.sort(perm), np.arange(5))
        # groups should not always stay in the same position
        unique_perms = {tuple(p) for p in perms}
        assert len(unique_perms) > 1

    def test_permute_groups_single_group_of_size(self):
        # only one group of its size: no group swap possible,
        # only within-group member shuffle
        groups = [np.array([0, 1, 2])]
        perms = permute_groups(groups, n_perms=200, seed=0)
        for perm in perms:
            # same members, possibly different order
            np.testing.assert_array_equal(np.sort(perm[[0, 1, 2]]), [0, 1, 2])
        unique_orderings = {tuple(p) for p in perms}
        assert len(unique_orderings) > 1

    def test_permute_groups_multiple_groups_same_size(self):
        # multiple groups of the same size: both group-level swap and
        # within-group member shuffle occur
        groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
        original_groups = [frozenset(g) for g in groups]
        perms = permute_groups(groups, n_perms=200, seed=0)
        for perm in perms:
            np.testing.assert_array_equal(np.sort(perm), np.arange(6))
            # each position receives members of exactly one original group
            for g in groups:
                assert frozenset(perm[g]) in original_groups
        # group positions should vary: group 0's slot gets different members
        slot0_members = {frozenset(perm[[0, 1]]) for perm in perms}
        assert len(slot0_members) > 1
        # within-group order should also vary
        unique_perms = {tuple(perm) for perm in perms}
        assert len(unique_perms) > 1

    def test_ensemble_lstsq(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 50))
        beta0 = rng.standard_normal((50, 20))
        Y = rng.standard_normal((100, 20)) * 0.1 + X @ beta0

        beta, Yhat, R2, r = nb.linalg.ensemble_lstsq(X, Y, n_folds=5, n_reps=20, seed=0)

        assert beta.shape == (50, 20)
        assert Yhat.shape == (100, 20)
        assert R2.shape == (20,)
        assert r.shape == (20,)

        assert np.all(r > 0)
        assert np.all(R2 > 0)

    def test_ensemble_lstsq_parallel(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 50))
        beta0 = rng.standard_normal((50, 20))
        Y = rng.standard_normal((100, 20)) * 0.1 + X @ beta0

        beta_1, Yhat_1, R2_1, r_1 = nb.linalg.ensemble_lstsq(
            X, Y, n_folds=5, n_reps=20, seed=0, n_jobs=1
        )
        beta_2, Yhat_2, R2_2, r_2 = nb.linalg.ensemble_lstsq(
            X, Y, n_folds=5, n_reps=20, seed=0, n_jobs=max(cpu_count(), 2)
        )
        np.testing.assert_allclose(beta_1, beta_2)
        np.testing.assert_allclose(Yhat_1, Yhat_2)
        np.testing.assert_allclose(R2_1, R2_2)
        np.testing.assert_allclose(r_1, r_2)
