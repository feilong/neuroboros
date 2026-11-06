import numpy as np
import pytest

import neuroboros as nb
from neuroboros.searchlights import _UnpackedSearchlights


class TestSearchlights:
    def test_basic(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                sls = nb.sls(lr, 20, space, mask=False)
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == 10241
                assert np.concatenate(sls).min() == 0

    def test_masked(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                sls = nb.sls(lr, 20, space, mask=True)
                nv = nb.mask(lr, space).sum()
                assert len(sls) == nv
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == nv - 1
                assert np.concatenate(sls).min() == 0

    def test_radius(self):
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            for lr in "lr":
                radius = 10
                dists = nb.sls(lr, radius, space, mask=False, return_dists=True)[1]
                assert all([np.all(_ <= radius) for _ in dists])
                dists = nb.sls(lr, radius, space, mask=True, return_dists=True)[1]
                assert all([np.all(_ <= radius) for _ in dists])

    def test_center_space(self):
        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(lr, 20, space, center_space=center_space, mask=False)
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).max() == 40961
                assert np.concatenate(sls).min() == 0

    def test_center_mask(self):
        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=True,
                    mask=False,
                )
                assert len(sls) == nb.mask(lr, center_space).sum()
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).min() == 0
                sls2 = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=True,
                    mask=True,
                )
                assert len(sls2) == nb.mask(lr, center_space).sum()
                assert all([isinstance(_, np.ndarray) for _ in sls2])
                assert np.concatenate(sls2).min() == 0
                assert not all(
                    [np.array_equal(sl1, sl2) for sl1, sl2 in zip(sls, sls2)]
                )

        for space in [_ + "-ico64" for _ in ["onavg", "fsavg", "fslr"]]:
            center_space = space.replace("-ico64", "-ico32")
            for lr in "lr":
                sls = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=False,
                    mask=True,
                )
                assert len(sls) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls])
                assert np.concatenate(sls).min() == 0
                sls2 = nb.sls(
                    lr,
                    20,
                    space,
                    center_space=center_space,
                    center_mask=False,
                    mask=False,
                )
                assert len(sls2) == 10242
                assert all([isinstance(_, np.ndarray) for _ in sls2])
                assert np.concatenate(sls2).min() == 0
                assert not all(
                    [np.array_equal(sl1, sl2) for sl1, sl2 in zip(sls, sls2)]
                )

    def test_lr(self):
        rng = np.random.default_rng(0)
        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            lh = rng.standard_normal((10, 10242))
            rh = rng.standard_normal((10, 10242))
            dm = np.concatenate([lh, rh], axis=1)
            sls = nb.sls("lr", 20, space, mask=False)
            sls_l = nb.sls("l", 20, space, mask=False)
            sls_r = nb.sls("r", 20, space, mask=False)
            lh_m = np.stack([lh[:, sl].mean(axis=1) for sl in sls_l], axis=1)
            rh_m = np.stack([rh[:, sl].mean(axis=1) for sl in sls_r], axis=1)
            m = np.stack([dm[:, sl].mean(axis=1) for sl in sls], axis=1)
            np.testing.assert_allclose(
                m, np.concatenate([lh_m, rh_m], axis=1), atol=1e-6
            )

        for space in [_ + "-ico32" for _ in ["onavg", "fsavg", "fslr", "mkavg"]]:
            lh_nv = nb.mask("l", space).sum()
            rh_nv = nb.mask("r", space).sum()
            lh = rng.standard_normal((10, lh_nv))
            rh = rng.standard_normal((10, rh_nv))
            dm = np.concatenate([lh, rh], axis=1)
            sls = nb.sls("lr", 20, space, mask=True)
            sls_l = nb.sls("l", 20, space, mask=True)
            sls_r = nb.sls("r", 20, space, mask=True)
            lh_m = np.stack([lh[:, sl].mean(axis=1) for sl in sls_l], axis=1)
            rh_m = np.stack([rh[:, sl].mean(axis=1) for sl in sls_r], axis=1)
            m = np.stack([dm[:, sl].mean(axis=1) for sl in sls], axis=1)
            np.testing.assert_allclose(
                m, np.concatenate([lh_m, rh_m], axis=1), atol=1e-6
            )


def _toy_searchlights(nv=12, with_dists=True):
    rng = np.random.default_rng(0)
    sls, dists = [], []
    for center in range(nv):
        n = rng.integers(3, 6)
        sl = np.unique(np.r_[center, rng.choice(nv, n, replace=False)])
        d = np.abs(sl - center).astype(float)
        sls.append(sl)
        dists.append(d)
    if not with_dists:
        return sls, None
    return sls, dists


def _fit_in_worker(sls, seed, max_nbytes):
    """Fit a transform in a joblib worker and report what was shared."""
    import pickle

    from neuroboros.ha import searchlight_ridge

    structure = sls.structure()
    internal = structure.internal_matrix
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((30, sls.nv))
    Y = rng.standard_normal((30, sls.nv))
    xfm = searchlight_ridge(X, Y, sls, mat0=structure, weights=sls.weights())
    return dict(
        type=type(sls).__name__,
        n=len(sls),
        nv=sls.nv,
        has_dists=sls.dists is not None,
        fingerprint=structure.structure_fingerprint(),
        indices_memmap=isinstance(internal.indices, np.memmap),
        sls_memmap=isinstance(sls[0].base, np.memmap),
        dists_memmap=isinstance(sls.dists[0].base, np.memmap),
        result=pickle.dumps(xfm.to_scipy()),
    )


BOTH = pytest.mark.parametrize(
    "cls", [nb.Searchlights, _UnpackedSearchlights], ids=["packed", "unpacked"]
)


class TestSearchlightsClass:
    @BOTH
    def test_list_behavior(self, cls):
        sls_list, dists = _toy_searchlights()
        sls = cls(
            sls_list, dists, radius=5, nv=12, space="toy", lr="l"
        )
        assert isinstance(sls, list)
        assert type(sls) is cls
        assert len(sls) == len(sls_list)
        assert all(np.array_equal(a, b) for a, b in zip(sls, sls_list))
        assert all(np.array_equal(a, b) for a, b in zip(sls.dists, dists))
        np.testing.assert_array_equal(sls[3], sls_list[3])
        np.testing.assert_array_equal(np.concatenate(sls), np.concatenate(sls_list))
        assert sls.radius == 5 and sls.nv == 12 and sls.space == "toy"
        assert sls.center_space is None and sls.lr == "l"
        assert "n=12" in repr(sls) and "radius=5" in repr(sls)

    @BOTH
    def test_nv_inferred(self, cls):
        sls_list, _ = _toy_searchlights(nv=12, with_dists=False)
        sls = cls(sls_list)
        assert sls.nv == np.concatenate(sls_list).max() + 1
        assert sls.dists is None
        assert cls([]).nv == 0
        assert cls([np.zeros(0, dtype=int)]).nv == 0

    @BOTH
    def test_dists_length_mismatch(self, cls):
        sls_list, dists = _toy_searchlights()
        with pytest.raises(ValueError, match="entries"):
            cls(sls_list, dists[:-1])
        dists[2] = dists[2][:-1]
        with pytest.raises(ValueError, match="differ in length"):
            cls(sls_list, dists)

    def test_packed_entries_are_views(self):
        sls_list, dists = _toy_searchlights()
        sls = nb.Searchlights(sls_list, dists, radius=12)
        assert all(sl.base is sls[0].base for sl in sls)
        assert all(d.base is sls.dists[0].base for d in sls.dists)
        weights = sls.weights()
        assert all(w.base is weights[0].base for w in weights)

    def test_unpacked_entries_are_independent_copies(self):
        sls_list, dists = _toy_searchlights()
        sls = _UnpackedSearchlights(sls_list, dists, radius=12)
        assert all(np.array_equal(a, b) and a is not b for a, b in zip(sls, sls_list))
        assert all(sl.base is None for sl in sls)
        assert sls._packed is None

    @BOTH
    def test_inputs_are_left_untouched(self, cls):
        sls_list, dists = _toy_searchlights()
        cls(sls_list, dists, radius=12)
        assert all(sl.flags.writeable for sl in sls_list)
        assert all(d.flags.writeable for d in dists)

    @BOTH
    def test_immutable(self, cls):
        sls_list, dists = _toy_searchlights()
        sls = cls(sls_list, dists, radius=12)
        first = sls[0]
        for mutate in [
            lambda: sls.__setitem__(0, sls[1]),
            lambda: sls.__delitem__(0),
            lambda: sls.append(sls[1]),
            lambda: sls.extend([sls[1]]),
            lambda: sls.insert(0, sls[1]),
            lambda: sls.pop(),
            lambda: sls.remove(first),
            lambda: sls.clear(),
            lambda: sls.reverse(),
            lambda: sls.sort(),
            lambda: sls.__iadd__([sls[1]]),
            lambda: sls.__imul__(2),
        ]:
            with pytest.raises(TypeError, match="immutable"):
                mutate()
        assert len(sls) == len(sls_list) and sls[0] is first

        for attr in ["dists", "radius", "nv", "space", "center_space", "lr"]:
            with pytest.raises(AttributeError):
                setattr(sls, attr, None)
        with pytest.raises(TypeError):
            sls.dists[0] = sls.dists[1]
        for arr in [sls[0], sls.dists[0], sls.weights()[0]]:
            with pytest.raises(ValueError, match="read-only"):
                arr[0] = 0
        with pytest.raises(ValueError, match="read-only"):
            sls.structure().add_block(sls[0], sls[0], np.ones((len(sls[0]),) * 2))
        # A copy of the structure can be filled in as usual.
        xfm = sls.structure().copy()
        xfm.add_block(sls[0], sls[0], np.ones((len(sls[0]),) * 2))
        assert xfm.internal_matrix.data.any()
        assert not sls.structure().internal_matrix.data.any()

    @BOTH
    @pytest.mark.parametrize("round_trip", ["pickle4", "pickle5", "deepcopy"])
    def test_round_trip_matches_and_stays_read_only(self, cls, round_trip):
        """Workers must see the same searchlights as the parent."""
        import copy
        import pickle

        if round_trip == "deepcopy":
            round_trip = copy.deepcopy
        else:
            protocol = int(round_trip[-1])
            round_trip = lambda obj: pickle.loads(pickle.dumps(obj, protocol=protocol))

        sls_list, dists = _toy_searchlights()
        sls = cls(sls_list, dists, radius=12)
        sls.weights()
        sls.structure()
        loaded = round_trip(sls)
        assert type(loaded) is cls
        assert all(np.array_equal(a, b) for a, b in zip(loaded, sls))
        assert all(np.array_equal(a, b) for a, b in zip(loaded.dists, sls.dists))
        assert all(np.array_equal(a, b) for a, b in zip(loaded.weights(), sls.weights()))
        assert (
            loaded.structure().structure_fingerprint()
            == sls.structure().structure_fingerprint()
        )
        for arr in [loaded[0], loaded.dists[0], loaded.weights()[0]]:
            assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            loaded.structure().add_block(
                loaded[0], loaded[0], np.ones((len(loaded[0]),) * 2)
            )
        assert not loaded.structure().internal_matrix.data.any()

        # Weights computed after the round trip are read-only as well.
        loaded = round_trip(cls(sls_list, dists, radius=12))
        assert not loaded.weights()[0].flags.writeable

    @BOTH
    def test_empty_weights(self, cls):
        import pickle

        assert cls([]).weights() == ()
        weights = cls([np.zeros(0, dtype=int)]).weights()
        assert len(weights) == 1 and weights[0].shape == (0,)
        sls = cls([np.zeros(0, dtype=int)] * 2, [np.zeros(0)] * 2, radius=1)
        weights = sls.weights()
        assert len(weights) == 2 and all(w.shape == (0,) for w in weights)
        assert len(pickle.loads(pickle.dumps(sls)).weights()) == 2

    @BOTH
    def test_weights_uniform(self, cls):
        sls_list, _ = _toy_searchlights(with_dists=False)
        sls = cls(sls_list)
        weights = sls.weights()
        assert sls.weights() is weights
        counts = _vertex_counts(sls)
        total = np.zeros(sls.nv)
        for sl, w in zip(sls, weights):
            np.testing.assert_allclose(w, 1.0 / counts[sl])
            total[sl] += w
        np.testing.assert_allclose(total, 1.0)

    @BOTH
    def test_weights_distance(self, cls):
        sls_list, dists = _toy_searchlights()
        radius = 12
        sls = cls(sls_list, dists, radius=radius)
        weights = sls.weights()
        raw = [(radius - d) / radius for d in dists]
        raw_sum = np.zeros(sls.nv)
        for sl, r in zip(sls, raw):
            raw_sum[sl] += r
        total = np.zeros(sls.nv)
        for sl, r, w in zip(sls, raw, weights):
            np.testing.assert_allclose(w, r / raw_sum[sl])
            total[sl] += w
        np.testing.assert_allclose(total, 1.0)

    def test_weights_distance_requires_radius(self):
        sls_list, dists = _toy_searchlights()
        with pytest.raises(ValueError, match="radius"):
            nb.Searchlights(sls_list, dists).weights()

    @BOTH
    def test_structure(self, cls):
        from neuroboros.ha import Transform

        sls_list, dists = _toy_searchlights()
        sls = cls(sls_list, dists, radius=5, nv=15)
        structure = sls.structure()
        assert isinstance(structure, Transform)
        assert structure.shape == (15, 15)
        assert sls.structure() is structure
        expected = Transform.from_searchlights(sls_list, nv=15)
        assert structure.structure_fingerprint() == expected.structure_fingerprint()
        # `from_searchlights` picks up `nv` from the object as well.
        assert Transform.from_searchlights(sls).shape == (15, 15)
        assert Transform.from_searchlights(sls, sls_out=sls).shape == (15, 15)

    def test_packed_and_unpacked_agree(self):
        sls_list, dists = _toy_searchlights(nv=20)
        packed = nb.Searchlights(sls_list, dists, radius=12, nv=20)
        unpacked = _UnpackedSearchlights(sls_list, dists, radius=12, nv=20)
        assert all(np.array_equal(a, b) for a, b in zip(packed, unpacked))
        assert all(np.array_equal(a, b) for a, b in zip(packed.dists, unpacked.dists))
        assert all(
            np.allclose(a, b) for a, b in zip(packed.weights(), unpacked.weights())
        )
        assert (
            packed.structure().structure_fingerprint()
            == unpacked.structure().structure_fingerprint()
        )
        assert packed.nv == unpacked.nv and len(packed) == len(unpacked)

    @BOTH
    def test_pickle(self, cls):
        import pickle

        sls_list, dists = _toy_searchlights()
        sls = cls(
            sls_list, dists, radius=12, nv=15, space="toy"
        )
        loaded = pickle.loads(pickle.dumps(sls))
        assert isinstance(loaded, nb.Searchlights)
        assert type(loaded) is cls
        assert len(loaded) == len(sls)
        assert all(np.array_equal(a, b) for a, b in zip(loaded, sls))
        assert all(np.array_equal(a, b) for a, b in zip(loaded.dists, dists))
        assert (loaded.radius, loaded.nv, loaded.space) == (12, 15, "toy")
        assert loaded._structure is None
        assert loaded.weights() is not None

        # Cached weights and structure travel with the object.
        sls.weights()
        sls.structure()
        loaded = pickle.loads(pickle.dumps(sls))
        assert all(np.array_equal(a, b) for a, b in zip(loaded._weights, sls._weights))
        assert (
            loaded.structure().structure_fingerprint()
            == sls.structure().structure_fingerprint()
        )

        loaded = pickle.loads(pickle.dumps(cls(sls_list)))
        assert loaded.dists is None and len(loaded) == len(sls_list)
        loaded = pickle.loads(pickle.dumps(cls([])))
        assert len(loaded) == 0 and loaded.nv == 0
        empty = cls(
            [np.zeros(0, dtype=int)], [np.zeros(0)], radius=1
        )
        loaded = pickle.loads(pickle.dumps(empty))
        assert len(loaded) == 1 and len(loaded[0]) == 0 and len(loaded.dists) == 1

    def test_packed_pickle_is_compact(self):
        import io
        import pickle

        sls_list, dists = _toy_searchlights(nv=20)
        packed = nb.Searchlights(sls_list, dists, radius=12, nv=20)
        unpacked = _UnpackedSearchlights(sls_list, dists, radius=12, nv=20)

        def n_arrays(obj):
            """Number of ndarrays pickled, as joblib would see them."""
            seen = []

            class Counter(pickle.Pickler):
                def reducer_override(self, o):
                    if isinstance(o, np.ndarray):
                        seen.append(o)
                    return NotImplemented

            Counter(io.BytesIO()).dump(obj)
            return len(seen)

        # The packed form pickles one array per kind plus the offsets,
        # the unpacked form one per searchlight and one per distance array.
        assert n_arrays(packed) == 3
        assert n_arrays(unpacked) == 2 * len(sls_list)

    @BOTH
    def test_joblib_workers_share_structure(self, cls):
        import pickle

        from joblib import Parallel, delayed

        from neuroboros.ha import searchlight_ridge

        sls_list, dists = _toy_searchlights(nv=20)
        sls = cls(sls_list, dists, radius=12, nv=20)
        structure = sls.structure()
        # ``max_nbytes=1`` makes joblib memory-map every array, as it would
        # for the large arrays of a real structure.
        results = Parallel(n_jobs=2, max_nbytes=1)(
            delayed(_fit_in_worker)(sls, seed, 1) for seed in range(4)
        )
        for seed, r in enumerate(results):
            assert r["type"] == cls.__name__
            assert (r["n"], r["nv"], r["has_dists"]) == (len(sls), 20, True)
            assert r["fingerprint"] == structure.structure_fingerprint()
            assert r["indices_memmap"]
            if cls is nb.Searchlights:
                # Entries are views into one memory-mapped array each.
                assert r["sls_memmap"] and r["dists_memmap"]
            rng = np.random.default_rng(seed)
            X = rng.standard_normal((30, 20))
            Y = rng.standard_normal((30, 20))
            expected = searchlight_ridge(
                X, Y, sls, mat0=structure, weights=sls.weights()
            )
            got = pickle.loads(r["result"])
            np.testing.assert_allclose(got.toarray(), expected.to_scipy().toarray())
        # The parent's structure is still all zeros.
        assert not structure.internal_matrix.data.any()

    def test_get_searchlights_returns_object(self):
        sls = nb.sls("l", 20, "onavg-ico32")
        assert type(sls) is nb.Searchlights
        assert sls.nv == nb.mask("l", "onavg-ico32").sum()
        assert sls.dists is not None and len(sls.dists) == len(sls)
        assert (sls.radius, sls.space, sls.center_space, sls.lr) == (
            20, "onavg-ico32", "onavg-ico32", "l")
        sls2, dists = nb.sls("l", 20, "onavg-ico32", return_dists=True)
        assert isinstance(sls2, nb.Searchlights) and dists is sls2.dists

        assert nb.sls("l", 20, "onavg-ico32", mask=False).nv == 10242

        both = nb.sls("lr", 10, "onavg-ico32")
        assert both.nv == sls.nv + nb.mask("r", "onavg-ico32").sum()
        assert both.radius == 10 and both.lr == "lr"
        assert len(both.dists) == len(both)
        assert np.concatenate(both).max() == both.nv - 1

        unpacked = _UnpackedSearchlights(both, both.dists, both.radius, nv=both.nv)
        assert all(np.array_equal(a, b) for a, b in zip(unpacked, both))
        assert all(np.allclose(a, b) for a, b in zip(unpacked.weights(), both.weights()))


def _vertex_counts(sls):
    counts = np.zeros(sls.nv)
    for sl in sls:
        counts[sl] += 1
    return counts
