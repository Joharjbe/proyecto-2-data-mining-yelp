"""Pruebas pequeñas y deterministas de los algoritmos implementados a mano."""
import unittest

import numpy as np
import pandas as pd

from src import clustering, critical_analysis, dimensionality_reduction, graphs, recommenders, streaming


class GraphTests(unittest.TestCase):
    def test_pagerank_web_1839(self):
        # Deck 07, pág. 18: y->{y,a}, a->{y,m}, m->{a}.
        origin = np.array([0, 0, 1, 1, 2])
        destination = np.array([0, 1, 0, 2, 1])
        rank = graphs.pagerank(
            graphs.construir_csr(origin, destination, 3),
            beta=1.0,
            eps=1e-12,
            verbose=False,
        )
        np.testing.assert_allclose(rank, [0.4, 0.4, 0.2], atol=1e-10)

    def test_hits_and_modularity(self):
        # Dos hubs apuntan a authority 2; solo uno apunta a authority 3.
        hub, authority = graphs.hits(
            np.array([0, 1, 1]), np.array([2, 2, 3]), 4, verbose=False
        )
        self.assertGreater(authority[2], authority[3])
        q = graphs.modularidad(
            np.array([0, 0, 1, 1]), np.array([0, 2]), np.array([1, 3]), 4
        )
        self.assertAlmostEqual(q, 0.5)

    def test_spearman_uses_average_ties(self):
        result = graphs.spearman([1, 1, 2], [1, 2, 3])
        self.assertAlmostEqual(result, np.sqrt(3) / 2)


class ClusteringTests(unittest.TestCase):
    def test_kmeans_dbscan_and_summary(self):
        x = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
        result = clustering.kmeans(x, 2, seed=42)
        np.testing.assert_allclose(np.sort(result["centroids"].ravel()), [2, 11])
        self.assertAlmostEqual(result["sse"], 4.0)

        db = np.array([[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1], [12, 12]])
        labels = clustering.dbscan(db, eps=0.25, min_pts=3)["labels"]
        self.assertEqual(len(set(labels[labels >= 0])), 2)
        self.assertEqual(labels[-1], -1)

        summary = clustering.Summary.from_points(np.array([[1.0], [2.0], [3.0]]))
        self.assertEqual(summary.n, 3)
        np.testing.assert_allclose(summary.centroid, [2.0])
        np.testing.assert_allclose(summary.var, [2 / 3])

    def test_k_distance_handles_large_k(self):
        values = clustering.k_distance_values(
            np.array([[0.0], [1.0], [3.0]]), k=20, sample_size=3, seed=1
        )
        self.assertEqual(len(values), 3)
        self.assertTrue(np.isfinite(values).all())


class RecommenderTests(unittest.TestCase):
    def test_similarity_and_empty_neighbor_case(self):
        rows = [
            ("u0", "i0", 5), ("u0", "i1", 5), ("u0", "i2", 1),
            ("u1", "i0", 4), ("u1", "i1", 4), ("u1", "i2", 2),
            ("u2", "i0", 1), ("u2", "i1", 1), ("u2", "i2", 5),
        ]
        matrix = recommenders.matriz_train(pd.DataFrame(rows, columns=["user_id", "business_id", "stars"]))
        neighbors = recommenders.similitud_item_item(matrix, min_coratings=2, shrink=0, topk=2)
        i0, i1 = matrix["i2x"]["i0"], matrix["i2x"]["i1"]
        a, b = neighbors["indptr"][i0:i0 + 2]
        scores = dict(zip(neighbors["vecino"][a:b], neighbors["sim"][a:b]))
        self.assertGreater(scores[i1], 0.9)

        singleton = recommenders.matriz_train(
            pd.DataFrame([("u0", "i0", 5)], columns=["user_id", "business_id", "stars"])
        )
        empty = recommenders.similitud_item_item(singleton)
        self.assertEqual(len(empty["sim"]), 0)
        np.testing.assert_array_equal(empty["indptr"], [0, 0])

    def test_metrics(self):
        rel = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(recommenders.precision_at_k(rel, 2), 0.5)
        self.assertAlmostEqual(recommenders.recall_at_k(rel, 2, 2), 0.5)
        self.assertGreater(recommenders.ndcg_at_k(rel, 4), 0.9)
        self.assertAlmostEqual(recommenders.rmse([1, 3], [1, 1]), np.sqrt(2))


class StreamingTests(unittest.TestCase):
    def test_window_cms_and_dgim(self):
        window = streaming.SlidingWindow(4)
        sums = [window.update(t, value).total for t, value in enumerate([1, 2, 3, 4, 5])]
        self.assertEqual(sums, [1, 3, 6, 10, 14])

        codes = np.array([0, 1, 0, 2, 0, 1, 3, 0])
        cms = streaming.CountMinSketch(width=5, depth=2, seed=42)
        cms.update_batch(codes)
        self.assertTrue(np.all(cms.query_batch(np.arange(4)) >= [4, 2, 1, 1]))
        with self.assertRaises(ValueError):
            cms.update(-1)

        rng = np.random.default_rng(42)
        bits = (rng.random(500) < 0.35).astype(int)
        audit = streaming.evaluate_dgim(bits, window_size=64)
        self.assertTrue(audit.invariant_ok.all())
        self.assertLessEqual(audit.loc[audit.exact > 0, "relative_error"].max(), 0.5 + 1e-12)


class DimensionalityTests(unittest.TestCase):
    def test_pca_and_svd_reconstruction(self):
        x = np.array([[2.0, 0], [0, 1], [-2, 0], [0, -1]])
        pca = dimensionality_reduction.pca_fit(x)
        reconstructed = dimensionality_reduction.pca_inverse(pca.scores, pca)
        self.assertLess(dimensionality_reduction.relative_frobenius_error(x, reconstructed), 1e-12)

        a = np.array([[1.0, 0, 1], [0, 1, 1], [1, 1, 2], [2, 1, 3]])
        svd = dimensionality_reduction.randomized_svd_csr(
            dimensionality_reduction.dense_to_csr(a), k=2, oversampling=1, power_iterations=2
        )
        approx = (svd.U * svd.singular_values) @ svd.Vt
        self.assertLess(dimensionality_reduction.relative_frobenius_error(a, approx), 1e-6)
        self.assertEqual(dimensionality_reduction.relative_frobenius_error(np.zeros((2, 2)), np.zeros((2, 2))), 0)


class CriticalAnalysisTests(unittest.TestCase):
    def test_concentration_and_attack(self):
        self.assertAlmostEqual(critical_analysis.gini([1, 1, 1]), 0)
        self.assertAlmostEqual(critical_analysis.top_share([1, 2, 7], 1 / 3), 0.7)
        stats = pd.DataFrame({"business_id": ["a"], "n_reviews": [5], "mean_stars": [3.0]})
        detail, _ = critical_analysis.review_attack_stress(stats, attacks=(5,), target_stars=(5,))
        self.assertAlmostEqual(detail.iloc[0].mean_after, 4.0)


if __name__ == "__main__":
    unittest.main()

