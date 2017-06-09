import unittest
import numpy as np
import evaluate


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_labels = np.array([[0], [1], [2]])
        cls.y_labels = np.array([[0], [1], [2]])

        cls.x0 = np.array([0.9, 0.1, 0.0])
        cls.x1 = np.array([0.7, 0.2, 0.1])
        cls.x2 = np.array([0.0, 0.0, 0.5])

        cls.y0 = np.array([0.5, 0.0, 0.0])
        cls.y1 = np.array([0.0, 1.0, 0.0])
        cls.y2 = np.array([0.0, 0.0, 0.2])

        cls.x = np.array([cls.x0, cls.x1, cls.x2])
        cls.y = np.array([cls.y0, cls.y1, cls.y2])

    def test_precision(self):
        self.assertAlmostEqual(
            0.25,
            evaluate.precision(0, [[0], [1], [2], [3]])
        )
        self.assertAlmostEqual(0, evaluate.precision(0, [[1], [2], [3]]))
        self.assertAlmostEqual(0, evaluate.precision(0, []))
        self.assertAlmostEqual(1, evaluate.precision(0, [[0], [0], [0]]))

    def test_closest_docs_by_index(self):
        result = evaluate.closest_docs_by_index(self.x, self.y, 1)
        assert result.shape == (len(self.y), 1)
        assert np.all(result == [[0], [1], [2]])

        result = evaluate.closest_docs_by_index(self.x, self.y, 2)
        assert result.shape == (len(self.y), 2)
        assert np.all(result == [[0, 1], [1, 0], [2, 1]])

        result = evaluate.closest_docs_by_index(self.x, self.y, 3)
        assert result.shape == (len(self.y), 3)
        assert np.all(result == [[0, 1, 2], [1, 0, 2], [2, 1, 0]])

    def test_evaluate_single_label(self):
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            self.y_labels,
            [1.0]
        )[0]
        self.assertAlmostEqual(result, 1.0/3)

        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            self.y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0)

        y_labels = np.array([[3], [3], [3]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [1.0]
        )[0]
        self.assertAlmostEqual(result, 0.0)

        y_labels = np.array([[3], [3], [3]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 0.0)

        y_labels = np.array([[0], [1], [1]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 2.0/3)

        y_labels = np.array([[0], [0], [0]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0/3)

        y_labels = np.array([[1], [0], [0]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 0.0)

        y_labels = np.array([[2], [2], [2]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0/3)

    def test_evaluate_multi_label(self):
        x_labels = np.array([[1, 0, 4], [0, 1, 4], [2, 4]])
        y_labels = np.array([[0], [1], [2]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0)

        y = np.array([self.y[0]])
        y_labels = np.array([[1, 0, 4]])
        result = evaluate.evaluate(
            self.x,
            y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0/3)

        y = np.array([self.y[0]])
        y_labels = np.array([[1, 0]])
        result = evaluate.evaluate(
            self.x,
            y,
            self.x_labels,
            y_labels,
            [0.33]
        )[0]
        self.assertAlmostEqual(result, 1.0/2)

        x_labels = np.array([[0, 1], [1], [2]])
        y = np.array([self.y[0]])
        y_labels = np.array([[1, 0]])
        result = evaluate.evaluate(self.x, y, x_labels, y_labels, [0.33])[0]
        self.assertAlmostEqual(result, 1.0)

        x_labels = np.array([[3, 0, 4], [3, 1, 4], [2, 4, 5]])
        y_labels = np.array([[0], [1], [2]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            x_labels,
            y_labels,
            [1.0]
        )[0]
        self.assertAlmostEqual(result, 1.0/3)

        x_labels = np.array([[0, 1], [1, 3], [2, 1]])
        y_labels = np.array([[0], [1], [2]])
        result = evaluate.evaluate(
            self.x,
            self.y,
            x_labels,
            y_labels,
            [1.0]
        )[0]
        self.assertAlmostEqual(result, ((1.0/3) + 1.0 + (1.0/3))/3)


if __name__ == '__main__':
    unittest.main()
