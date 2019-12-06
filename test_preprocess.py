import unittest
import preprocess as pp


class TestPreprocess(unittest.TestCase):
    def test_preprocess(self):
        vocab = {'a': 0, 'b': 1, 'c': 2}

        x = 'a a b b'
        output = [int(i) for i in pp.preprocess(x, vocab).split()]
        assert len(output) == 2
        assert set(output) == set([0, 1])

        x = 'b b b c c c'
        output = [int(i) for i in pp.preprocess(x, vocab).split()]
        assert len(output) == 2
        assert set(output) == set([1, 2])

        x = 'a b c'
        output = [int(i) for i in pp.preprocess(x, vocab).split()]
        assert len(output) == 3
        assert set(output) == set([0, 1, 2])


if __name__ == '__main__':
    unittest.main()
