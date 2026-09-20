"""Model the four-warp rescale threshold protocol (no CUDA dependency)."""
import itertools
import unittest


def live_shared_threshold(previous, warp_maxima):
    maximum = previous
    for value in warp_maxima:
        if value - maximum > 8:
            maximum = max(maximum, value)
    return maximum


def snapshot_threshold(previous, warp_maxima):
    maximum = previous
    for value in warp_maxima:
        if value - previous > 8:
            maximum = max(maximum, value)
    return maximum


class RescaleSnapshotTests(unittest.TestCase):
    def test_live_read_is_order_dependent(self):
        self.assertEqual(
            {live_shared_threshold(float("-inf"), p)
             for p in itertools.permutations([1, 2, 3, 4])},
            {1, 2, 3, 4},
        )
        self.assertEqual(
            {live_shared_threshold(0, p)
             for p in itertools.permutations([1, 2, 9, 10])},
            {9, 10},
        )

    def test_snapshot_matches_collective_threshold(self):
        for previous, values in (
            (float("-inf"), [1, 2, 3, 4]),
            (0, [1, 2, 9, 10]),
            (0, [1, 2, 7, 8]),
            (8, [9, 10, 17, 20]),
        ):
            expected = max(values) if max(values) - previous > 8 else previous
            for order in itertools.permutations(values):
                self.assertEqual(snapshot_threshold(previous, order), expected)


if __name__ == "__main__":
    unittest.main()
