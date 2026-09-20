"""CPU checks for the experimental half-stage addressing and ring mapping.

These are protocol models, not proof of CUDA memory-ordering correctness.
Actual kernel equivalence and performance must still be tested on idle GPUs.
"""
import random
import unittest


def atom_coordinates(atom):
    dp_group, k_half = atom % 2, atom // 2
    return {
        (warp * 32 + dp_group * 16 + row, k_half * 64 + sub_k * 32 + column)
        for warp in range(4)
        for row in range(16)
        for sub_k in range(2)
        for column in range(32)
    }


class PartialReadyProtocolTests(unittest.TestCase):
    def test_two_dp_atoms_cover_exact_k_half(self):
        halves = []
        for half in range(2):
            a, b = atom_coordinates(half * 2), atom_coordinates(half * 2 + 1)
            self.assertFalse(a & b)
            covered = a | b
            self.assertEqual(covered, {(m, k) for m in range(128)
                                       for k in range(half * 64, (half + 1) * 64)})
            halves.append(covered)
        self.assertFalse(halves[0] & halves[1])
        self.assertEqual(len(halves[0] | halves[1]), 128 * 128)

    def test_ring_reuse_waits_for_mma_retirement(self):
        # Model producer, MMA issue and MMA retirement as independent actors.
        # Memory keys include the half, so a retired half can be reused while
        # the adjacent half remains in flight without sharing any addresses.
        for seed in range(100):
            rng = random.Random(seed)
            resident = {}
            produced = issued = retired = 0
            total = 128
            while retired < total:
                actions = []
                if produced < total and produced % 16 not in resident:
                    actions.append("produce")
                if issued < produced:
                    actions.append("issue")
                if retired < issued:
                    actions.append("retire")
                self.assertTrue(actions)
                action = rng.choice(actions)
                if action == "produce":
                    barrier_slot = produced % 16
                    self.assertEqual((barrier_slot // 2, barrier_slot % 2),
                                     ((produced // 2) % 8, produced % 2))
                    resident[barrier_slot] = produced
                    produced += 1
                elif action == "issue":
                    self.assertEqual(resident[issued % 16], issued)
                    issued += 1
                else:
                    self.assertEqual(resident.pop(retired % 16), retired)
                    retired += 1
                self.assertLessEqual(len(resident), 16)
            self.assertFalse(resident)


if __name__ == "__main__":
    unittest.main()
