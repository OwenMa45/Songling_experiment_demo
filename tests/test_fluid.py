import unittest
from piper_titration.fluid import Fluid


PARAMETERS = dict(initial_volume_ml=2, drop_volume_ml=.05, compression_threshold=.12, displacement_ml=.12, relaxation_seconds=.10, release_seconds=.18)


class FluidTests(unittest.TestCase):
    def simulate(self, compression, tip=(0, 0, .2), volume=2):
        f = Fluid(dict(PARAMETERS, initial_volume_ml=volume))
        for i in range(1000):
            f.step(.002, compression(i*.002), tip, (0, 0, .1), .014, i*.002)
        return f

    def test_light_pressure_does_not_dispense(self):
        self.assertEqual(self.simulate(lambda t: .1).emitted, 0)

    def test_static_squeeze_has_finite_displacement(self):
        f = self.simulate(lambda t: .625)
        self.assertEqual(f.received, 1)
        self.assertLessEqual(f.emitted*.05, .12)

    def test_missed_tube_is_spill(self):
        f = self.simulate(lambda t: .625, tip=(.1, 0, .2))
        self.assertEqual(f.received, 0)
        self.assertEqual(f.spilled, 1)

    def test_exhaustion_conserves_volume(self):
        f = self.simulate(lambda t: 1 if int(t*10)%2 else 0, volume=.12)
        self.assertEqual(f.emitted, 2)
        self.assertGreaterEqual(f.remaining, 0)
        self.assertAlmostEqual(f.remaining+f.accumulator+f.emitted*.05, .12)

    def test_release_does_not_refill(self):
        f = self.simulate(lambda t: .625 if t < .25 else 0)
        self.assertLess(f.remaining, 2)
        self.assertLess(f.pressure_volume, 1e-8)
        self.assertEqual(f.compression, 0)

    def test_in_flight_drop_arrives_after_release(self):
        f = Fluid(PARAMETERS)
        for i in range(140):
            f.step(.002, .625, (0, 0, 1.1), (0, 0, .1), .014, i*.002)
        self.assertEqual(f.emitted, 1)
        self.assertEqual(f.received, 0)
        for i in range(140, 600):
            f.step(.002, 0, (0, 0, 1.1), (0, 0, .1), .014, i*.002)
        self.assertEqual(f.received, 1)


if __name__ == "__main__":
    unittest.main()
