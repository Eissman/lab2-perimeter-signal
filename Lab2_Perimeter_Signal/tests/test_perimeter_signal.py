import math
import unittest

from src.perimeter_signal import (
    ExponentialMovingAverage,
    RawSample,
    RingBuffer,
    harmonic_signal,
    receive_and_process,
    validate_raw_sample,
)


class SignalLabTests(unittest.TestCase):
    def test_harmonic_signal_at_zero_is_zero(self):
        self.assertAlmostEqual(harmonic_signal(0.0), 0.0, places=12)

    def test_mismatched_harmonic_vectors_raise(self):
        with self.assertRaises(ValueError):
            harmonic_signal(1.0, (1.0, 0.5), (0.1,))

    def test_ema_first_output_equals_first_input(self):
        ema = ExponentialMovingAverage(period=9)
        self.assertAlmostEqual(ema.update(2.5), 2.5)
        self.assertAlmostEqual(ema.alpha, 0.2)

    def test_ring_buffer_keeps_only_latest_values(self):
        buffer = RingBuffer(capacity=2)
        buffer.append(RawSample(0, 0.0, 1.0))
        buffer.append(RawSample(1, 0.1, 2.0))
        buffer.append(RawSample(2, 0.2, 3.0))
        self.assertEqual(len(buffer), 2)
        self.assertEqual([s.value for s in buffer.items()], [2.0, 3.0])

    def test_invalid_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_raw_sample(RawSample(0, 0.0, math.nan))

    def test_sample_count_matches_rate_times_duration(self):
        samples, buffer, _ = receive_and_process(
            sample_rate_hz=10, duration_s=2, noise_std=0, ema_period=5, buffer_size=7
        )
        self.assertEqual(len(samples), 20)
        self.assertEqual(len(buffer), 7)
        self.assertEqual(samples[0].index, 0)
        self.assertAlmostEqual(samples[-1].time_s, 1.9)


if __name__ == "__main__":
    unittest.main()
