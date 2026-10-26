"""Check schedule validity and the actual sampler's intervals and call budget."""

import unittest

import torch

from model import ModelFewNFE, ModelOneNFE
from src.sampling_schedule import power_schedule


class PowerScheduleTests(unittest.TestCase):
    def test_default_preserves_original_uniform_times(self):
        self.assertEqual(power_schedule(4), (1.0, 0.75, 0.5, 0.25, 0.0))
        self.assertEqual(power_schedule(1), (1.0, 0.0))

    def test_nonuniform_times_and_endpoints(self):
        self.assertEqual(power_schedule(4, 2), (1.0, 0.5625, 0.25, 0.0625, 0.0))
        for rho in (0.7, 1.0, 1.4, 2.0):
            with self.subTest(rho=rho):
                times = power_schedule(4, rho)
                self.assertEqual(len(times), 5)
                self.assertEqual((times[0], times[-1]), (1.0, 0.0))
                self.assertTrue(all(a > b for a, b in zip(times, times[1:])))

    def test_invalid_steps(self):
        for steps in (0, -1, 1.5, True, "4", None):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                power_schedule(steps)

    def test_invalid_rho(self):
        for rho in (0, -1, float("nan"), float("inf"), -float("inf"),
                    True, "1", None, 1j, 10 ** 1000):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                power_schedule(4, rho)

    def test_rejects_intervals_collapsed_by_rounding(self):
        for rho in (1e-308, 1e308):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                power_schedule(4, rho)


class RecordingField(torch.nn.Module):
    num_classes = 151

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, timestep, *, r, category):
        self.calls.append((timestep.clone(), r.clone(), category.clone()))
        return (0.17 * x + 0.2 * timestep[:, None, None, None]
                + 0.1 * r[:, None, None, None]
                + 0.01 * category[:, None, None, None])


def make_sampler(model_class, rho):
    # Exercise the production sampling method without constructing its 56M U-Net.
    model = model_class.__new__(model_class)
    torch.nn.Module.__init__(model)
    model.backbone = RecordingField()
    model.sampling_rho = rho
    return model


class SamplerScheduleTests(unittest.TestCase):
    shape = (2, 3, 64, 64)

    def setUp(self):
        self.category = torch.tensor([0, 21])

    def test_four_calls_receive_nonuniform_intervals(self):
        model = make_sampler(ModelFewNFE, 2.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model.sample(self.shape, device="cpu", category=self.category)
        intervals = ((1.0, 0.5625), (0.5625, 0.25),
                     (0.25, 0.0625), (0.0625, 0.0))
        self.assertEqual(len(model.backbone.calls), 4)
        for (t, r, category), (start, end) in zip(model.backbone.calls, intervals):
            torch.testing.assert_close(t, torch.full((2,), start), rtol=0, atol=0)
            torch.testing.assert_close(r, torch.full((2,), end), rtol=0, atol=0)
            self.assertTrue(torch.equal(category, self.category))

    def test_training_wrapper_without_constructor_uses_uniform_default(self):
        model = make_sampler(ModelFewNFE, 2.0)
        del model.sampling_rho  # diagnostic_fid.py only installs a shared backbone.
        model.sample(self.shape, device="cpu", category=self.category)
        self.assertEqual(len(model.backbone.calls), 4)
        for (t, r, _), start, end in zip(
                model.backbone.calls, (1, .75, .5, .25), (.75, .5, .25, 0)):
            self.assertTrue(torch.equal(t, torch.full((2,), start, dtype=torch.float32)))
            self.assertTrue(torch.equal(r, torch.full((2,), end, dtype=torch.float32)))

    def test_uniform_schedule_reproduces_previous_sampler_output(self):
        model = make_sampler(ModelFewNFE, 1.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            actual = model.sample(self.shape, device="cpu", category=self.category)
            torch.manual_seed(1234)
            expected = torch.randn(self.shape)
            # Original four-step implementation, independent of power_schedule.
            for start, end in ((1.0, 0.75), (0.75, 0.5), (0.5, 0.25), (0.25, 0.0)):
                t = expected.new_full((2,), start)
                r = expected.new_full((2,), end)
                field = (0.17 * expected + 0.2 * t[:, None, None, None]
                         + 0.1 * r[:, None, None, None]
                         + 0.01 * self.category[:, None, None, None])
                expected = expected - (start - end) * field
        self.assertEqual(len(model.backbone.calls), 4)
        self.assertTrue(torch.equal(actual, expected.clamp(-1, 1)))

    def test_one_nfe_output_and_single_call_unchanged_by_rho(self):
        outputs = []
        for rho in (0.7, 1.0, 1.4, 2.0):
            model = make_sampler(ModelOneNFE, rho)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
                outputs.append(model.sample(self.shape, device="cpu", category=self.category))
            self.assertEqual(len(model.backbone.calls), 1)
            t, r, category = model.backbone.calls[0]
            self.assertTrue(torch.equal(t, torch.ones(2)))
            self.assertTrue(torch.equal(r, torch.zeros(2)))
            self.assertTrue(torch.equal(category, self.category))
        self.assertTrue(all(torch.equal(outputs[0], x) for x in outputs[1:]))


if __name__ == "__main__":
    unittest.main()
