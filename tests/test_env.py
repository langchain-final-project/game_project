"""Adapter boundary tests: public observations, episode endings and reward safety."""
from copy import deepcopy
import subprocess
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from kkamang.env import MansionEnv, encode_observation, OBSERVATION_VERSION
from kkamang.levels import load_level
from tools.evaluate import TimedGreedyPolicy


class EnvironmentTests(unittest.TestCase):
    def test_gym_checker_and_bounds_all_stages_modes(self):
        for stage in range(1, 13):
            for mode in ('basic', 'relaxed'):
                with self.subTest(stage=stage, mode=mode):
                    env = MansionEnv(stage, mode=mode, randomize_phase=True)
                    check_env(env, skip_render_check=True)
                    obs, _ = env.reset(seed=123)
                    self.assertEqual(obs.shape, (610,))
                    self.assertEqual(obs.dtype, np.float32)
                    self.assertTrue(env.observation_space.contains(obs))
                    env.action_space.seed(444)
                    done = False
                    while not done:
                        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertTrue(np.isfinite(reward))
                        done = terminated or truncated

    def test_layout_slots_and_independent_arrays(self):
        env = MansionEnv(12)
        obs, info = env.reset(seed=5)
        snapshot = info['snapshot']
        np.testing.assert_array_equal(obs[:144].reshape(9, 16),
                                      [[c == '#' for c in row] for row in snapshot['grid']])
        self.assertEqual(obs[288 + snapshot['exit'][1]*16 + snapshot['exit'][0]], 1)
        self.assertEqual(obs[481], 1)  # Ghost A presence.
        self.assertEqual(obs[534], 1)  # Ghost B presence.
        self.assertEqual(obs[587], 1)  # Shortcut A presence.
        self.assertEqual(obs[593], 1)  # Shortcut B presence.
        original = obs.copy()
        changed = deepcopy(snapshot)
        changed['items'][0]['active'] = False
        second = encode_observation(changed)
        np.testing.assert_array_equal(second[441:445], np.zeros(4))
        np.testing.assert_array_equal(second[445:481], obs[445:481])
        obs[:] = 0
        snapshot['courage'] = 0
        fresh = encode_observation(env.core.snapshot())
        np.testing.assert_array_equal(fresh, original)

    def test_same_seed_and_actions_reproduce_random_phases(self):
        a = MansionEnv(12, randomize_phase=True)
        b = MansionEnv(12, randomize_phase=True)
        oa, ia = a.reset(seed=99)
        ob, ib = b.reset(seed=99)
        np.testing.assert_array_equal(oa, ob)
        self.assertEqual(ia, ib)
        for action in [3, 3, 1, 4, 5, 1, 3, 5]:
            ra, rb = a.step(action), b.step(action)
            np.testing.assert_array_equal(ra[0], rb[0])
            self.assertEqual(ra[1:], rb[1:])
        reset_obs, _ = a.reset(seed=99)
        np.testing.assert_array_equal(reset_obs, ob)

    def test_game_timeout_is_terminal_external_cutoff_is_truncation(self):
        config = load_level(1)
        config['time_budget_ticks'] = 1
        config['star_fast_ticks'] = 0
        timeout = MansionEnv(config=config)
        timeout.reset(seed=0)
        _, reward, terminated, truncated, info = timeout.step(5)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['outcome']['reason'], 'timeout')
        self.assertEqual(info['outcome']['banked'], 0)
        self.assertEqual(reward, 0)
        cutoff = MansionEnv(1, max_episode_steps=1)
        cutoff.reset(seed=0)
        _, reward, terminated, truncated, info = cutoff.step(5)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIsNone(info['outcome'])
        self.assertFalse(info['snapshot']['terminal'])
        self.assertEqual(info['truncation_reason'], 'external_step_limit')
        self.assertEqual(reward, 0)
        with self.assertRaises(RuntimeError):
            cutoff.step(5)

    def test_safety_penalty_only_on_actual_damage(self):
        config = load_level(1)
        config['ghosts'] = [{'id': 'A', 'route': [[2, 1], [3, 1]],
                             'initial_index': 0, 'direction': 1,
                             'initial_countdown': 2, 'ticks_per_move': 2}]
        env = MansionEnv(config=config, safety_coefficient=0.1)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(3)
        self.assertEqual(info['snapshot']['courage'], 2)
        self.assertAlmostEqual(reward, -0.1)
        # Following the ghost into its next cell makes contact while protected.
        _, reward, _, _, info = env.step(3)
        self.assertEqual(info['snapshot']['courage'], 2)
        self.assertEqual(reward, 0)

    def test_banked_reward_not_collection_reward_and_no_progression(self):
        env = MansionEnv(1)
        _, info = env.reset(seed=0)
        policy = TimedGreedyPolicy()
        done = False
        while not done:
            _, reward, terminated, truncated, info = env.step(policy.action(info['snapshot']))
            done = terminated or truncated
            if not done:
                self.assertEqual(reward, 0)
        self.assertEqual(info['outcome']['status'], 'COMPLETE')
        self.assertEqual(reward, 2 + info['outcome']['banked']/14)
        self.assertEqual(info['controller'], 'ai')
        self.assertNotIn('campaign', vars(env))

    def test_reject_bad_versions_bounds_and_actions(self):
        with self.assertRaises(ValueError):
            MansionEnv(expected_observation_version='unknown')
        env = MansionEnv(1)
        self.assertEqual(env.observation_version, OBSERVATION_VERSION)
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        for action in (-1, 6, True, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)
        invalid = env.core.snapshot()
        invalid['initial_ticks'] = 513
        with self.assertRaises(ValueError):
            encode_observation(invalid)
        invalid = env.core.snapshot()
        invalid['items'][0]['id'] = 'heat_01'
        with self.assertRaises(ValueError):
            encode_observation(invalid)

    def test_headless_import_does_not_load_pygame(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; from kkamang.env import MansionEnv; e=MansionEnv(); "
            "e.reset(seed=0); e.step(5); assert 'pygame' not in sys.modules"],
            check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
