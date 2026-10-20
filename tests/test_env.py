"""강화학습 환경의 관측·종료 조건·보상 계약을 검증한다."""
from copy import deepcopy
import subprocess
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from kkamang.env import MansionEnv, encode_observation, OBSERVATION_VERSION, OBSERVATION_SIZE
from kkamang.levels import load_level
from tools.evaluate import TimedGreedyPolicy


class EnvironmentTests(unittest.TestCase):
    def test_greedy_policy_requires_core_instead_of_using_a_legacy_fallback(self):
        with self.assertRaises(TypeError):
            TimedGreedyPolicy()
        with self.assertRaises(ValueError):
            TimedGreedyPolicy(core=None)

    def test_greedy_handles_a_chaser_stage_without_items_reproducibly(self):
        from tools.evaluate import project_path
        first, second = MansionEnv(12), MansionEnv(12)
        _, a = first.reset(seed=20261018)
        _, b = second.reset(seed=20261018)
        policies = [TimedGreedyPolicy(core=env.core) for env in (first, second)]
        for _ in range(5):
            initial = first.core.snapshot()
            actions = [policies[0].action(a['snapshot']), policies[1].action(b['snapshot'])]
            self.assertEqual(actions[0], actions[1])
            self.assertEqual(first.core.snapshot(), initial)
            self.assertIsNotNone(project_path(first.core, [actions[0]], allow_item_invincibility=True))
            _, _, _, _, a = first.step(actions[0])
            _, _, _, _, b = second.step(actions[1])
            self.assertEqual(a['snapshot'], b['snapshot'])

    def test_evaluation_pairs_phases_and_reports_score_items_and_searches(self):
        from tools.evaluate import evaluate, POLICY_VERSION
        rows, report = evaluate(1, [2], 20261018, 'basic', True)
        self.assertFalse(report['errors'])
        self.assertEqual(report['policy_version'], POLICY_VERSION)
        random, greedy = rows
        self.assertEqual((random['seed'], random['initial_phase'], random['config_hash']),
                         (greedy['seed'], greedy['initial_phase'], greedy['config_hash']))
        self.assertEqual(greedy['status'], 'COMPLETE')
        self.assertEqual(greedy['score_total'],
                         greedy['score_warmth'] + greedy['score_speed'] + greedy['score_courage'])
        self.assertEqual(greedy['stars'], 3)
        self.assertTrue(greedy['magnet_collected'])
        self.assertGreater(greedy['magnetic_heat'], 0)
        self.assertGreater(greedy['search_found'], 0)
        self.assertEqual(random['search_found'], 0)
        self.assertEqual(greedy['damage_count'], 0)

    def test_invincibility_observation_pickup_expiry_and_reset(self):
        env = MansionEnv(3)
        obs, _ = env.reset(seed=0)
        np.testing.assert_allclose(obs[615:620], [1, 0, 7 / 15, 5 / 8, 0])
        env.core._state['position'] = [7, 6]
        obs, reward, _, _, info = env.step(0)
        np.testing.assert_allclose(obs[615:620], [1, 1, 7 / 15, 5 / 8, .9])
        self.assertEqual(reward, 0)
        for _ in range(9):
            obs, _, _, _, _ = env.step(5)
        np.testing.assert_allclose(obs[615:620], [1, 1, 7 / 15, 5 / 8, 0])
        self.assertEqual(info['snapshot']['protection'], 0)
        self.assertEqual(env.reset(seed=0)[0][616], 0)
        self.assertFalse(MansionEnv(1).reset(seed=0)[0][615:620].any())
        with self.assertRaises(ValueError):
            MansionEnv(3, expected_observation_version='vector_v4_magnet')

    def test_invincibility_observation_rejects_invalid_state(self):
        for field, value in (('remaining_ticks', 11), ('remaining_ticks', True),
                             ('duration_ticks', 13), ('collected', 1)):
            state = MansionEnv(3).core.snapshot()
            state['invincibility'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                encode_observation(state)
        state = MansionEnv(3).core.snapshot()
        state['invincibility']['remaining_ticks'] = 1
        with self.assertRaises(ValueError):
            encode_observation(state)

    def test_gym_checker_and_bounds_all_stages_modes(self):
        for stage in range(1, 13):
            for mode in ('basic', 'relaxed'):
                with self.subTest(stage=stage, mode=mode):
                    env = MansionEnv(stage, mode=mode, randomize_phase=True)
                    check_env(env, skip_render_check=True)
                    obs, _ = env.reset(seed=123)
                    self.assertEqual(obs.shape, (OBSERVATION_SIZE,))
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
        self.assertEqual(obs[481], 1)
        self.assertEqual(obs[534], 1)
        self.assertEqual(obs[587], 1)
        self.assertEqual(obs[593], 1)
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
        config['ghosts'] = [{'id': 'A', 'route': [[5, 1], [6, 1]],
                             'initial_index': 0, 'direction': 1,
                             'initial_countdown': 2, 'ticks_per_move': 2}]
        env = MansionEnv(config=config, safety_coefficient=0.1)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(3)
        self.assertEqual(info['snapshot']['courage'], 2)
        self.assertAlmostEqual(reward, -0.1)

        _, reward, _, _, info = env.step(3)
        self.assertEqual(info['snapshot']['courage'], 2)
        self.assertEqual(reward, 0)

    def test_banked_reward_not_collection_reward_and_no_progression(self):
        env = MansionEnv(1)
        _, info = env.reset(seed=0)
        policy = TimedGreedyPolicy(core=env.core)
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

    def test_magnet_observation_distinguishes_absent_pickup_active_and_expired(self):
        absent, _ = MansionEnv(1).reset(seed=0)
        np.testing.assert_array_equal(absent[610:615], np.zeros(5))
        env = MansionEnv(2)
        before, _ = env.reset(seed=0)
        magnet_x, magnet_y = env.core.config['magnet']['position']
        np.testing.assert_allclose(before[610:615], [1, 0, magnet_x / 15, magnet_y / 8, 0])
        for action in (1, 1, 1, 3, 3):
            active, reward, _, _, info = env.step(action)
        self.assertEqual(reward, 0)
        self.assertEqual(info['snapshot']['carried'], 2)
        self.assertEqual(active[611], 1)
        self.assertAlmostEqual(active[614], 12 / 13)
        for _ in range(12):
            expired, _, _, _, _ = env.step(5)
        self.assertEqual(expired[611], 1)
        self.assertEqual(expired[614], 0)
        reset, _ = env.reset(seed=0)
        np.testing.assert_array_equal(reset, before)
        with self.assertRaises(ValueError):
            MansionEnv(expected_observation_version='vector_v3_chaser')

    def test_invalid_magnet_observation_is_rejected(self):
        env = MansionEnv(2)
        for field, value in (('collected', 1), ('remaining_ticks', -1),
                             ('remaining_ticks', 14), ('remaining_ticks', True),
                             ('duration_ticks', 12), ('position', [16, 0]),
                             ('remaining_ticks', 1)):
            with self.subTest(field=field, value=value):
                snapshot = env.core.snapshot()
                snapshot['magnet'][field] = value
                with self.assertRaises(ValueError):
                    encode_observation(snapshot)

    def test_greedy_magnet_simulation_does_not_advance_the_live_game(self):
        env = MansionEnv(2)
        _, info = env.reset(seed=0)
        policy = TimedGreedyPolicy(core=env.core)
        while True:
            before = env.core.snapshot()
            action = policy.action(info['snapshot'])
            self.assertEqual(env.core.snapshot(), before)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        self.assertEqual(info['outcome']['status'], 'COMPLETE')
        self.assertEqual(info['outcome']['banked'], 14)
        self.assertEqual(info['outcome']['damage_count'], 0)
        self.assertTrue(info['snapshot']['magnet']['collected'])


if __name__ == '__main__':
    unittest.main()
