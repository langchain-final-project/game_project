"""Rule-boundary tests independent of presentation and learned policies."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from kkamang.core import Action, GameCore
from kkamang.levels import load_level, load_levels, validate_level


def fixture(*, ghosts=0, budget=80):
    """Small synthetic variants retain the authored schema and 14-point item set."""
    config = load_level(1)
    config['time_budget_ticks'] = budget
    config['star_fast_ticks'] = max(0, budget - 1)
    config['quest']['position'] = [1,2]
    if ghosts:
        config['stage'] = 2 if ghosts == 1 else 7
        config['id'] = f"stage_{config['stage']:02}"
        config['ghosts'] = [
            {'id': chr(65 + n), 'route': [[2, 1], [3, 1]],
             'initial_index': 0, 'direction': 1, 'initial_countdown': 2,
             'ticks_per_move': 2}
            for n in range(ghosts)
        ]
    if ghosts == 2:
        config['shortcuts'] = [{'id': 'A', 'terminal': [4, 5],
                                'gate': [4, 6], 'initially_open': False}]
    return config


class CoreContractTests(unittest.TestCase):
    def test_all_twelve_authored_levels_validate(self):
        levels = load_levels()
        self.assertEqual([level['stage'] for level in levels], list(range(1, 13)))
        for config in levels:
            with self.subTest(stage=config['stage']):
                validate_level(config)
                state = GameCore(config).snapshot()
                self.assertEqual(state['carried'], 0)
                self.assertEqual(sum(i['value'] for i in state['items'] if i['active']), 14)
                self.assertFalse(state['terminal'])

    def test_initial_zero_points_and_blocked_return_do_not_end_episode(self):
        game=GameCore(load_level(1))
        result=game.step(Action.INTERACT)
        self.assertFalse(result.terminated)
        self.assertIsNone(result.outcome)
        self.assertEqual(result.snapshot['tick'],1)
        self.assertEqual(result.events[0]['type'],'return_blocked')

    def test_collection_requires_same_cell_interact_and_cannot_repeat(self):
        game = GameCore(load_level(1))
        moved = game.step(Action.RIGHT).snapshot
        self.assertEqual(moved['position'], [2, 1])
        self.assertEqual(moved['carried'], 0)
        first = game.step(Action.INTERACT)
        second = game.step(Action.INTERACT)
        self.assertEqual(first.snapshot['carried'], 1)
        self.assertEqual(second.snapshot['carried'], 1)
        self.assertEqual(second.snapshot['tick'], 3)
        self.assertFalse(any(e['type'] == 'collect' for e in second.events))

    def test_bank_value_is_actual_total_not_total_minus_target(self):
        game = GameCore(fixture())
        for action in (Action.DOWN, Action.INTERACT, Action.RIGHT, Action.INTERACT,
                       Action.UP, Action.INTERACT, Action.LEFT, Action.INTERACT):
            result = game.step(action)
        self.assertEqual(result.outcome['status'], 'COMPLETE')
        self.assertEqual(result.outcome['banked'], 4)
        self.assertEqual(result.outcome['carried'], 4)

    def test_heat_without_quest_cannot_bank_or_finish(self):
        game=GameCore(fixture())
        for action in (1,3,4,0,4,2,4): result=game.step(action)
        self.assertEqual(result.snapshot['carried'],4)
        self.assertFalse(result.terminated)
        self.assertIsNone(result.outcome)
        self.assertTrue(result.events[0]['missing_quest'])

    def test_blocked_move_and_empty_interact_consume_one_tick(self):
        game = GameCore(load_level(1))
        blocked = game.step(Action.LEFT)
        self.assertEqual(blocked.snapshot['position'], [1, 1])
        self.assertEqual(blocked.snapshot['remaining_ticks'], 79)
        self.assertTrue(any(e['type'] == 'blocked' for e in blocked.events))
        game.step(Action.DOWN)
        empty = game.step(Action.INTERACT)
        self.assertEqual(empty.snapshot['position'], [1, 2])
        self.assertEqual(empty.snapshot['tick'], 3)
        self.assertFalse(empty.terminated)

    def test_last_tick_return_beats_timeout(self):
        game = GameCore(fixture(budget=8))
        for action in (1, 4, 3, 4, 0, 4, 2, 4):
            result = game.step(action)
        self.assertEqual(result.snapshot['remaining_ticks'], 0)
        self.assertEqual(result.outcome['status'], 'COMPLETE')
        self.assertEqual(result.outcome['reason'], 'returned')
        self.assertEqual(result.outcome['banked'], 4)

    def test_last_tick_pickup_without_return_fails_and_banks_zero(self):
        game = GameCore(fixture(budget=3))
        for action in (Action.DOWN, Action.RIGHT, Action.INTERACT):
            result = game.step(action)
        self.assertEqual(result.snapshot['carried'], 3)
        self.assertEqual(result.outcome['status'], 'FAILED')
        self.assertEqual(result.outcome['reason'], 'timeout')
        self.assertEqual(result.outcome['banked'], 0)

    def test_contact_preserves_carried_value(self):
        config = fixture(ghosts=1)
        # First touch happens on the value-1 item's cell. Pickup during protection
        # is legal; no score penalty is added by either contact.
        game = GameCore(config)
        game.step(Action.RIGHT)
        result = game.step(Action.INTERACT)
        self.assertEqual(result.snapshot['courage'], 2)
        self.assertEqual(result.snapshot['carried'], 1)
        for _ in range(3):
            result = game.step(Action.WAIT)
        self.assertEqual(result.snapshot['courage'], 1)
        self.assertEqual(result.snapshot['carried'], 1)

    def test_exactly_three_future_ticks_protected(self):
        game = GameCore(fixture(ghosts=1))
        result = game.step(Action.RIGHT)
        self.assertEqual((result.snapshot['courage'], result.snapshot['protection']), (2, 3))
        states = [game.step(Action.WAIT).snapshot for _ in range(4)]
        self.assertEqual([(s['courage'], s['protection']) for s in states],
                         [(2, 2), (2, 1), (2, 0), (1, 3)])

    def test_two_ghosts_cause_only_one_damage_per_tick(self):
        result = GameCore(fixture(ghosts=2)).step(Action.RIGHT)
        self.assertEqual(result.snapshot['courage'], 2)
        self.assertEqual(sum(e['type'] == 'damage' for e in result.events), 1)

    def test_edge_swap_counts_as_contact(self):
        config = fixture(ghosts=1)
        config['ghosts'][0]['route'] = [[2, 2], [1, 2]]
        game = GameCore(config)
        first = game.step(Action.DOWN)
        self.assertEqual(first.snapshot['courage'], 3)
        second = game.step(Action.RIGHT)
        self.assertEqual(second.snapshot['position'], [2, 2])
        self.assertEqual(second.snapshot['ghosts'][0]['position'], [1, 2])
        self.assertEqual(second.snapshot['courage'], 2)

    def test_entering_a_departed_cell_without_swap_is_not_contact(self):
        config = fixture(ghosts=1)
        config['ghosts'][0]['initial_countdown'] = 1
        result = GameCore(config).step(Action.RIGHT)
        self.assertEqual(result.snapshot['position'], [2, 1])
        self.assertEqual(result.snapshot['ghosts'][0]['position'], [3, 1])
        self.assertEqual(result.snapshot['courage'], 3)
        self.assertFalse(any(e['type'] == 'contact' for e in result.events))

    def test_zero_courage_precedes_simultaneous_timeout(self):
        game = GameCore(fixture(ghosts=1, budget=9))
        game.step(Action.RIGHT)
        for _ in range(8):
            result = game.step(Action.WAIT)
        self.assertEqual(result.snapshot['courage'], 0)
        self.assertEqual(result.snapshot['remaining_ticks'], 0)
        self.assertEqual(result.outcome['status'], 'FAILED')
        self.assertEqual(result.outcome['reason'], 'courage')

    def test_defensive_zero_courage_precedes_return_request(self):
        # Deliberate isolated terminal-priority fixture, NOT an authored/reachable
        # map state: validated patrols never visit the exit. It exercises the
        # documented defensive branch without weakening the level validator.
        game = GameCore(load_level(1))
        game._state['courage'] = 0
        game._state['carried'] = 4
        game._state['remaining_ticks'] = 1
        result = game.step(Action.INTERACT)
        self.assertEqual(result.outcome['status'], 'FAILED')
        self.assertEqual(result.outcome['reason'], 'courage')
        self.assertEqual(result.outcome['banked'], 0)
        self.assertEqual(sum(e['type'] == 'end' for e in result.events), 1)
        self.assertFalse(any(e['type'] == 'return' for e in result.events))

    def test_terminal_step_is_rejected_without_mutation(self):
        game = GameCore(load_level(1))
        game.abort()
        before = game.snapshot()
        with self.assertRaises((RuntimeError, ValueError)):
            game.step(Action.WAIT)
        self.assertEqual(game.snapshot(), before)

    def test_invalid_actions_are_rejected_without_consuming_time(self):
        game = GameCore(load_level(1))
        for value in (-1, 6, 1.5, 'RIGHT', None, True):
            with self.subTest(value=value):
                before = game.snapshot()
                with self.assertRaises((TypeError, ValueError)):
                    game.step(value)
                self.assertEqual(game.snapshot(), before)

    def test_snapshot_result_and_input_config_do_not_alias_state(self):
        config = load_level(2)
        game = GameCore(config)
        config['items'][0]['value'] = 999
        state = game.snapshot()
        state['position'][0] = 999
        state['items'][0]['value'] = 888
        state['ghosts'][0]['route'][0][0] = 777
        fresh = game.snapshot()
        self.assertEqual(fresh['position'], [1, 1])
        self.assertEqual(fresh['items'][0]['value'], 1)
        self.assertNotEqual(fresh['ghosts'][0]['route'][0][0], 777)
        result = game.step(Action.RIGHT)
        expected = deepcopy(result.snapshot)
        result.snapshot['carried'] = 555
        self.assertEqual(game.snapshot(), expected)

    def test_reset_restores_items_ghosts_time_health_and_terminal_state(self):
        game = GameCore(fixture(ghosts=1))
        initial = game.snapshot()
        game.step(Action.RIGHT)
        game.step(Action.INTERACT)
        game.abort()
        self.assertEqual(game.reset(seed=0), initial)

    def test_shortcut_opens_once_on_same_cell_and_reset_closes_it(self):
        config=load_level(5)
        config['shortcuts'][0]['terminal']=[5,2]
        game=GameCore(config)
        initial = game.snapshot()
        # Terminal A=(5,2); approach through top left room.
        for action in (3, 3, 3, 3, 1):
            game.step(action)
        self.assertEqual(game.snapshot()['position'], [5, 2])
        opened = game.step(Action.INTERACT)
        self.assertTrue(opened.snapshot['shortcuts'][0]['opened'])
        duplicate = game.step(Action.INTERACT)
        self.assertFalse(any(e['type'] == 'open' for e in duplicate.events))
        self.assertEqual(game.reset(seed=0), initial)

    def test_closed_gate_blocks_and_open_gate_can_be_crossed(self):
        config=load_level(5)
        config['shortcuts'][0]['terminal']=[5,2]
        game=GameCore(config)
        for action in (3, 3, 3, 3, 1, 3):
            game.step(action)
        result = game.step(Action.RIGHT)
        self.assertEqual(result.snapshot['position'], [6, 2])
        game.step(Action.LEFT)
        game.step(Action.INTERACT)
        game.step(Action.RIGHT)
        self.assertEqual(game.step(Action.RIGHT).snapshot['position'], [7, 2])

    def test_relaxed_budget_ceil_does_not_change_ghost_clock(self):
        config = load_level(5)
        basic, relaxed = GameCore(config), GameCore(config, mode='relaxed')
        self.assertEqual(relaxed.snapshot()['initial_ticks'], math.ceil(config['time_budget_ticks'] * 1.5))
        for _ in range(5):
            a, b = basic.step(Action.WAIT).snapshot, relaxed.step(Action.WAIT).snapshot
            self.assertEqual(a['ghosts'], b['ghosts'])
            self.assertEqual(a['courage'], b['courage'])

    def test_explicit_phase_override_and_seeded_randomization_repeat(self):
        config = load_level(7)
        game = GameCore(config)
        ghost = config['ghosts'][0]
        state = game.reset(phase_overrides={ghost['id']: {'index': 1, 'countdown': 1}})
        self.assertEqual(state['ghosts'][0]['route_index'], 1)
        self.assertEqual(state['ghosts'][0]['countdown'], 1)
        a = GameCore(config, seed=137, randomize_phase=True)
        b = GameCore(config, seed=137, randomize_phase=True)
        self.assertEqual(a.snapshot(), b.snapshot())
        for action in (5, 3, 1, 5, 4):
            self.assertEqual(a.step(action).snapshot, b.step(action).snapshot)

    def test_replay_record_recreates_randomized_episode(self):
        config = load_level(7)
        game = GameCore(config, mode='relaxed', seed=876, randomize_phase=True)
        for action in (3, 3, 1, 4, 5, 0, 2, 2, 4):
            game.step(action)
        record = game.replay_record()
        rebuilt = GameCore(config, mode=record['mode'], seed=record['seed'])
        rebuilt.reset(phase_overrides=record['initial_phase'])
        for action in record['actions']:
            rebuilt.step(action)
        self.assertEqual(game.snapshot(), rebuilt.snapshot())
        self.assertEqual(game.config_hash, rebuilt.config_hash)
        record['actions'].append(999)
        self.assertNotIn(999, game.actions)

    def test_invalid_reset_phase_does_not_destroy_current_episode(self):
        game = GameCore(load_level(2))
        game.step(Action.RIGHT)
        before = game.snapshot()
        with self.assertRaises(ValueError):
            game.reset(phase_overrides={'A': {'index': 999, 'countdown': 1}})
        self.assertEqual(game.snapshot(), before)

    def test_abort_banks_zero_and_is_not_a_sixth_gameplay_action(self):
        game = GameCore(load_level(1))
        game.step(Action.RIGHT)
        game.step(Action.INTERACT)
        before = game.snapshot()['tick']
        result = game.abort()
        self.assertEqual(result.outcome['status'], 'ABORTED')
        self.assertEqual(result.outcome['banked'], 0)
        self.assertEqual(result.snapshot['tick'], before)
        self.assertEqual(len(Action), 6)

    def test_level_validator_rejects_ambiguous_overlap(self):
        config = load_level(1)
        config['items'][0]['position'] = config['exit'][:]
        with self.assertRaises(ValueError):
            validate_level(config)


if __name__ == '__main__':
    unittest.main()
