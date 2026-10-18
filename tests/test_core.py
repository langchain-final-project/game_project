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
    def test_can_move_is_read_only_and_matches_step_for_terrain(self):
        for stage in (1, 5):
            game = GameCore(load_level(stage))
            state = game._state
            positions = [[1, 1], [0, 1]]  # floor/wall and boundary checks
            if state['shortcuts']:
                gate = state['shortcuts'][0]['gate']
                positions.extend([[gate[0] - 1, gate[1]],
                                  [gate[0] + 1, gate[1]]])
            for opened in (False, True):
                for shortcut in state['shortcuts']:
                    shortcut['opened'] = opened
                for position in positions:
                    state['position'] = list(position)
                    for action in range(4):
                        with self.subTest(stage=stage, opened=opened,
                                          position=position, action=action):
                            before = game.snapshot()
                            actions = game.actions
                            rng = game._rng.getstate()
                            possible = game.can_move(action)
                            self.assertEqual(game.snapshot(), before)
                            self.assertEqual(game.actions, actions)
                            self.assertEqual(game._rng.getstate(), rng)
                            result = game.step(action)
                            self.assertEqual(possible, any(
                                event['type'] == 'move' for event in result.events))
                            game._state = before
                            state = game._state

    def test_can_move_rejects_non_movement_actions_without_mutation(self):
        game = GameCore(load_level(1))
        before = game.snapshot()
        for action in (-1, 4, 5, True, 1.5, 'RIGHT', None):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    game.can_move(action)
        self.assertEqual(game.snapshot(), before)
        self.assertEqual(game.actions, [])

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

    def test_warmth_is_collected_on_entry_and_cannot_repeat(self):
        game = GameCore(load_level(1))

        # 오른쪽 온기 칸에 들어가면 E 없이 바로 획득합니다.
        first = game.step(Action.RIGHT)

        self.assertEqual(first.snapshot['position'], [2, 1])
        self.assertEqual(first.snapshot['carried'], 1)
        self.assertEqual(first.snapshot['tick'], 1)
        self.assertEqual(
            sum(e['type'] == 'collect' for e in first.events),
            1,
        )

        # 같은 칸에서 E를 눌러도 다시 획득하지 않습니다.
        interaction = game.step(Action.INTERACT)

        self.assertEqual(interaction.snapshot['carried'], 1)
        self.assertFalse(
            any(e['type'] == 'collect' for e in interaction.events)
        )

        # 나갔다가 다시 들어와도 중복 획득하지 않습니다.
        game.step(Action.LEFT)
        revisit = game.step(Action.RIGHT)

        self.assertEqual(revisit.snapshot['carried'], 1)
        self.assertFalse(
            any(e['type'] == 'collect' for e in revisit.events)
        )

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


class MagnetTests(unittest.TestCase):
    def config(self):
        config = load_level(1)
        config['magnet'] = {'position': [3, 3], 'duration_ticks': 13}
        return config

    def test_pickup_collects_both_values_and_diagonals_without_extra_action(self):
        game = GameCore(self.config())
        game._state['position'] = [3, 2]
        result = game.step(Action.DOWN)
        self.assertTrue(result.snapshot['magnet']['collected'])
        self.assertEqual(result.snapshot['magnet']['remaining_ticks'], 12)
        self.assertEqual(result.snapshot['carried'], 4)
        self.assertEqual(result.snapshot['tick'], 1)
        self.assertEqual(result.snapshot['remaining_ticks'], 79)
        collected = [e for e in result.events if e['type'] == 'collect']
        self.assertEqual({e['item_id'] for e in collected}, {'heat_02', 'heat_08'})
        self.assertTrue(all(e['magnetic'] for e in collected))
        self.assertEqual(next(e['source_position'] for e in collected if e['value'] == 3), [2, 2])
        self.assertTrue(all(e['position'] == [3, 3] for e in collected))
        again = game.step(Action.WAIT)
        self.assertEqual(again.snapshot['carried'], 4)
        self.assertFalse(any(e['type'] == 'collect' for e in again.events))

    def test_exactly_thirteen_ticks_including_pickup_and_last_tick_collection(self):
        config = self.config()
        config['items'][6]['position'] = [3, 6]
        game = GameCore(config)
        game._state['position'] = [3, 2]
        game.step(Action.DOWN)
        for remaining in range(11, 0, -1):
            self.assertEqual(game.step(Action.WAIT).snapshot['magnet']['remaining_ticks'], remaining)
        last = game.step(Action.DOWN)
        self.assertEqual(last.snapshot['tick'], 13)
        self.assertEqual(last.snapshot['magnet']['remaining_ticks'], 0)
        self.assertTrue(any(e['type'] == 'collect' and e['item_id'] == 'heat_03'
                            and e['magnetic'] for e in last.events))
        self.assertTrue(any(e['type'] == 'magnet_end' for e in last.events))
        after = game.step(Action.DOWN)
        self.assertTrue(after.snapshot['items'][6]['active'])
        direct = game.step(Action.DOWN)
        self.assertFalse(direct.snapshot['items'][6]['active'])
        self.assertTrue(any(e['type'] == 'collect' and not e['magnetic'] for e in direct.events))

    def test_empty_pickup_is_consumed_and_collects_after_moving(self):
        game = GameCore(load_level(7))
        game._state['position'] = [8, 3]
        first = game.step(Action.DOWN)
        self.assertTrue(first.snapshot['magnet']['collected'])
        self.assertEqual(first.snapshot['carried'], 0)
        self.assertEqual(first.snapshot['magnet']['remaining_ticks'], 12)
        moved = game.step(Action.DOWN)
        self.assertEqual(moved.snapshot['carried'], 2)
        self.assertEqual(moved.snapshot['magnet']['remaining_ticks'], 11)

    def test_walls_and_closed_doors_do_not_block_diagonal_attraction(self):
        config = load_level(1)
        config['items'][4]['position'] = [6, 1]
        config['items'][8]['position'] = [5, 2]
        config['shortcuts'] = [{'id': 'A', 'terminal': [1, 2], 'gate': [5, 1],
                                'initially_open': False}]
        config['magnet'] = {'position': [4, 1], 'duration_ticks': 13}
        game = GameCore(config)
        game._state['position'] = [3, 1]
        result = game.step(Action.RIGHT)
        self.assertEqual(result.snapshot['grid'][2][4], '#')
        self.assertFalse(result.snapshot['shortcuts'][0]['opened'])
        self.assertFalse(result.snapshot['items'][8]['active'])
        self.assertTrue(any(e['type'] == 'collect' and e['source_position'] == [5, 2]
                            for e in result.events))

    def test_reset_restores_magnet_and_snapshots_do_not_alias(self):
        game = GameCore(self.config())
        initial = game.snapshot()
        game._state['position'] = [3, 2]
        game.step(Action.DOWN)
        snapshot = game.snapshot()
        snapshot['magnet']['remaining_ticks'] = 0
        self.assertEqual(game.snapshot()['magnet']['remaining_ticks'], 12)
        self.assertEqual(game.reset(), initial)
        self.assertIsNone(GameCore(load_level(1)).snapshot()['magnet'])

    def test_blocked_actions_consume_effect_and_return_does_not_reactivate(self):
        config = load_level(1)
        config['magnet'] = {'position': [3, 4], 'duration_ticks': 13}
        game = GameCore(config)
        game._state['position'] = [3, 3]
        game.step(Action.DOWN)
        blocked = game.step(Action.RIGHT)
        self.assertTrue(any(e['type'] == 'blocked' for e in blocked.events))
        self.assertEqual(blocked.snapshot['magnet']['remaining_ticks'], 11)
        game.step(Action.UP)
        returned = game.step(Action.DOWN)
        self.assertEqual(returned.snapshot['magnet']['remaining_ticks'], 9)
        self.assertFalse(any(e['type'] == 'magnet_collect' for e in returned.events))

    def test_invalid_magnet_config_rejected(self):
        for magnet in (None, {}, {'position': [3, 3], 'duration_ticks': True},
                       {'position': [3, 3], 'duration_ticks': 12},
                       {'position': [0, 0], 'duration_ticks': 13},
                       {'position': [2, 1], 'duration_ticks': 13},
                       {'position': [3, 3], 'duration_ticks': 13, 'radius': 2}):
            with self.subTest(magnet=magnet):
                config = self.config()
                config['magnet'] = magnet
                with self.assertRaises(ValueError):
                    validate_level(config)
        config = load_level(2)
        config['magnet']['position'] = [5, 2]
        with self.assertRaisesRegex(ValueError, 'ghost route'):
            validate_level(config)


class InvincibilityTests(unittest.TestCase):
    def config(self):
        config = load_level(1)
        config['invincibility'] = {'position': [3, 3], 'duration_ticks': 10}
        config['ghosts'] = [{'id': 'A', 'route': [[3, 4], [3, 5]],
                             'direction': 1, 'initial_index': 0,
                             'initial_countdown': 100, 'ticks_per_move': 1}]
        return config

    def test_ten_ticks_including_pickup_then_damage_without_extra_protection(self):
        game = GameCore(self.config())
        game._state['position'] = [3, 2]
        first = game.step(Action.DOWN)
        self.assertEqual(first.snapshot['invincibility']['remaining_ticks'], 9)
        self.assertEqual(sum(e['type'] == 'invincibility_collect' for e in first.events), 1)
        for tick in range(2, 11):
            result = game.step(Action.DOWN if tick == 2 else Action.WAIT)
            contact = next(e for e in result.events if e['type'] == 'contact')
            self.assertTrue(contact['protected'] and contact['item_invincible'])
            self.assertEqual(result.snapshot['courage'], 3)
            self.assertEqual(result.snapshot['damage_count'], 0)
            self.assertEqual(result.snapshot['protection'], 0)
            self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], 10 - tick)
        self.assertTrue(any(e['type'] == 'invincibility_end' for e in result.events))
        after = game.step(Action.WAIT)
        self.assertEqual((after.snapshot['courage'], after.snapshot['protection']), (2, 3))
        self.assertFalse(next(e for e in after.events if e['type'] == 'contact')['item_invincible'])

    def test_pickup_tick_protects_and_chaser_contact_still_stuns(self):
        game = GameCore(load_level(5))
        game._state['position'] = [12, 1]
        game._state['quest']['collected'] = True
        game._state['chaser'].update(active=True, position=[12, 3], countdown=1)
        result = game.step(Action.DOWN)
        self.assertEqual(result.snapshot['courage'], 3)
        self.assertEqual(result.snapshot['protection'], 0)
        self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], 9)
        self.assertEqual(result.snapshot['chaser']['position'], [12, 2])
        self.assertEqual(result.snapshot['chaser']['countdown'], 4)
        for remaining in (8, 7, 6):
            result = game.step(Action.RIGHT if remaining == 8 else Action.WAIT)
            self.assertEqual(result.snapshot['chaser']['position'], [12, 2])
            self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], remaining)
        self.assertEqual(game.step(Action.WAIT).snapshot['chaser']['position'], [13, 2])

    def test_normal_protection_decays_independently_and_pickup_does_not_repeat(self):
        game = GameCore(self.config())
        initial = game.snapshot()
        game._state['position'] = [3, 2]
        game._state['protection'] = 3
        game.step(Action.DOWN)
        moved = game.step(Action.DOWN)
        self.assertEqual(moved.snapshot['protection'], 1)
        returned = game.step(Action.UP)
        self.assertEqual(returned.snapshot['invincibility']['remaining_ticks'], 7)
        self.assertFalse(any(e['type'] == 'invincibility_collect' for e in returned.events))
        snap = game.snapshot()
        snap['invincibility']['remaining_ticks'] = 99
        self.assertEqual(game.snapshot()['invincibility']['remaining_ticks'], 7)
        self.assertEqual(game.reset(), initial)
        self.assertIsNone(GameCore(load_level(1)).snapshot()['invincibility'])

    def test_blocked_move_and_interaction_consume_item_time(self):
        config = load_level(1)
        config['invincibility'] = {'position': [3, 4], 'duration_ticks': 10}
        game = GameCore(config)
        game._state['position'] = [3, 3]
        game.step(Action.DOWN)
        result = game.step(Action.RIGHT)
        self.assertTrue(any(e['type'] == 'blocked' for e in result.events))
        self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], 8)
        self.assertEqual(game.step(Action.INTERACT).snapshot['invincibility']['remaining_ticks'], 7)

    def test_invalid_item_configuration_and_overlaps_rejected(self):
        for item in (None, {}, {'position': [3, 3], 'duration_ticks': True},
                     {'position': [3, 3], 'duration_ticks': 9},
                     {'position': [0, 0], 'duration_ticks': 10},
                     {'position': [2, 1], 'duration_ticks': 10},
                     {'position': [3, 4], 'duration_ticks': 10},
                     {'position': [3, 3], 'duration_ticks': 10, 'radius': 1}):
            with self.subTest(item=item):
                config = self.config()
                config['invincibility'] = item
                with self.assertRaises(ValueError):
                    validate_level(config)
        config = self.config()
        config['magnet'] = {'position': [3, 3], 'duration_ticks': 13}
        with self.assertRaises(ValueError):
            validate_level(config)

    def test_both_item_effects_can_run_independently(self):
        config = load_level(1)
        config['magnet'] = {'position': [3, 3], 'duration_ticks': 13}
        config['invincibility'] = {'position': [3, 4], 'duration_ticks': 10}
        game = GameCore(config)
        game._state['position'] = [3, 2]
        game.step(Action.DOWN)
        result = game.step(Action.DOWN)
        self.assertEqual(result.snapshot['magnet']['remaining_ticks'], 11)
        self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], 9)
        for _ in range(9):
            result = game.step(Action.WAIT)
        self.assertEqual(result.snapshot['magnet']['remaining_ticks'], 2)
        self.assertEqual(result.snapshot['invincibility']['remaining_ticks'], 0)
        self.assertEqual(result.snapshot['protection'], 0)


if __name__ == '__main__':
    unittest.main()
