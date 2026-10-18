"""Headless SDL integration: real input, screen flow, settlement and persistence.

These checks run the app controller and rendering code. They are not a substitute
for testing a packaged executable on Windows or for human playtesting.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame

from kkamang.app import GameApp
from kkamang.campaign import SaveStore
from kkamang.core import Action

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / 'validation' / 'reference_routes.json').read_text(encoding='utf-8'))
ROUTES = {(route['level_id'], route['target']): route
          for route in REFERENCE['routes'] if 'trace' in route}
ACTION_KEYS = {0: pygame.K_UP, 1: pygame.K_DOWN, 2: pygame.K_LEFT,
               3: pygame.K_RIGHT, 4: pygame.K_e}


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='kkamang-app-test-')
        self.save_path = Path(self.directory.name) / 'save.json'
        self.app = GameApp(save_path=self.save_path, headless=True, no_audio=True)

    def tearDown(self):
        self.app.close()
        self.directory.cleanup()

    def assert_rendered(self, scene, required=()):
        self.assertEqual(self.app.scene, scene)
        self.app.draw()
        controls = self.app.controls
        ids = [control['id'] for control in controls]
        self.assertEqual(len(ids), len(set(ids)), f'Duplicate control IDs on {scene}')
        for ident in required:
            self.assertIn(ident, ids)
        for control in controls:
            self.assertIsInstance(control['rect'], pygame.Rect)
            self.assertGreater(control['rect'].width, 0)
            self.assertGreater(control['rect'].height, 0)
            self.assertTrue(pygame.Rect(0, 0, 1280, 720).contains(control['rect']),
                            f'Control extends outside logical canvas: {scene}/{control["id"]}')

    def first_start(self):
        self.assert_rendered('title', ('new_game',))
        self.app.dispatch('new_game')
        self.assert_rendered('story', ('story_skip',))
        self.app.dispatch('story_skip')
        self.assert_rendered('home')
        self.assertTrue(self.app.campaign.data['intro_seen'])
        self.assertTrue(self.save_path.exists())

    def click_control(self, ident):
        self.app.draw()
        control = next(button for button in self.app.controls if button['id'] == ident)
        self.assertFalse(control['disabled'])
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                               pos=control['rect'].center, button=1))

    def click_result_primary(self):
        self.app.draw()
        # Result layout places its gold main CTA last. Inspect the actual
        # renderer output so a mismatched visual label/action cannot be missed.
        primary = self.app.controls[-1]
        if self.app.last_reward.get('new_photo'):
            self.assertEqual(primary['label'], '새 기억 펼치기')
            self.assertEqual(primary['id'], 'view_memory')
        self.click_control(primary['id'])

    def enter_stage(self, stage=1):
        self.app.dispatch('rooms')
        self.assert_rendered('rooms', (f'stage:{stage}',))
        self.app.dispatch(f'stage:{stage}')
        self.assert_rendered('briefing', ('start',))
        self.app.dispatch('start')
        if self.app.scene == 'tutorial':
            self.assert_rendered('tutorial', ('tutorial_done',))
            self.app.dispatch('tutorial_done')
        self.assert_rendered('game')

    # def apply_action(self, action):
    #     self.assertEqual(self.app.scene, 'game')
    #     key = ACTION_KEYS.get(int(action))
    #     if key is not None:
    #         self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
    #     self.app.advance(0.3)
    #     if key is not None:
    #         self.app.handle_event(pygame.event.Event(pygame.KEYUP, key=key, mod=0))
    def apply_action(self, action, *, short_tap=False):
        self.assertEqual(self.app.scene, 'game')

        before_tick = self.app.core.snapshot()['tick']
        tick_seconds = float(
            self.app.core.config.get('tick_seconds', 0.3)
        )
        key = ACTION_KEYS.get(int(action))

        if key is not None:
            self.app.handle_event(pygame.event.Event(
                pygame.KEYDOWN, key=key, mod=0,
            ))

            if short_tap:
                self.app.handle_event(pygame.event.Event(
                    pygame.KEYUP, key=key, mod=0,
                ))

        # 키를 누른 순간 행동했다면 추가로 시간을 보내지 않습니다.
        if self.app.core.snapshot()['tick'] == before_tick:
            self.app.advance(
                max(0., tick_seconds - self.app.accumulator)
            )

        if key is not None and not short_tap:
            self.app.handle_event(pygame.event.Event(
                pygame.KEYUP, key=key, mod=0,
            ))

        # 함수 한 번에 행동이 정확히 한 번 실행됐는지 확인합니다.
        self.assertEqual(
            self.app.core.snapshot()['tick'],
            before_tick + 1,
        )

    def complete_stage(self, stage, target=4):
        route = ROUTES[(f'stage_{stage:02}', target)]
        for action in route['actions']:
            self.apply_action(action)
        self.assertEqual(self.app.last_outcome['status'], 'COMPLETE')
        self.assertEqual(self.app.last_outcome['banked'], route['banked'])
        self.assertEqual(self.app.last_outcome['ticks_used'], route['ticks'])

    def test_first_run_story_tutorial_real_keys_and_memory_reward(self):
        self.first_start()
        self.enter_stage(1)
        self.assertTrue(self.app.campaign.data['tutorial_seen'])
        self.complete_stage(1)
        self.assert_rendered('result')
        self.assertTrue(self.app.last_reward['first_complete'])
        self.assertTrue(self.app.last_reward['new_photo'])
        self.assertEqual(self.app.campaign.data['photos'], [1])
        self.assertTrue(self.app.campaign.is_unlocked(2))
        self.click_result_primary()
        self.assert_rendered('story', ('story_skip',))
        self.app.dispatch('story_skip')
        self.assert_rendered('home')
        loaded = SaveStore(self.save_path).load()
        self.assertEqual(loaded.data['photos'], [1])
        self.assertEqual(loaded.data['completed'], [1])

    def test_replay_stage_does_not_duplicate_photo_and_can_earn_extra_badge(self):
        self.first_start()
        self.enter_stage(1)
        self.complete_stage(1, target=4)
        first_badges = set(self.app.campaign.data['badges'])
        self.app.dispatch('home')
        self.enter_stage(1)
        self.complete_stage(1, target=14)
        self.assertFalse(self.app.last_reward['first_complete'])
        self.assertFalse(self.app.last_reward['new_photo'])
        self.assertEqual(self.app.campaign.data['photos'], [1])
        self.assertEqual(self.app.campaign.data['completed'], [1])
        self.assertGreater(len(self.app.campaign.data['badges']), len(first_badges))
        self.assertEqual(self.app.campaign.optional_badges(), 2)
        # Settling the same result again remains idempotent at the app boundary.
        before = deepcopy(self.app.campaign.data)
        self.app.finish_run(type('Result', (), {'outcome': self.app.last_outcome})())
        self.assertEqual(self.app.campaign.data, before)

    def test_all_twelve_stages_complete_with_saved_photos_and_ending(self):
        self.first_start()
        for stage in range(1, 13):
            with self.subTest(stage=stage):
                self.enter_stage(stage)
                self.complete_stage(stage, target=14)
                self.assert_rendered('result')
                self.click_result_primary()
                self.assert_rendered('story')
                self.app.dispatch('story_skip')
                self.assert_rendered('ending' if stage == 12 else 'home')
        self.assertEqual(self.app.campaign.data['completed'], list(range(1, 13)))
        self.assertEqual(self.app.campaign.data['photos'], list(range(1, 13)))
        self.assertEqual(self.app.campaign.optional_badges(), 24)
        self.assertEqual(len(self.app.campaign.unlocked_cosmetics('frame')), 3)
        ending_count = len(self.app.dialogues)
        self.assertGreater(ending_count, 0)
        self.assertTrue(self.app.ending_art_only)
        self.click_control('ending_continue')
        for _ in range(ending_count):
            self.assertEqual(self.app.scene, 'ending')
            self.app.advance(30)  # Complete the current typewriter line.
            self.click_control('ending_continue')
        self.assert_rendered('home')
        self.app.dispatch('album')
        self.assert_rendered('album')
        self.app.dispatch('photo:12')
        self.app.dispatch('replay_memory')
        self.assert_rendered('story')
        self.app.dispatch('story_skip')
        self.assert_rendered('album')
        saved = SaveStore(self.save_path).load()
        self.assertEqual(saved.data['completed'], list(range(1, 13)))
        self.assertEqual(saved.optional_badges(), 24)

    def test_pause_and_focus_loss_freeze_clock_and_clear_held_key(self):
        self.first_start()
        self.enter_stage(1)
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0))
        self.app.advance(0.15)
        before = self.app.core.snapshot()
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
        self.assert_rendered('pause', ('resume', 'restart'))
        self.app.advance(60)
        self.assertEqual(self.app.core.snapshot(), before)
        self.app.dispatch('resume')
        self.app.advance(0.3)
        self.assertEqual(
            self.app.core.snapshot()['position'],
            before['position'],
        )
        self.assertEqual(
            self.app.core.snapshot()['tick'],
            before['tick'] + 1,
        )
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0))
        self.app.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        self.assert_rendered('pause')
        before = self.app.core.snapshot()
        self.app.advance(30)

        # 일시정지 중에는 모든 상태가 그대로여야 합니다.
        self.assertEqual(self.app.core.snapshot(), before)

        self.app.dispatch('resume')
        self.app.advance(0.3)

        # 재개 후 방향키가 풀려 있어 이동 없이 한 틱 대기합니다.
        self.assertEqual(
            self.app.core.snapshot()['position'],
            before['position'],
        )
        self.assertEqual(
            self.app.core.snapshot()['tick'],
            before['tick'] + 1,
        )

    def test_pause_preserves_partial_tick_and_resume_has_no_backward_interpolation(self):
        self.first_start()
        self.enter_stage(1)
        self.apply_action(Action.RIGHT)
        self.app.advance(0.2)
        before = self.app.core.snapshot()
        self.assertEqual(before['tick'], 1)
        self.app.go_back()
        self.assert_rendered('pause', ('resume',))
        self.app.advance(10)
        self.assertEqual(self.app.core.snapshot(), before)
        self.assertEqual(self.app.controls[self.app.focus]['id'], 'resume')
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
        self.assertEqual(self.app.scene, 'game')
        self.assertAlmostEqual(self.app.accumulator, 0.2)
        self.assertEqual(self.app.previous, self.app.core.snapshot())
        self.app.advance(0.1)
        self.assertEqual(self.app.core.snapshot()['tick'], 2)
        self.assertEqual(self.app.core.snapshot()['position'], before['position'])
        self.assertAlmostEqual(self.app.accumulator, 0)

    def test_short_taps_complete_route_and_default_enter_opens_new_memory(self):
        self.first_start()
        self.enter_stage(1)
        for action in ROUTES[('stage_01', 4)]['actions']:
            self.apply_action(action, short_tap=True)
        # for action in ROUTES[('stage_01', 4)]['actions']:
        #     key = ACTION_KEYS.get(action)
        #     if key is not None:
        #         self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        #         self.app.handle_event(pygame.event.Event(pygame.KEYUP, key=key, mod=0))
        #     self.app.advance(0.3)
        self.assertEqual(self.app.last_outcome['status'], 'COMPLETE')
        self.assert_rendered('result')
        self.assertEqual(self.app.controls[self.app.focus]['id'], 'view_memory')
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
        self.assert_rendered('story')
        self.assertEqual(self.app.story_kind, 'memory')

    def test_quitting_to_home_and_restart_abort_without_rewards(self):
        self.first_start()
        self.enter_stage(1)
        self.apply_action(Action.RIGHT)
        self.apply_action(Action.INTERACT)
        self.assertEqual(self.app.core.snapshot()['carried'], 1)
        self.app.go_back()
        self.app.dispatch('restart')
        self.assert_rendered('confirm', ('confirm_no', 'confirm_yes'))
        self.assertEqual(self.app.controls[0]['id'], 'confirm_no')
        self.app.dispatch('confirm_yes')
        self.assert_rendered('game')
        self.assertEqual(self.app.core.snapshot()['tick'], 0)
        self.assertEqual(self.app.core.snapshot()['carried'], 0)
        self.assertEqual(self.app.campaign.data['photos'], [])
        self.app.go_back()
        self.app.dispatch('home')
        self.app.dispatch('confirm_yes')
        self.assert_rendered('home')
        self.assertEqual(self.app.core.snapshot()['outcome']['status'], 'ABORTED')
        self.assertEqual(self.app.campaign.data['completed'], [])
        self.assertEqual(self.app.campaign.optional_badges(), 0)
        replays = list((self.save_path.parent / 'replays').glob('*.json'))
        self.assertEqual(len(replays), 2)
        for path in replays:
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['outcome']['status'], 'ABORTED')

    def test_save_failure_keeps_memory_and_retry_persists_exact_result(self):
        self.first_start()
        self.enter_stage(1)
        old_bytes = self.save_path.read_bytes()
        with patch.object(self.app.store, '_atomic_write', side_effect=OSError('simulated disk full')):
            self.complete_stage(1)
        self.assert_rendered('save_error', ('save_retry', 'save_continue'))
        self.assertEqual(self.app.campaign.data['photos'], [1])
        self.assertEqual(self.save_path.read_bytes(), old_bytes)
        before = deepcopy(self.app.campaign.data)
        self.app.dispatch('save_retry')
        self.assert_rendered('result')
        self.assertIsNone(self.app.store.error)
        self.assertEqual(self.app.campaign.data, before)
        self.assertEqual(SaveStore(self.save_path).load().data, before)

    def test_save_continue_retains_session_progress_and_retry_later(self):
        self.first_start()
        self.enter_stage(1)
        with patch.object(self.app.store, '_atomic_write', side_effect=OSError('simulated read only')):
            self.complete_stage(1)
        self.app.dispatch('save_continue')
        self.assert_rendered('result')
        self.assertIsNotNone(self.app.store.error)
        self.assertEqual(self.app.campaign.data['photos'], [1])
        self.app.dispatch('home')
        self.app.dispatch('settings')
        self.assert_rendered('settings')
        self.app.dispatch('back')
        self.assert_rendered('home')
        self.assertIsNone(self.app.store.error)
        self.assertEqual(SaveStore(self.save_path).load().data['photos'], [1])

    def test_menu_draw_is_read_only_and_controls_remain_inside_canvas(self):
        self.first_start()
        self.enter_stage(1)
        self.complete_stage(1)
        for scene in ('home', 'rooms', 'album', 'appearance', 'settings'):
            self.app.dispatch(scene)
            before = deepcopy(self.app.campaign.data)
            self.assert_rendered(scene)
            self.app.draw()
            self.assertEqual(self.app.campaign.data, before)

    def test_locked_stage_cannot_start_via_dispatch(self):
        self.first_start()
        self.app.dispatch('rooms')
        self.app.dispatch('stage:12')
        self.assertEqual(self.app.scene, 'rooms')
        self.app.dispatch('start')
        if self.app.scene == 'tutorial':
            self.app.dispatch('tutorial_done')
        self.assertNotEqual(self.app.scene, 'game')
        self.assertIsNone(self.app.core)

    def test_new_game_confirmation_cancel_preserves_progress(self):
        self.first_start()
        self.enter_stage(1)
        self.complete_stage(1)
        before = deepcopy(self.app.campaign.data)
        self.app.transition('title')
        self.app.dispatch('new_game')
        self.assert_rendered('confirm')
        self.app.dispatch('confirm_no')
        self.assert_rendered('title')
        self.assertEqual(self.app.campaign.data, before)

    def test_tutorial_from_title_or_settings_returns_without_starting_run(self):
        self.assert_rendered('title', ('tutorial',))
        self.click_control('tutorial')
        self.assert_rendered('tutorial', ('tutorial_done',))
        self.click_control('tutorial_done')
        self.assert_rendered('title')
        self.assertIsNone(self.app.core)
        self.assertFalse(self.app.campaign.data['intro_seen'])
        self.app.dispatch('settings')
        self.assert_rendered('settings', ('tutorial',))
        self.click_control('tutorial')
        self.assert_rendered('tutorial')
        self.click_control('tutorial_done')
        self.assert_rendered('settings')
        self.assertIsNone(self.app.core)

    def test_tutorial_back_returns_to_its_entry_screen(self):
        self.app.dispatch('tutorial')
        self.app.go_back()
        self.assert_rendered('title')
        self.app.dispatch('settings')
        self.app.dispatch('tutorial')
        self.app.go_back()
        self.assert_rendered('settings')
        self.assertIsNone(self.app.core)

    def test_render_cadence_does_not_change_wait_ticks(self):
        self.first_start()
        self.enter_stage(1)
        for _ in range(18):
            self.app.advance(1 / 60)
        after_60fps = self.app.core.snapshot()
        self.assertEqual(after_60fps['tick'], 1)
        self.app.start_run()
        for _ in range(9):
            self.app.advance(1 / 30)
        self.assertEqual(self.app.core.snapshot(), after_60fps)
    def test_starts_moving_on_keydown(self):
        self.first_start()
        self.enter_stage(1)

        before = self.app.core.snapshot()

        self.app.handle_event(pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_RIGHT,
            mod=0,
        ))

        after = self.app.core.snapshot()

        # advance()를 호출하기 전부터 이동 행동이 실행되어야 합니다.
        self.assertEqual(after['position'], [2, 1])
        self.assertEqual(after['tick'], before['tick'] + 1)
        self.app.draw()

    def test_interaction_does_not_interrupt_movement(self):
        self.first_start()
        self.enter_stage(1)

        self.app.handle_event(pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_RIGHT,
            mod=0,
        ))
        self.app.advance(0.1)

        with patch('kkamang.ui.field.cat') as draw_cat:
            self.app.draw()
            foot_before = draw_cat.call_args[0][1]

        before = self.app.core.snapshot()
        progress_before = self.app.accumulator

        self.app.handle_event(pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e,
            mod=0,
        ))

        after = self.app.core.snapshot()

        # 기다리지 않아도 획득되고, 이동 진행 시간은 유지됩니다.
        # 온기는 이동하면서 이미 획득했습니다.
        # E를 눌러도 온기가 추가로 수집되지는 않습니다.
        self.assertEqual(before['carried'], 1)
        self.assertEqual(after['carried'], before['carried'])
        self.assertEqual(after['tick'], before['tick'] + 1)
        self.assertAlmostEqual(
            self.app.accumulator,
            progress_before,
        )

        with patch('kkamang.ui.field.cat') as draw_cat:
            self.app.draw()
            foot_after = draw_cat.call_args[0][1]
            mood_after = draw_cat.call_args[0][5]

        self.assertEqual(foot_after, foot_before)
        self.assertEqual(mood_after, 'walk')

        # E를 누른 채로 있어도 획득 행동을 반복하지 않습니다.
        self.app.handle_event(pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_e,
            mod=0,
        ))
        self.assertEqual(
            self.app.core.snapshot()['tick'],
            after['tick'],
        )

        # 기존 이동 일정대로 다음 칸으로 계속 이동합니다.
        self.app.advance(0.2)
        self.assertEqual(
            self.app.core.snapshot()['position'],
            [3, 1],
        )


if __name__ == '__main__':
    unittest.main()
