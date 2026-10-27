"""게임 화면과 입력 흐름을 제어한다. 탐사 규칙은 코어에 위임하고 결과 정산·저장·엔딩을 연결한다."""
from pathlib import Path
import json
import math
import time
import uuid
import pygame

from .paths import user_data_dir
from .core import GameCore
from .realtime import RealtimeGameCore, FRAME_SECONDS, FRAMES_PER_TICK
from .levels import load_levels
from .campaign import Campaign, SaveStore
from .story import STAGES, PROLOGUE, ENDING, EXTENDED_ENDING, MEMORIES, memory_for_stage
from . import likes, collection
from .ending_content import ENDING_CONTENT
from .input import InputAdapter
from .audio import AudioEngine
from .ui.theme import Assets, text, panel, CREAM, INK, NIGHT
from .ui.field import draw_game
from .ui.ghost_sprites import movement_direction
from .ui.menus import render_menu
from .ui.collection import move_focus as collection_focus

SIZE = (1280, 720)
MEMORY_REVEAL_SECONDS = 2.
CINEMATIC_DURATIONS = {'prologue': 3.6}
VOLUME_SLIDERS = {
    'music': pygame.Rect(504, 199, 279, 25),
    'sfx': pygame.Rect(504, 271, 279, 25),
}

class GameApp:
    def __init__(self, save_path=None, headless=False, no_audio=False, size=SIZE, seed=0,
                 score_provider=None, movement_mode='realtime'):
        if movement_mode not in ('realtime', 'grid'):
            raise ValueError('movement_mode must be realtime or grid')
        self.movement_mode = movement_mode
        self.motion_accumulator = 0.
        # 외부 채점 함수는 (level, outcome, snapshot, mode)를 받아 점수 내역 또는 None을 반환한다.
        self.score_provider = score_provider
        self.ending_id = None
        self.ending_art_only = False
        self.ending_replay = False
        self.accumulator = 0.
        self.ghost_elapsed = 0.3
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE | (pygame.HIDDEN if headless else 0))
        pygame.display.set_caption('까망이와 유령 저택')
        self.canvas = pygame.Surface(SIZE)
        self.assets = Assets()
        try:
            pygame.display.set_icon(self.assets.cat((64,64)))
        except (pygame.error, FileNotFoundError):
            pass
        self.levels = load_levels()
        self.store = SaveStore(Path(save_path) if save_path else user_data_dir() / 'save_v02.json')
        self.campaign = self.store.load()
        self.audio = AudioEngine(not (headless or no_audio))
        self.audio.configure(self.campaign.data['settings'])
        self.input = InputAdapter()
        self.seed = seed
        self.running = True
        self.scene = 'title'
        self.transition_counter = 0
        self.focus = 0
        self.needs_focus = True
        self.controls = []
        self.stage = min(12, len(self.campaign.data['completed']) + 1)
        self.mode = self.campaign.data['settings'].get('mode','basic')
        self.core = None
        self.previous = None
        self.preview = None
        self.accumulator = 0.
        self.pause_accumulator = 0.
        self.elapsed = 0.
        self.scene_elapsed = 0.
        self.toast = ''
        self.toast_timer = 0.
        self.last_outcome = None
        self.display_elapsed_seconds = None
        self.last_reward = None
        self.run_id = None
        self.dialogues = []
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.story_kind = None
        self.ending_variant = 'basic'
        self.story_return = 'home'
        self.replaying_story = False
        self.photo_selected = self.campaign.data['photos'][-1] if self.campaign.data['photos'] else 1
        completed = self.campaign.data['completed']
        self.memory_selected = memory_for_stage(max(completed) if completed else 1)[0]
        self.reveal_region = None
        self.memory_completion_sound_played = False
        self.previous_regions = list(self.campaign.memory_regions(self.memory_selected))
        self.collection_tab = 'looks'
        self.collection_detail = None
        self.collection_selected = None
        self.collection_cover = False
        self.collection_notice = ''
        self.settings_return = 'title'
        self.tutorial_return = 'title'
        self.tutorial_starts_run = False
        self.confirm_kind = ''
        self.confirm_return = 'title'
        self.save_return = 'home'
        self.home_view = 'current'
        self.mouse = (-1,-1)
        self.volume_drag_name = None
        self.viewport = pygame.Rect(0,0,*size)
        self.shake_timer = 0.
        self.last_replay_error = None
        self.facing = 'down'
        self.ghost_facings = {}
        self.cat_mood = 'idle'
        self.mood_timer = 0.
        self.magnet_flights = []

    @property
    def story(self):
        return STAGES[self.stage - 1]

    def transition(self, scene):
        if self.scene == 'game':
            self.pause_accumulator=self.accumulator
        self.transition_counter += 1
        self.scene = scene
        self.audio.set_scene(scene, self.stage, story_kind=self.story_kind)
        self.scene_elapsed = 0.
        self.focus = 0
        self.needs_focus = True
        self.controls = []
        self.input.clear()
        self.accumulator = self.pause_accumulator if scene == 'game' else 0.
        if scene == 'game' and self.core is not None:
            self.previous = self.core.snapshot()
            self.ghost_elapsed = float(
                self.core.config.get('tick_seconds', 0.3)
            )

    def persist(self, next_scene=None):
        ok = self.store.save(self.campaign)
        if not ok:
            self.save_return = next_scene or self.scene
            self.transition('save_error')
        elif next_scene:
            self.transition(next_scene)
        return ok

    def prepare(self, stage=None):
        if stage is not None:
            self.stage = int(stage)
        if not self.campaign.is_unlocked(self.stage):
            return
        self.mode = self.campaign.data['settings'].get('mode','basic')
        self.preview = GameCore(self.levels[self.stage-1], mode=self.mode, seed=self.seed).snapshot()
        self.transition('briefing')

    def start_run(self):
        # 기본 모드의 대표 기록 선택이 남아 있어도 연습은 시작할 수 있으며 호감도 기록은 바꾸지 않는다.
        if self.mode == 'basic' and self.campaign.data['likes']['pending'] is not None:
            self.transition('memory_choice')
            return
        if not self.campaign.is_unlocked(self.stage):
            return
        core_class = RealtimeGameCore if self.movement_mode == 'realtime' else GameCore
        self.core = core_class(self.levels[self.stage-1], mode=self.mode, seed=self.seed)
        self.previous = self.core.snapshot()
        self.accumulator = 0.
        self.pause_accumulator=0.
        self.motion_accumulator = 0.
        self.facing='down'
        self.ghost_facings.clear()
        self.cat_mood='idle'
        self.mood_timer=0.
        self.magnet_flights.clear()
        self.run_id = uuid.uuid4().hex
        self.last_outcome = None
        self.display_elapsed_seconds = None
        self.last_reward = None
        self.toast = '아이템 + 온기 4점 이상 · 귀환점에서 E'
        self.toast_timer = 3.
        self.transition('game')

    def start_prologue(self):
        self._start_cinematic('prologue')

    def _start_cinematic(self, kind):
        self.dialogues = []
        self.story_kind = kind
        self.story_return = 'home'
        self.replaying_story = False
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.transition('story')

    def start_memory(self, stage=None, replay=False):
        if stage is not None:
            self.stage = int(stage)
        if not self.campaign.completed(self.stage):
            return
        self.memory_selected, region = memory_for_stage(self.stage)
        self.previous_regions = list(self.campaign.memory_regions(self.memory_selected))
        self.reveal_region = None if replay else region
        self.memory_completion_sound_played = False
        if self.reveal_region is not None:
            self.previous_regions[region] = False
        self.dialogues = []
        self.story_kind = 'memory'
        self.story_return = 'album' if replay else ('ending' if self.stage == 12 else 'home')
        self.replaying_story = replay
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.transition('story')
        if not replay:
            self.audio.play('memory_cold' if self.memory_selected == 3 else 'memory_reveal')

    def finish_story(self):
        if self.story_kind == 'prologue':
            self.campaign.data['intro_seen'] = True
            self.persist('home')
        elif self.story_kind == 'ending':
            self.persist('home')
        elif self.story_kind == 'ending_intro':
            self._start_afterword()
        elif self.story_return == 'ending':
            self.start_ending()
        else:
            self.transition(self.story_return)

    def _start_afterword(self):
        # 엔딩 이력과 PNG·내레이션을 분리하고 다시 보기는 활성화된 쉼터 상태를 바꾸지 않는다.
        self.dialogues = list(ENDING_CONTENT[self.ending_id]['lines'])
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.story_kind = 'ending'
        self.ending_art_only = True
        self.campaign.record_ending(self.ending_id, activate=not self.ending_replay)
        self.persist('ending')

    def start_ending(self, replay=False):
        if self.campaign.data['likes']['pending'] is not None:
            self.transition('memory_choice')
            return
        selected = self.campaign.selected_ending()
        if selected is None:
            # 미완료·미평가 기록으로 엔딩을 만들지 않고 호감도 화면에서 누락 점수를 확인한다.
            self.transition('likes')
            return
        self.ending_id = selected
        self.ending_variant = 'extended' if selected == 'happy_all' else 'basic'
        self.ending_replay = bool(replay)
        self.replaying_story = bool(replay)
        if replay:
            self._start_afterword()
        else:
            self._start_cinematic('ending_intro')

    def next_dialogue(self):

        # 기억 화면의 Enter·Space는 현재 선택한 버튼을 실행합니다.
        if (
            self.scene == 'story'
            and self.story_kind == 'memory'
            and not self.replaying_story
        ):
            if self.scene_elapsed < MEMORY_REVEAL_SECONDS:
                return

            if 0 <= self.focus < len(self.controls):
                selected = self.controls[self.focus]
                if selected['id'] in ('story_finish_home', 'story_finish_next', 'story_skip'):
                    if not selected.get('disabled'):
                        self.dispatch(selected['id'])
                    return

            # 아직 화면을 그리지 않았거나 마지막 방의 '엔딩 보기'인 경우입니다.
            self.dispatch('story_finish_next')
            return

        if self.scene == 'ending':
            if self.ending_art_only and self.dialogues:
                self.ending_art_only = False
                self.dialogue_time = 0.
            else:
                # 내레이션 다음 복귀 입력은 문장 표시를 기다리지 않고 즉시 처리한다.
                self.finish_story()
            return
        if self.dialogue_index >= len(self.dialogues):
            self.finish_story()
            return
        line = self.dialogues[self.dialogue_index][1]
        if int(self.dialogue_time * 38) < len(line):
            self.dialogue_time = len(line) / 38 + 1
            return
        self.dialogue_index += 1
        self.dialogue_time = 0.
        if self.dialogue_index >= len(self.dialogues):
            self.finish_story()

    def request_confirm(self, kind):
        self.confirm_kind = kind
        self.confirm_return = self.scene
        self.transition('confirm')
        self.focus = 0

    def write_replay(self):
        if not self.core:
            return
        try:
            folder = self.store.path.parent / 'replays'
            folder.mkdir(parents=True,exist_ok=True)
            record = self.core.replay_record()
            record['controller_type'] = 'human'
            record['settlement_id'] = self.run_id
            path = folder / f'{self.core.config["id"]}_{self.run_id}.json'
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            # 탐사 재생 파일은 캠페인 저장과 분리하며 최근 50개만 보관한다.
            old = sorted(folder.glob('*.json'), key=lambda p:p.stat().st_mtime)
            for p in old[:-50]:
                p.unlink()
        except OSError as exc:
            self.last_replay_error = str(exc)

    def finish_run(self, result):
        self.last_outcome = dict(result.outcome)

        # 중복 정산 호출에는 결과만 전달될 수 있다.
        snapshot = getattr(result, "snapshot", None)
        if snapshot is None:
            snapshot = self.core.snapshot()
        motion = snapshot.get("motion")

        if motion is not None:
            used_seconds = motion["frame"] * FRAME_SECONDS
        else:
            tick_seconds = float(
                self.core.config.get("tick_seconds", 0.3)
            )
            used_seconds = result.outcome["ticks_used"] * tick_seconds

        # 표시용 정밀 시간은 공식 결과·저장·점수 계산과 분리한다.
        self.display_elapsed_seconds = used_seconds
        memory = None
        if (self.core.mode == 'basic' and result.outcome['status'] == 'COMPLETE'
                and self.run_id not in self.campaign.data['settled_ids']):
            scores = (self.score_provider(self.core.config, result.outcome,
                                         result.snapshot, self.core.mode)
                      if self.score_provider else
                      (result.outcome.get('score_breakdown') or likes.from_game_score(result.outcome.get('score'))))
            memory = likes.make_record(self.core.config, result.outcome, result.snapshot,
                                       self.core.mode, self.run_id, self.core.config_hash, scores)
        self.last_reward = self.campaign.settle(self.core.config, result.outcome,
             self.core.config_hash, self.core.mode, self.run_id, controller='human',
             memory_record=memory)
        self.write_replay()
        self.audio.play('return' if result.outcome['status'] in ('COMPLETE','RETURN_LOW') else 'fail')
        needs_memory_choice = (self.core.mode == 'basic'
                               and self.campaign.data['likes']['pending'] is not None)
        self.persist('memory_choice' if needs_memory_choice else 'result')

    def abandon(self):
        if self.core and not self.core.snapshot()['terminal']:
            self.core.abort()
            self.write_replay()

    def dispatch(self, ident):
        self.audio.play('select')
        if ident in ('memory_keep', 'memory_replace'):
            self.resolve_memory(ident == 'memory_replace')
        elif ident == 'show_ending':
            self.start_ending(replay=self.campaign.selected_ending() in
                              self.campaign.data['likes']['seen_endings'])
        elif ident == 'choose_memory':
            self.transition('memory_choice')
        elif ident == 'quit':
            self.request_confirm('quit')
        elif ident == 'continue':
            if self.campaign.data['likes']['pending'] is not None:
                self.transition('memory_choice')
            elif self.campaign.data.get('intro_seen'):
                self.transition('home')
            else:
                self.start_prologue()
        elif ident == 'title':
            self.persist('title')
        elif ident == 'new_game':
            if self.store.has_save or self.campaign.data['completed']:
                self.request_confirm('new_game')
            else:
                self.campaign = Campaign()
                self.collection_notice = ''
                self.stage = 1
                self.start_prologue()
        elif ident in ('ending_conditions', 'likes'):
            self.transition('likes')
        elif ident in ('explore','rooms'):
            self.transition('rooms')
        elif ident.startswith('stage:'):
            self.prepare(int(ident.split(':')[1]))
        elif ident == 'start':
            if not self.campaign.data.get('tutorial_seen'):
                self.tutorial_return = 'briefing'
                self.tutorial_starts_run = True
                self.transition('tutorial')
            else:
                self.start_run()
        elif ident == 'tutorial_done':
            self.campaign.data['tutorial_seen'] = True
            if self.persist(self.tutorial_return) and self.tutorial_starts_run:
                self.start_run()
        elif ident == 'tutorial':
            self.tutorial_return = self.scene
            self.tutorial_starts_run = False
            self.transition('tutorial')
        elif ident == 'retry':
            # 재도전은 같은 방의 준비 화면으로 돌아가 목표와 모드를 다시 확인한다.
            self.prepare(self.stage)
        elif ident == 'restart':
            if self.scene in ('pause','game'):
                self.request_confirm('restart')
            else:
                self.start_run()
        elif ident == 'next_stage':
            if self.scene == 'result' and (self.last_reward or {}).get('practice_mode'):
                self.transition('rooms')
                return
            if self.scene == 'result' and (self.last_reward or {}).get('first_complete'):
                self.start_memory()
            elif self.stage == 12:
                self.start_ending()
            else:
                self.prepare(self.stage+1)
        elif ident in ('memory','view_memory'):
            self.start_memory()
        elif ident == 'story_finish_next':
            # 새 기억 감상 후에는 다음 방으로 이동하되 앨범 다시 보기와 마지막 방은 원래 복귀 흐름을 따른다.
            if self.story_kind == 'memory' and not self.replaying_story and self.stage < 12:
                self.prepare(self.stage + 1)
            else:
                self.finish_story()
        elif ident == 'story_finish_home':
            self.finish_story()
        elif ident == 'ending':
            self.start_ending(replay=self.scene != 'result' and
                              self.campaign.selected_ending() in
                              self.campaign.data['likes']['seen_endings'])
        elif ident == 'story_next':
            self.next_dialogue()
        elif ident == 'story_skip':
            self.finish_story()
        elif ident == 'home':
            if self.scene in ('pause','game'):
                self.request_confirm('home')
            else:
                self.transition('home')
        elif ident == 'toggle_home_view':
            self.home_view = 'before' if self.home_view == 'current' else 'current'
            event = 'shelter_before' if self.home_view == 'before' else 'shelter_return'
            self._collection_discovery(event)
        elif ident == 'album':
            self.transition('album')
        elif ident == 'friends':
            self.transition('likes')
        elif ident.startswith('memory:'):
            memory_id = int(ident.split(':')[1])
            if 1 <= memory_id <= len(MEMORIES):
                self.memory_selected = memory_id
                if self.scene == 'home':
                    self.transition('album')
        elif ident.startswith('photo:'):
            self.photo_selected = int(ident.split(':')[1])
            self.memory_selected = memory_for_stage(self.photo_selected)[0]
        elif ident == 'replay_memory':
            completed = [stage for stage in self.campaign.data['completed']
                         if memory_for_stage(stage)[0] == self.memory_selected]
            if completed:
                self.start_memory(max(completed), replay=True)
        elif ident == 'visit_memory':
            stages = range((self.memory_selected - 1) * 4 + 1, self.memory_selected * 4 + 1)
            available = [stage for stage in stages if self.campaign.is_unlocked(stage)]
            if available:
                target = next((stage for stage in available if not self.campaign.completed(stage)), available[-1])
                self.prepare(target)
        elif ident == 'visit_photo':
            self.prepare(self.photo_selected)
        elif ident == 'favorite_photo':
            if self.photo_selected in self.campaign.data['photos']:
                self.campaign.data['favorite_photo']=self.photo_selected
                self.persist()
        elif ident == 'appearance':
            collection.sync(self.campaign.data)
            self.collection_tab = 'looks'
            self.collection_detail = None
            self.collection_selected = None
            self.collection_cover = self.campaign.data['collection']['favorite'] is not None
            self.transition('appearance')
        elif ident.startswith(('collection_tab:', 'collection_open:')):
            self.collection_tab = ident.split(':')[1]
            self.collection_detail = None
            self.collection_selected = None
            self.collection_cover = False
            self.transition('appearance')
        elif ident.startswith('collection_card:'):
            _, kind, number = ident.split(':')
            self.collection_detail = (kind, int(number))
            self.collection_selected = int(number)
            self.transition('appearance')
            if kind == 'looks' and collection.mark_seen(self.campaign.data, int(number)):
                self.persist()
        elif ident == 'collection_favorite':
            if (self.collection_detail and self.collection_detail[0] == 'looks'
                    and collection.set_favorite(self.campaign.data, self.collection_detail[1])):
                self.persist()
        elif ident == 'collection_cover':
            self.collection_detail = None
            self.collection_cover = True
            self.transition('appearance')
        elif ident == 'collection_close':
            self.collection_detail = None
            self.transition('appearance')
        elif ident == 'collection_memory_paw':
            if self.scene == 'album' and self.campaign.memory_complete(self.memory_selected):
                self._collection_discovery('memory_paw')
        elif ident == 'collection_notice_close':
            self.collection_notice = ''
        elif ident == 'collection_notice_open':
            self.collection_notice = ''
            self.collection_detail = None
            self.collection_cover = False
            self.collection_tab = 'looks'
            self.transition('appearance')
            self.persist()
        elif ident == 'remember_friend':
            self.resolve_memory(True)
        elif ident == 'keep_friend':
            self.resolve_memory(False)
        elif ident == 'settings':
            self.settings_return = self.scene
            self.transition('settings')
        elif ident in ('music_down','music_up','sfx_down','sfx_up'):
            name,direction = ident.split('_')
            settings=self.campaign.data['settings']
            settings[name] = round(max(0.,min(1.,settings.get(name,.5) + (.1 if direction=='up' else -.1))),2)
            self.audio.configure(settings)
        elif ident == 'shake':
            settings=self.campaign.data['settings'];settings['shake']=not settings.get('shake',False)
        elif ident == 'mode':
            if self.scene != 'pause':
                self.mode = 'relaxed' if self.mode == 'basic' else 'basic'
                self.campaign.data['settings']['mode']=self.mode
                if self.scene == 'briefing':
                    self.preview=GameCore(self.levels[self.stage-1],mode=self.mode,seed=self.seed).snapshot()
        elif ident == 'resume':
            self.persist('game')
        elif ident == 'back':
            self.go_back()
        elif ident == 'confirm_no':
            self.transition(self.confirm_return)
        elif ident == 'confirm_yes':
            kind=self.confirm_kind
            if kind == 'quit':
                self.abandon()
                self.running=False
            elif kind == 'new_game':
                self.abandon()
                self.campaign=Campaign()
                self.collection_notice = ''
                self.stage=1
                self.mode='basic'
                self.start_prologue()
            elif kind == 'restart':
                self.abandon();self.start_run()
            elif kind == 'home':
                self.abandon();self.transition('home')
        elif ident == 'save_retry':
            self.persist(self.save_return if self.scene=='save_error' else self.scene)
        elif ident == 'retry_unsaved':
            self.persist(self.scene)
        elif ident == 'save_continue':
            self.transition(self.save_return)
        elif ident == 'ending_continue':
            self.next_dialogue()

    def go_back(self):
        s=self.scene
        if s == 'appearance' and self.collection_detail:
            self.collection_detail = None
            self.transition('appearance')
        elif s == 'memory_choice':
            self.resolve_memory(False)
        elif s == 'game':
            self.transition('pause')
        elif s == 'pause':
            self.persist('game')
        elif s in ('story','ending'):
            self.finish_story()
        elif s == 'confirm':
            self.transition(self.confirm_return)
        elif s == 'save_error':
            self.transition(self.save_return)
        elif s == 'home':
            self.persist('title')
        elif s == 'title':
            self.request_confirm('quit')
        elif s == 'settings':
            self.persist(self.settings_return)
        elif s in ('briefing','tutorial'):
            self.transition('rooms' if s == 'briefing' else self.tutorial_return)
        else:
            self.transition('home')

    def _collection_discovery(self, event):
        before = tuple(self.campaign.data['collection']['discoveries'])
        new = collection.discover(self.campaign.data, event)
        if new:
            self.collection_notice = f'새로운 모습 {len(new)}장을 발견했어요'
        if new or before != tuple(self.campaign.data['collection']['discoveries']):
            self.persist()

    def resolve_memory(self, replace):
        if self.campaign.choose_friend_memory(replace):
            if self.last_reward is not None:
                self.last_reward['friend_memory_replaced' if replace else 'friend_memory_kept'] = True
                selected = self.campaign.selected_ending()
                self.last_reward['new_ending'] = bool(
                    selected and selected not in self.campaign.data['likes']['seen_endings'])
            # 불러온 미확정 기록에는 직전 결과 화면의 임시 문맥이 없을 수 있다.
            self.persist('result' if self.last_outcome is not None else 'likes')

    def _perform_action(self, action, animation_age=0.):
        """한 틱을 실행하고 보간 시작점에는 실제 이전 칸을 보관한다."""
        if self.scene != 'game' or self.core is None:
            return

        # 방향 전환 때 두 축이 동시에 보간되지 않도록 화면 중간 좌표 대신 코어의 정수 좌표를 보관한다.
        before = self.core.snapshot()
        self.previous = before
        result = self.core.step(action)
        self.ghost_elapsed = animation_age
        self._remember_ghost_facings(before, result.snapshot)
        self._process_result(result)
        return result

    def _remember_ghost_facings(self, before, after):
        """대기·접촉 정지 중에도 화면에 표시할 유령 방향을 유지한다."""
        old = {g['id']: g['position'] for g in before['ghosts']}
        if before.get('chaser') and before['chaser']['active']:
            old[before['chaser']['id']] = before['chaser']['position']
        current = list(after['ghosts'])
        if after.get('chaser') and after['chaser']['active']:
            current.append(after['chaser'])
        for ghost in current:
            if ghost['id'] in old and old[ghost['id']] != ghost['position']:
                self.ghost_facings[ghost['id']] = movement_direction(
                    old[ghost['id']], ghost['position'],
                    self.ghost_facings.get(ghost['id'], 'down'))

    def _process_result(self, result):
        """공통 코어 이벤트를 표시하고 끝난 탐사는 한 번만 정산한다."""
        for event in result.events:
            kind = event['type']
            self.audio.play_event(event, quest=self.core.config['quest'] if kind == 'quest_collect' else None)

            if kind == 'move':
                dx = event['to'][0] - event['from'][0]
                dy = event['to'][1] - event['from'][1]
                self.facing = (
                    'right' if dx > 0
                    else 'left' if dx < 0
                    else 'down' if dy > 0
                    else 'up'
                )

            if kind in ('collect', 'quest_collect'):
                self.cat_mood = 'collect'
                self.mood_timer = .5

            if kind == 'quest_collect':
                self.toast = (
                    event['name']
                    + ' 회수! 온기와 함께 문으로 돌아가요.'
                )
                self.toast_timer = 2.

            elif kind == 'return_blocked':
                self.toast = (
                    '필수 물건과 온기를 모두 챙겨야 해요. '
                    '탐사는 계속할 수 있어요.'
                )
                self.toast_timer = 2.

            if kind == 'collect':
                if event.get('magnetic'):
                    self.magnet_flights.append({
                        'origin': list(event['source_position']),
                        'value': event['value'],
                        'age': 0.,
                        'duration': .34,
                    })
                self.toast = (
                    f'온기를 모았어요 · 보유 '
                    f'{result.snapshot["carried"]}점'
                )
                self.toast_timer = 1.5

            elif kind == 'magnet_collect':
                self.toast = '자석 획득! 13틱 동안 주변 3×3 온기를 끌어와요'
                self.toast_timer = 2.

            elif kind == 'invincibility_collect':
                self.toast = '무적 획득! 10틱 동안 유령과 부딪혀도 안전해요'
                self.toast_timer = 2.

            elif kind == 'open':
                self.toast = '낮은 샛문이 열렸어요'
                self.toast_timer = 1.5

            elif kind == 'damage':
                self.cat_mood = 'surprised'
                self.mood_timer = .9
                self.toast = (
                    '용기 -1 · 다음 3틱 보호'
                )
                self.toast_timer = 1.5
                self.shake_timer = .2

        if result.terminated:
            self.finish_run(result)

        # 같은 틱에 여러 온기를 얻어도 필수 물건 안내를 덮어쓰지 않는다.
        if (any(event['type'] == 'magnet_collect' for event in result.events)
                and not any(event['type'] == 'damage' for event in result.events)):
            self.toast = '자석 획득! 13틱 동안 주변 3×3 온기를 끌어와요'
            self.toast_timer = 2.
        if (any(event['type'] == 'invincibility_collect' for event in result.events)
                and not any(event['type'] == 'damage' for event in result.events)):
            self.toast = '무적 획득! 10틱 동안 유령과 부딪혀도 안전해요'
            self.toast_timer = 2.

        return result

    def _start_volume_drag(self):
        if self.scene not in ('settings', 'pause'):
            return False

        for name, rect in VOLUME_SLIDERS.items():
            if rect.collidepoint(self.mouse):
                self.volume_drag_name = name
                self._set_volume_from_slider()
                return True

        return False

    def _set_volume_from_slider(self):
        rect = VOLUME_SLIDERS[self.volume_drag_name]
        slider_value = (self.mouse[0] - rect.left) / rect.width
        slider_value = round(max(0.0, min(1.0, slider_value)), 2)

        settings = self.campaign.data['settings']
        settings[self.volume_drag_name] = slider_value
        self.audio.configure(settings)
    
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.request_confirm('quit')
            return
        if event.type == pygame.WINDOWFOCUSLOST:
            self.input.clear()
            if self.scene == 'game':
                self.transition('pause')
            return
        if event.type == pygame.VIDEORESIZE:
            self.window=pygame.display.set_mode((max(800,event.w),max(450,event.h)),pygame.RESIZABLE)
            return
        if event.type in (pygame.MOUSEMOTION,pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            self.mouse=self.to_canvas(event.pos)
        if self.scene == 'game':
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.transition('pause')
                return

            self.input.handle(event)

            # 첫 입력만 즉시 처리하고 이후 입력은 공통 틱에 맞춰 연타가 유령·시간을 가속하지 않게 한다.
            if (self.movement_mode == 'grid'
                    and self.core.snapshot()['tick'] == 0 and self.input.pending):
                action = (4 if self.input.consume_interaction()
                          else self.input.consume_movement(self.core.can_move))
                self.accumulator = 0.
                self._perform_action(action)

            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._start_volume_drag():
                return
        elif event.type == pygame.MOUSEMOTION and self.volume_drag_name is not None:
            self._set_volume_from_slider()
            return
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.volume_drag_name is not None:
                self._set_volume_from_slider()
                self.volume_drag_name = None
                return
        
        if event.type == pygame.MOUSEMOTION:
            for i,button in enumerate(self.controls):
                if not button.get('disabled') and button['rect'].collidepoint(self.mouse):
                    self.focus=i
                    break
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.controls:
                if not button.get('disabled') and button['rect'].collidepoint(self.mouse):
                    self.dispatch(button['id']);break
        elif event.type == pygame.KEYDOWN:
            self.mouse=(-1,-1)
            if event.key == pygame.K_ESCAPE:
                self.go_back()
            elif self.scene in ('story','ending') and event.key in (pygame.K_RETURN,pygame.K_SPACE):
                self.next_dialogue()
            elif self.scene == 'appearance' and event.key in (pygame.K_DOWN, pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT):
                self.focus = collection_focus(self.controls, self.focus, event.key)
            elif event.key in (pygame.K_TAB,pygame.K_DOWN,pygame.K_UP,pygame.K_LEFT,pygame.K_RIGHT):
                direction=-1 if event.key in (pygame.K_UP,pygame.K_LEFT) or (event.key==pygame.K_TAB and event.mod & pygame.KMOD_SHIFT) else 1
                if self.controls:
                    for _ in self.controls:
                        self.focus=(self.focus+direction)%len(self.controls)
                        if not self.controls[self.focus].get('disabled'):
                            break
            elif event.key in (pygame.K_RETURN,pygame.K_SPACE) and self.controls:
                self.focus=min(self.focus,len(self.controls)-1)
                button=self.controls[self.focus]
                if not button.get('disabled'):
                    self.dispatch(button['id'])

    def advance(self, dt):
        self.audio.update(dt)
        self.elapsed += dt
        self.scene_elapsed += dt
        self.dialogue_time += dt

        self.toast_timer = max(0., self.toast_timer - dt)
        self.mood_timer = max(0., self.mood_timer - dt)
        self.shake_timer = max(0., self.shake_timer - dt)

        if self.scene == 'story':
            if self.story_kind == 'memory':
                # 기억 공개가 끝나면 활성화된 '다음 방으로'를 기본 선택합니다.
                if (
                    not self.replaying_story
                    and self.scene_elapsed - dt < MEMORY_REVEAL_SECONDS <= self.scene_elapsed
                ):
                    self.needs_focus = True
                complete = self.campaign.memory_complete(self.memory_selected)
                if (not self.replaying_story and complete
                        and self.scene_elapsed >= MEMORY_REVEAL_SECONDS
                        and not self.memory_completion_sound_played):
                    self.memory_completion_sound_played = True
                    self.audio.play('memory_complete')
                # if (not self.replaying_story and not complete
                #         and self.scene_elapsed >= MEMORY_REVEAL_SECONDS):
                #     self.finish_story()
            elif self.scene_elapsed >= CINEMATIC_DURATIONS.get(self.story_kind, float('inf')):
                self.finish_story()
            return

        if self.scene != 'game' or self.core is None:
            return

        # 메뉴·일시정지 중에는 흡입 연출도 멈춘다.
        for flight in self.magnet_flights:
            flight['age'] += dt
        self.magnet_flights = [flight for flight in self.magnet_flights
                               if flight['age'] < flight['duration']]

        if self.movement_mode == 'realtime':
            self._advance_realtime(dt)
            return

        self.accumulator += dt
        self.ghost_elapsed += dt

        tick_seconds = float(
            self.core.config.get('tick_seconds', 0.3)
        )

        # 한 틱은 상호작용·이동·대기 중 하나만 처리하며 상호작용 뒤에도 미처리 이동 입력은 보존한다.
        while (
            self.scene == 'game'
            and self.accumulator + 1e-9 >= tick_seconds
        ):
            self.accumulator = max(
                0.,
                self.accumulator - tick_seconds,
            )

            action = (4 if self.input.consume_interaction()
                      else self.input.consume_movement(self.core.can_move))

            self._perform_action(
                action,
                animation_age=self.accumulator,
            )

    def _advance_realtime(self, dt):
        self.motion_accumulator += dt
        self.ghost_elapsed += dt
        while self.scene == 'game' and self.motion_accumulator + 1e-12 >= FRAME_SECONDS:
            self.motion_accumulator = max(0., self.motion_accumulator - FRAME_SECONDS)
            before = self.core.snapshot()
            # 칸 중간에서도 최신 방향을 즉시 적용하며 상호작용도 한 프레임을 소비한다.
            action = (4 if self.input.consume_interaction()
                      else self.input.consume_movement())
            result = self.core.step(action)
            if result.snapshot['tick'] != before['tick']:
                self.previous = before
                self._remember_ghost_facings(before, result.snapshot)
                self.ghost_elapsed = self.motion_accumulator
            self._process_result(result)
        if self.scene == 'game':
            self.accumulator = (self.core.snapshot()['motion']['phase'] *
                                FRAMES_PER_TICK * FRAME_SECONDS + self.motion_accumulator)

    def context(self):
        snapshot = self.core.snapshot() if self.core else self.preview
        if self.scene == 'briefing':
            snapshot=self.preview
        dialogue=None
        if (self.scene == 'ending' and not self.ending_art_only
                and self.dialogues and self.dialogue_index < len(self.dialogues)):
            speaker,line=self.dialogues[self.dialogue_index]
            dialogue=(speaker,line[:int(self.dialogue_time*38)])
        story=self.story
        confirm_messages={
            'quit':('저택을 나갈까요?','현재 탐사 중인 온기는 확정되지 않아요. 이미 저장한 기억은 그대로 남아요.'),
            'new_game':('처음부터 다시 만날까요?','현재 이야기 진행을 새 진행으로 덮어써요. 이전 사진과 배지 기록도 초기화돼요.'),
            'restart':('같은 방에 다시 들어갈까요?','이번 판의 온기는 포기하고, 같은 배치와 순찰 박자로 시작해요.'),
            'home':('쉼터로 돌아갈까요?','귀환문에서 돌아온 것이 아니므로 이번 판의 온기는 기록되지 않아요.')}
        title,message=confirm_messages.get(self.confirm_kind,('확인','계속할까요?'))
        regions = list(self.campaign.memory_regions(self.memory_selected))
        showing_memory = self.scene == 'story' and self.story_kind == 'memory'
        silent = self.scene == 'story' and self.story_kind in ('prologue', 'memory', 'ending_intro')
        active_ending = self.campaign.active_ending()
        released = active_ending == 'happy_all'
        duration = CINEMATIC_DURATIONS.get(self.story_kind, MEMORY_REVEAL_SECONDS)
        quest = self.levels[self.stage - 1]['quest']
        ending_content = ENDING_CONTENT.get(self.ending_id)
        return dict(scene=self.scene,campaign=self.campaign,levels=self.levels,stories=STAGES,stage=self.stage,mode=self.mode,
            snapshot=snapshot,outcome=self.last_outcome,reward=self.last_reward,story=story,dialogue=dialogue,
            display_elapsed_seconds=(self.display_elapsed_seconds
                                     if self.last_outcome is not None else None),
            photo_selected=self.photo_selected,has_save=self.store.has_save,
            collection_tab=self.collection_tab,collection_detail=self.collection_detail,
            collection_cover=self.collection_cover,
            collection_notice=getattr(self, 'collection_notice', ''),
            preferred_photo=self.campaign.data.get('favorite_photo'),
            message=('저장되지 않은 진행이 있어요. 다시 저장해 주세요.' if self.store.error else ''),from_game=self.scene=='pause',background_snapshot=snapshot,
            elapsed=self.elapsed,scene_elapsed=self.scene_elapsed,dialogue_index=self.dialogue_index,total_dialogue=len(self.dialogues),
            story_kind=self.story_kind, ending_variant=self.ending_variant,
            warmth_count=len(self.campaign.warmth_memories()),
            released=released,active_ending=active_ending,
            selected_ending=self.campaign.selected_ending(),
            released_variants=self.campaign.released_variants(),
            ending_art_only=self.ending_art_only,
            narration_progress=min(1., self.dialogue_time / .5),
            confirm_kind=self.confirm_kind,confirm_title=title,confirm_text=message,save_error=self.store.error,
            replay=self.replaying_story,settings_return=self.settings_return,home_view=self.home_view,
            total_stars=self.campaign.total_stars(),
            audio_available=self.audio.available,ending_lines=ENDING,
            likes_totals=likes.totals(self.campaign.data['likes']),
            memory_comparison=likes.comparison(self.campaign.data['likes']),
            memory_selected=self.memory_selected,memory_id=self.memory_selected,
            memory=MEMORIES[self.memory_selected - 1],regions=regions,
            previous_regions=list(self.previous_regions) if showing_memory and not self.replaying_story else list(regions),
            reveal_region=self.reveal_region if showing_memory else None,
            reveal_progress=1. if self.replaying_story else min(1.,self.scene_elapsed / MEMORY_REVEAL_SECONDS),
            memory_progress=self.campaign.memory_progress(),silent=silent,
            cinematic_kind=self.story_kind if silent else None,
            cinematic_progress=min(1.,self.scene_elapsed / duration),
            item_kind=quest['kind'],item_name=quest['name'],current_story_item_kind=quest['kind'],
            ending_id=self.ending_id, ending_content=ending_content)

    def draw(self):
        self.audio.configure(self.campaign.data['settings'],
                             completed=self.campaign.data['completed'],
                             released=self.campaign.active_ending() == 'happy_all')
        self.canvas.fill(NIGHT)
        if self.scene == 'game':
            snapshot=self.core.snapshot()
            tick_seconds = float(
                self.core.config.get('tick_seconds', 0.3)
            )

            remaining_seconds = max(
                0.,
                snapshot['remaining_ticks'] * tick_seconds
                - self.accumulator,
            )

            draw_game(
                self.canvas,
                self.assets,
                snapshot,
                self.story,
                elapsed=self.elapsed,
                previous=self.previous,
                interpolation=min(
                    1.,
                    self.ghost_elapsed / tick_seconds,
                ),
                player_interpolation=min(
                    1.,
                    self.accumulator / tick_seconds,
                ),
                toast=(
                    self.toast if self.toast_timer > 0 else ''
                ),
                cosmetics=self.campaign.data['cosmetics'],
                facing=self.facing,
                mood=(
                    self.cat_mood
                    if self.mood_timer > 0
                    else 'idle'
                ),
                magnet_flights=self.magnet_flights,
                ghost_facings=self.ghost_facings,
                remaining_seconds=remaining_seconds,
            )
            self.controls=[]
        else:
            screen='settings' if self.scene == 'pause' else self.scene
            self.controls=render_menu(self.canvas,self.assets,screen,self.context(),self.focus,self.mouse)
            if self.needs_focus:
                preferred={'result':['view_memory','ending','show_ending','next_stage','retry'], 'pause':['resume'],
                    'memory_choice':['memory_keep'], 'likes':['home'],
                    'settings':['back'], 'briefing':['start'], 'rooms':[f'stage:{min(12,len(self.campaign.data["completed"])+1)}'],
                    'home':['explore'], 'friends':['back'],
                    # 기억 화면의 기본 포인터: 다음 방 → 계속/엔딩 → 쉼터 순서입니다.
                    'story':['story_finish_next','story_next','story_finish_home'],
                    'ending':['ending_continue'],
                    'tutorial':['tutorial_done'], 'confirm':['confirm_no'], 'save_error':['save_retry'],
                    'album':[f'memory:{self.memory_selected}',f'photo:{self.photo_selected}'],
                    'appearance':(['collection_favorite','collection_close'] if self.collection_detail else
                        ['collection_open:looks'] if self.collection_cover else
                        [f'collection_card:{self.collection_tab}:{self.collection_selected}',
                         f'collection_tab:{self.collection_tab}'])}.get(self.scene,[])
                for ident in preferred:
                    match=next((i for i,b in enumerate(self.controls) if b['id']==ident and not b.get('disabled')),None)
                    if match is not None:
                        self.focus=match
                        break
                self.needs_focus=False
                self.controls=render_menu(self.canvas,self.assets,screen,self.context(),self.focus,self.mouse)
            self.focus=min(self.focus,max(0,len(self.controls)-1))
        # 저장 실패 후 계속하기를 선택해도 현재 세션의 진행은 유지한다.
        if self.store.error and self.scene not in ('save_error','title'):
            panel(self.canvas,(880,6,380,34),fill=(71,42,66),border=(217,151,129),radius=10)
            label='아직 저장되지 않았어요 · 일시정지 후 재시도' if self.scene=='game' else '아직 저장되지 않았어요 · 클릭해서 재시도'
            text(self.canvas,self.assets,label, (1070,23),size=15,color=CREAM,anchor='center')
            if self.scene != 'game':
                self.controls.append({'id':'retry_unsaved','rect':pygame.Rect(880,6,380,34),'label':label,'disabled':False})
        w,h=self.window.get_size()
        scale=min(w/1280,h/720)
        out_size=(max(1,round(1280*scale)),max(1,round(720*scale)))
        self.viewport=pygame.Rect((w-out_size[0])//2,(h-out_size[1])//2,*out_size)
        self.window.fill((15,12,23))
        image=self.canvas if out_size==SIZE else pygame.transform.smoothscale(self.canvas,out_size)
        offset=(0,0)
        if self.shake_timer>0 and self.scene=='game' and self.campaign.data['settings'].get('shake'):
            offset=(int(math.sin(self.elapsed*90)*3),int(math.cos(self.elapsed*70)*2))
        self.window.blit(image,(self.viewport.x+offset[0],self.viewport.y+offset[1]))
        pygame.display.flip()

    def to_canvas(self, pos):
        if not self.viewport.collidepoint(pos):
            return (-1,-1)
        return ((pos[0]-self.viewport.x)*1280/self.viewport.width,(pos[1]-self.viewport.y)*720/self.viewport.height)

    def run(self, max_frames=None, screenshot=None):
        clock=pygame.time.Clock()
        frames=0
        self.draw()
        while self.running:
            dt=clock.tick(60)/1000
            before_events = self.transition_counter

            # 멈췄던 프레임에서는 입력 실행보다 일시정지를 우선한다.
            if self.scene == 'game' and dt > 1.0:
                self.transition('pause')

            for event in pygame.event.get():
                self.handle_event(event)
                
            self.advance(0. if self.transition_counter != before_events else dt)
            self.draw()
            frames+=1
            if max_frames is not None and frames>=max_frames:
                break
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(self.canvas,str(screenshot))
        self.close()

    def close(self):
        self.audio.close()
        pygame.quit()
