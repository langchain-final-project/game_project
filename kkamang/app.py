"""Human application controller. Rendering reads state; only GameCore advances runs."""
from pathlib import Path
import json
import math
import time
import uuid
import pygame

from .paths import user_data_dir
from .core import GameCore
from .levels import load_levels
from .campaign import Campaign, SaveStore
from .story import STAGES, PROLOGUE, ENDING
from . import likes
from .ending_content import ENDING_CONTENT
from .input import InputAdapter
from .audio import AudioEngine
from .ui.theme import Assets, text, panel, CREAM, INK, NIGHT
from .ui.field import draw_game
from .ui.menus import render_menu

SIZE = (1280, 720)

class GameApp:
    def __init__(self, save_path=None, headless=False, no_audio=False, size=SIZE, seed=0,
                 score_provider=None):
        # Owner-supplied callable(level, outcome, snapshot, mode) -> score_breakdown | None.
        self.score_provider = score_provider
        self.ending_id = None
        self.accumulator = 0.
        self.ghost_elapsed = 0.3
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE | (pygame.HIDDEN if headless else 0))
        pygame.display.set_caption('까망이와 잠 못 드는 저택')
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
        self.toast = ''
        self.toast_timer = 0.
        self.last_outcome = None
        self.last_reward = None
        self.run_id = None
        self.dialogues = []
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.story_kind = None
        self.story_return = 'home'
        self.replaying_story = False
        self.photo_selected = self.campaign.data['photos'][-1] if self.campaign.data['photos'] else 1
        self.appearance_kind = 'scarf'
        self.settings_return = 'title'
        self.tutorial_return = 'title'
        self.tutorial_starts_run = False
        self.confirm_kind = ''
        self.confirm_return = 'title'
        self.save_return = 'home'
        self.home_view = 'current'
        self.mouse = (-1,-1)
        self.viewport = pygame.Rect(0,0,*size)
        self.shake_timer = 0.
        self.last_replay_error = None
        self.facing = 'down'
        self.cat_mood = 'idle'
        self.mood_timer = 0.

    @property
    def story(self):
        return STAGES[self.stage - 1]

    def transition(self, scene):
        if self.scene == 'game':
            self.pause_accumulator=self.accumulator
        self.transition_counter += 1
        self.scene = scene
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
        if self.campaign.data['likes']['pending'] is not None:
            self.transition('memory_choice')
            return
        if not self.campaign.is_unlocked(self.stage):
            return
        self.core = GameCore(self.levels[self.stage-1], mode=self.mode, seed=self.seed)
        self.previous = self.core.snapshot()
        self.pause_accumulator=0.
        self.facing='down'
        self.cat_mood='idle'
        self.mood_timer=0.
        self.run_id = uuid.uuid4().hex
        self.last_outcome = None
        self.last_reward = None
        self.toast = '필수 물건과 온기 4점! 같은 칸에서 E로 회수해요.'
        self.toast_timer = 3.
        self.transition('game')

    def start_prologue(self):
        self.dialogues = list(PROLOGUE)
        self.story_kind = 'prologue'
        self.story_return = 'home'
        self.replaying_story = False
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.transition('story')

    def start_memory(self, stage=None, replay=False):
        if stage is not None:
            self.stage = int(stage)
        if self.stage not in self.campaign.data['photos']:
            return
        self.dialogues = list(self.story['lines'])
        self.story_kind = 'memory'
        self.story_return = 'album' if replay else ('ending' if self.stage == 12 else 'home')
        self.replaying_story = replay
        self.dialogue_index = 0
        self.dialogue_time = 0.
        self.transition('story')

    def finish_story(self):
        if self.story_kind == 'prologue':
            self.campaign.data['intro_seen'] = True
            self.persist('home')
        elif self.story_kind == 'ending':
            self.transition('home')
        elif self.story_return == 'ending':
            self.start_ending()
        else:
            self.transition(self.story_return)

    def start_ending(self):
        if self.campaign.data['likes']['pending'] is not None:
            self.transition('memory_choice')
            return
        if len(self.campaign.data['completed']) != likes.ROOM_COUNT:
            self.transition('likes')
            return
        self.ending_id = likes.select_ending(self.campaign.data['likes'], self.campaign.data['completed'])
        content = ENDING_CONTENT.get(self.ending_id)
        self.dialogues = list(content['lines'] if content else ENDING)
        self.dialogue_index=0
        self.dialogue_time=0.
        self.story_kind='ending'
        if self.ending_id and self.ending_id not in self.campaign.data['likes']['seen_endings']:
            self.campaign.data['likes']['seen_endings'].append(self.ending_id)
            self.persist('ending')
        else:
            self.transition('ending')

    def next_dialogue(self):
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
        self.focus = 0  # Menus put safe cancel first.

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
            # Keep local disk usage bounded; replay export is separate from campaign save.
            old = sorted(folder.glob('*.json'), key=lambda p:p.stat().st_mtime)
            for p in old[:-50]:
                p.unlink()
        except OSError as exc:
            self.last_replay_error = str(exc)

    def finish_run(self, result):
        self.last_outcome = result.outcome
        memory = None
        if (result.outcome['status'] == 'COMPLETE'
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
        self.persist('memory_choice' if self.campaign.data['likes']['pending'] else 'result')

    def abandon(self):
        if self.core and not self.core.snapshot()['terminal']:
            self.core.abort()
            self.write_replay()

    def dispatch(self, ident):
        self.audio.play('select')
        if ident in ('memory_keep', 'memory_replace'):
            self.resolve_memory(ident == 'memory_replace')
        elif ident == 'show_ending':
            self.start_ending()
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
                self.stage = 1
                self.start_prologue()
        # 엔딩 조건
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
        elif ident in ('retry','restart'):
            if self.scene in ('pause','game'):
                self.request_confirm('restart')
            else:
                self.start_run()
        elif ident == 'next_stage':
            if self.scene == 'result' and (self.last_reward or {}).get('new_photo'):
                self.start_memory()
            elif self.stage == 12:
                self.start_ending()
            else:
                self.prepare(self.stage+1)
        elif ident in ('memory','view_memory'):
            self.start_memory()
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
        elif ident == 'album':
            self.transition('album')
        elif ident == 'friends':
            self.transition('likes')
        elif ident.startswith('photo:'):
            self.photo_selected = int(ident.split(':')[1])
        elif ident == 'replay_memory':
            self.start_memory(self.photo_selected,replay=True)
        elif ident == 'visit_photo':
            self.prepare(self.photo_selected)
        elif ident == 'favorite_photo':
            if self.photo_selected in self.campaign.data['photos']:
                self.campaign.data['favorite_photo']=self.photo_selected
                self.persist()
        elif ident == 'appearance':
            self.transition('appearance')
        elif ident.startswith('appearance_tab:'):
            self.appearance_kind = ident.split(':')[1]
        elif ident.startswith('cosmetic:'):
            _,kind,value=ident.split(':')
            self.campaign.set_cosmetic(kind,value)
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
        if s == 'memory_choice':
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
        elif s == 'title':
            self.request_confirm('quit')
        elif s == 'settings':
            self.persist(self.settings_return)
        elif s in ('briefing','tutorial'):
            self.transition('rooms' if s == 'briefing' else self.tutorial_return)
        else:
            self.transition('home')

    def resolve_memory(self, replace):
        if self.campaign.choose_friend_memory(replace):
            if self.last_reward is not None:
                self.last_reward['friend_memory_replaced' if replace else 'friend_memory_kept'] = True
            # A reloaded pending choice has no transient result-screen context.
            self.persist('result' if self.last_outcome is not None else 'likes')

    def _perform_action(self, action, animation_age=0.):
        """행동을 실행하되, E는 고양이의 이동 애니메이션을 유지합니다."""
        if self.scene != 'game' or self.core is None:
            return

        tick_seconds = float(
            self.core.config.get('tick_seconds', 0.3)
        )
        before = self.core.snapshot()

        if self.previous is not None:
            # 유령은 현재 화면에 보이는 위치에서 다음 이동을 시작합니다.
            alpha = min(1., self.ghost_elapsed / tick_seconds)

            old_ghosts = {
                ghost['id']: ghost
                for ghost in self.previous['ghosts']
            }

            for ghost in before['ghosts']:
                old = old_ghosts.get(ghost['id'])

                if old is not None:
                    ghost['position'] = [
                        old['position'][k]
                        + (
                            ghost['position'][k]
                            - old['position'][k]
                        ) * alpha
                        for k in (0, 1)
                    ]

            chaser = before.get('chaser')
            old_chaser = self.previous.get('chaser')

            if (
                chaser is not None
                and old_chaser is not None
                and chaser['active']
                and old_chaser['active']
                and chaser['id'] == old_chaser['id']
            ):
                chaser['position'] = [
                    old_chaser['position'][k]
                    + (
                        chaser['position'][k]
                        - old_chaser['position'][k]
                    ) * alpha
                    for k in (0, 1)
                ]

            # E를 눌러도 고양이의 이동 시작 좌표를 바꾸지 않습니다.
            if action == 4:
                before['position'] = list(
                    self.previous['position']
                )

        self.previous = before
        result = self.core.step(action)
        self.ghost_elapsed = animation_age

        for event in result.events:
            kind = event['type']

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
                self.audio.play('collect')
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
                self.audio.play('collect')
                self.toast = (
                    f'온기를 모았어요 · 보유 '
                    f'{result.snapshot["carried"]}점'
                )
                self.toast_timer = 1.5

            elif kind == 'open':
                self.audio.play('open')
                self.toast = '낮은 샛문이 열렸어요'
                self.toast_timer = 1.5

            elif kind == 'damage':
                self.cat_mood = 'surprised'
                self.mood_timer = .9
                self.audio.play('damage')
                self.toast = (
                    '괜찮아, 잠깐 숨을 고르자 · 다음 3틱 보호'
                )
                self.toast_timer = 1.5
                self.shake_timer = .2

        if result.terminated:
            self.finish_run(result)

        return result
    
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
        if event.type in (pygame.MOUSEMOTION,pygame.MOUSEBUTTONDOWN):
            self.mouse=self.to_canvas(event.pos)
        if self.scene == 'game':
            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                self.transition('pause')
                return

            self.input.handle(event)

            # E는 다음 정기 틱까지 기다리지 않고 실행합니다.
            if self.input.consume_interaction():
                self._perform_action(4)

            # 귀환·실패로 화면이 바뀌었다면 입력 처리를 끝냅니다.
            if self.scene != 'game':
                return

            position = self.core.snapshot()['position']
            moving = (
                self.previous is not None
                and self.previous['position'] != position
            )

            # 멈춰 있을 때 누른 방향키는 즉시 이동을 시작합니다.
            if not moving:
                action = self.input.consume_movement()

                if 0 <= action <= 3:
                    self.accumulator = 0.
                    self._perform_action(action)

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
        self.elapsed += dt
        self.dialogue_time += dt

        self.toast_timer = max(0., self.toast_timer - dt)
        self.mood_timer = max(0., self.mood_timer - dt)
        self.shake_timer = max(0., self.shake_timer - dt)

        if self.scene != 'game' or self.core is None:
            return

        self.accumulator += dt
        self.ghost_elapsed += dt

        tick_seconds = float(
            self.core.config.get('tick_seconds', 0.3)
        )

        # 누르고 있는 방향키의 이동과 자동 대기를 처리합니다.
        while (
            self.scene == 'game'
            and self.accumulator + 1e-9 >= tick_seconds
        ):
            self.accumulator = max(
                0.,
                self.accumulator - tick_seconds,
            )

            action = self.input.consume_movement()

            self._perform_action(
                action,
                animation_age=self.accumulator,
            )
    def context(self):
        snapshot = self.core.snapshot() if self.core else self.preview
        if self.scene == 'briefing':
            snapshot=self.preview
        dialogue=None
        if self.dialogues and self.dialogue_index < len(self.dialogues):
            speaker,line=self.dialogues[self.dialogue_index]
            dialogue=(speaker,line[:int(self.dialogue_time*38)])
        story=self.story
        if self.story_kind == 'prologue' and self.scene == 'story':
            story={'stage':0,'name':'폭풍이 지나가는 밤','title':'문 안쪽의 작은 불빛','photo':'첫 번째 만남','art':'title_background.png','lines':list(PROLOGUE)}
        confirm_messages={
            'quit':('저택을 나갈까요?','현재 탐사 중인 온기는 확정되지 않아요. 이미 저장한 기억은 그대로 남아요.'),
            'new_game':('처음부터 다시 만날까요?','현재 이야기 진행을 새 진행으로 덮어써요. 이전 사진과 배지 기록도 초기화돼요.'),
            'restart':('같은 방에 다시 들어갈까요?','이번 판의 온기는 포기하고, 같은 배치와 순찰 박자로 시작해요.'),
            'home':('쉼터로 돌아갈까요?','귀환문에서 돌아온 것이 아니므로 이번 판의 온기는 기록되지 않아요.')}
        title,message=confirm_messages.get(self.confirm_kind,('확인','계속할까요?'))
        return dict(campaign=self.campaign,levels=self.levels,stories=STAGES,stage=self.stage,mode=self.mode,
            snapshot=snapshot,outcome=self.last_outcome,reward=self.last_reward,story=story,dialogue=dialogue,
            photo_selected=self.photo_selected,appearance_kind=self.appearance_kind,has_save=self.store.has_save,
            preferred_photo=self.campaign.data.get('favorite_photo'),
            message=('저장되지 않은 진행이 있어요. 다시 저장해 주세요.' if self.store.error else ''),from_game=self.scene=='pause',background_snapshot=snapshot,
            elapsed=self.elapsed,dialogue_index=self.dialogue_index,total_dialogue=len(self.dialogues),
            story_kind=self.story_kind,
            confirm_kind=self.confirm_kind,confirm_title=title,confirm_text=message,save_error=self.store.error,
            replay=self.replaying_story,settings_return=self.settings_return,home_view=self.home_view,
            total_stars=self.campaign.total_stars(),
            audio_available=self.audio.available,ending_lines=ENDING,
            likes_totals=likes.totals(self.campaign.data['likes']),
            memory_comparison=likes.comparison(self.campaign.data['likes']),
            ending_id=self.ending_id, ending_content=ENDING_CONTENT.get(self.ending_id))

    def draw(self):
        self.audio.configure(self.campaign.data['settings'])
        self.canvas.fill(NIGHT)
        if self.scene == 'game':
            snapshot=self.core.snapshot()
            tick_seconds = float(
                self.core.config.get('tick_seconds', 0.3)
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
            )
            self.controls=[]
        else:
            screen='settings' if self.scene == 'pause' else self.scene
            self.controls=render_menu(self.canvas,self.assets,screen,self.context(),self.focus,self.mouse)
            if self.needs_focus:
                preferred={'result':['view_memory','next_stage','retry'], 'pause':['resume'],
                    'memory_choice':['memory_keep'], 'likes':['home'],
                    'settings':['back'], 'briefing':['start'], 'rooms':[f'stage:{min(12,len(self.campaign.data["completed"])+1)}'],
                    'home':['explore'], 'friends':['back'], 'story':['story_next'], 'ending':['ending_continue'],
                    'tutorial':['tutorial_done'], 'confirm':['confirm_no'], 'save_error':['save_retry'],
                    'album':[f'photo:{self.photo_selected}'], 'appearance':[f'appearance_tab:{self.appearance_kind}']}.get(self.scene,[])
                for ident in preferred:
                    match=next((i for i,b in enumerate(self.controls) if b['id']==ident and not b.get('disabled')),None)
                    if match is not None:
                        self.focus=match
                        break
                self.needs_focus=False
                self.controls=render_menu(self.canvas,self.assets,screen,self.context(),self.focus,self.mouse)
            self.focus=min(self.focus,max(0,len(self.controls)-1))
        # Unsaved progress remains visible after choosing to continue the session.
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

            # 멈췄던 프레임에서는 즉시 입력 실행보다 일시정지가 먼저입니다.
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
