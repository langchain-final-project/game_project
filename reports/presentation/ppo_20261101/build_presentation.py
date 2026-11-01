"""발표 산출물만 생성합니다. 게임 코드·학습 모델은 변경하지 않습니다."""

from pathlib import Path
import hashlib
import html
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

RUN = ROOT / "reports/local/experiments/stage2_magnet_connection_full/quick/seed_11_20261101_032919_437907"
S1 = ROOT / "reports/local/stage1/evaluation/heldout_seeds/evaluation.json"
S2 = RUN / "evaluation/step_451765/evaluation.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gather():
    import torch
    from stable_baselines3 import PPO
    from rl import ppo_config as cfg
    from rl.evaluate_ppo import evaluate_model
    s1, s2 = load(S1), load(S2)
    proof = OUT / "stage2_new_seeds_evidence.json"
    if not proof.exists():
        torch.set_num_threads(1)
        model = PPO.load(RUN / "models/best_model.zip", device="cpu")
        steps = model.num_timesteps
        cfg.EVAL_CONFIG["stochastic_seeds"] = list(range(50000, 50100))
        cfg.EVAL_CONFIG["deterministic_seeds"] = list(range(50000, 50020))
        report = evaluate_model(model)
        assert model.num_timesteps == steps
        report["training_performed"] = False
        report["presentation_evidence"] = True
        report["source_model"] = str(RUN / "models/best_model.zip")
        proof.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    held = load(proof)
    summary = {"stage1_new": s1["results"]["stochastic"]["summary"],
               "stage2_selection": s2["results"]["stochastic"]["summary"],
               "stage2_new": held["results"]["stochastic"]["summary"]}
    (OUT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def screenshots():
    if (OUT / "game_play.png").exists() and (OUT / "game_clear.png").exists():
        return
    import pygame
    from rl.watch_ppo import load_player
    from kkamang.ui.field import draw_game
    from kkamang.ui.theme import Assets
    from kkamang.story import STAGES
    pygame.display.init()
    pygame.font.init()
    surface = pygame.display.set_mode((1280, 720))
    assets = Assets()
    player, _ = load_player(RUN, seed=11, stochastic=False)
    try:
        captured = False
        for _ in range(2000):
            player.advance()
            snapshot = player.info["snapshot"]
            frame = snapshot["motion"]["frame"]
            if not captured and frame >= 180 or player.finished:
                draw_game(surface, assets, snapshot, STAGES[1], elapsed=frame / 60,
                          facing=player.facing, remaining_seconds=(snapshot["remaining_ticks"] - snapshot["motion"]["phase"]) * .3,
                          toast="PPO 정책 재생 · Stage 2 · 결정적 플레이")
                target = OUT / ("game_clear.png" if player.finished else "game_play.png")
                pygame.image.save(surface, str(target))
                captured = True
            if player.finished:
                (OUT / "screenshot_outcome.json").write_text(json.dumps(player.info["outcome"], ensure_ascii=False, indent=2), encoding="utf-8")
                break
        assert player.finished
    finally:
        player.env.close()
        pygame.quit()


STYLE = """
@font-face{font-family:Malgun;src:url('file:///C:/Windows/Fonts/malgun.ttf');font-weight:400}
@font-face{font-family:Malgun;src:url('file:///C:/Windows/Fonts/malgunbd.ttf');font-weight:700}
@page slides{size:1600px 900px;margin:0}
@page notes{size:A4 portrait;margin:0}
*{box-sizing:border-box}html,body{margin:0;padding:0;font-family:Malgun,sans-serif;color:#f5f0e6}
body{-webkit-print-color-adjust:exact;print-color-adjust:exact;background:#151322}
.slide{page:slides;width:1600px;height:900px;padding:44px 72px 40px;position:relative;overflow:hidden;break-after:page;background:#1c192b}
.slide:before{content:'';position:absolute;width:420px;height:420px;border:1px solid #3b334d;border-radius:50%;right:-160px;top:-240px}
.eyebrow{color:#dfbd79;font-size:17px;letter-spacing:2px;font-weight:700;margin:0 0 18px}
.slide h1{font-size:51px;letter-spacing:-2px;margin:0 0 12px;line-height:1.3;font-weight:700}
.subtitle{font-size:23px;color:#c3bacc;margin:0;line-height:1.5}
.footer{position:absolute;bottom:26px;left:72px;right:72px;display:flex;justify-content:space-between;border-top:1px solid #443a50;padding-top:12px;font-size:14px;color:#b5a8bf}
.mainrow{display:grid;grid-template-columns:580px 1fr;gap:44px;margin-top:32px;align-items:start}
.card{background:#292337;border:1px solid #483b55;border-radius:20px;padding:28px}
.big{font-size:100px;line-height:1;color:#e9c382;font-weight:700;letter-spacing:-6px}.big small{font-size:26px;letter-spacing:-1px;color:#f5f0e6;margin-left:18px}
.statrows{margin-top:18px}.statrow{display:flex;justify-content:space-between;border-bottom:1px solid #483d54;padding:12px 0;font-size:24px}
.statrow strong{color:#a8c8b0}.muted{color:#c7b9d0}.fine{font-size:17px;line-height:1.65;color:#c5b9cf;margin:18px 0 0}
.game{width:100%;height:auto;border-radius:16px;border:1px solid #55445d;display:block}.caption{font-size:16px;line-height:1.6;color:#bcaec5;margin:12px 0 0}
.flow{display:flex;align-items:center;gap:14px;margin-top:27px}.flow .node{flex:1;text-align:center;padding:20px 8px;border-radius:13px;background:#352a42;border:1px solid #5a4667;font-size:22px;font-weight:700}
.flow .arrow{font-size:25px;color:#dfbd79}.flow .node span{display:block;font-size:15px;color:#cbbbd4;margin-top:5px;font-weight:400}
.cards3{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:37px}.cards3 .card{height:377px;padding:28px}
.kicker{font-size:16px;letter-spacing:2px;color:#a8c8b0;margin:0 0 15px}.card h2{font-size:32px;line-height:1.4;margin:0 0 22px}.card p{font-size:22px;line-height:1.75;margin:0 0 12px}.gold{color:#e9c382}.sage{color:#a8c8b0}.pink{color:#f0b1b4}
.route{margin-top:29px;background:#efe5ce;border-radius:16px;padding:24px;color:#382c43;font-size:23px;text-align:center;font-weight:700}.route .sep{padding:0 20px;color:#9e7551}
.s2note{font-size:18px;color:#ccbed5;margin:20px 0}.results{display:grid;grid-template-columns:785px 1fr;gap:32px;margin-top:34px}.results .card{padding:24px}
table{border-collapse:collapse;width:100%;font-size:22px}th{text-align:right;color:#c6b7d0;font-size:18px;padding:14px 10px;border-bottom:1px solid #685271}td{text-align:right;padding:20px 10px;border-bottom:1px solid #4c3c5b;font-weight:700}th:first-child,td:first-child{text-align:left}td:first-child{font-size:20px}.selected td{color:#e9c382}.new td{color:#acd0b4}
.table-note{font-size:16px;color:#c2b4cd;line-height:1.75;margin:15px 0 0}.progress{margin-top:20px;font-size:21px}.bars{margin-top:10px;display:flex;gap:12px;align-items:center}.barbg{background:#4a3a56;border-radius:9px;height:15px;flex:1}.bar{background:#b39ac7;height:15px;border-radius:9px}.bar.goldbar{background:#e9c382}.label{min-width:75px;font-size:18px}.v{width:50px;text-align:right;font-size:20px;font-weight:700}
.callout{border-left:4px solid #e9c382;padding:13px 18px;background:#35283e;border-radius:0 12px 12px 0;margin-top:20px;font-size:21px;line-height:1.7}.bottom-note{font-size:17px;line-height:1.75;color:#c7b8d0;margin-top:22px}.results .game{margin-top:0}
.note{page:notes;width:794px;height:1122px;background:#fffdf8;color:#30283a;padding:52px 54px 44px;position:relative;overflow:hidden;break-after:page;font-size:14px;line-height:1.7}
.note .eyebrow{font-size:11px;color:#906d3c;margin-bottom:12px}.note h1{font-size:29px;line-height:1.35;margin:0 0 11px;letter-spacing:-1px}.note h2{font-size:17px;color:#563d67;margin:23px 0 8px}.note p{margin:8px 0}.note ul{padding-left:19px;margin:8px 0}.note li{margin:5px 0}.note .script{background:#f0e9f4;border-left:4px solid #8d6aa3;padding:17px 19px;font-size:14px;line-height:1.9;border-radius:0 10px 10px 0}.note .mini{font-size:12px;color:#6c5c75;line-height:1.7}.note table{font-size:12px;margin-top:10px}.note th{color:#594463;font-size:12px;background:#f0e9f4;text-align:left;padding:8px}.note td{font-weight:400;text-align:left;padding:8px;border-bottom:1px solid #e2d9e6;font-size:12px}.note td:first-child{font-size:12px}.note code{font-family:Malgun,sans-serif;font-size:11px;background:#eee8f0;padding:2px 4px;overflow-wrap:anywhere}.note a{color:#68517c}.notef{position:absolute;bottom:28px;left:54px;right:54px;display:flex;justify-content:space-between;border-top:1px solid #ded5e2;padding-top:9px;font-size:10px;color:#75677e}.note .qa p{font-size:13px;line-height:1.7;margin:5px 0 12px}.note .qa strong{color:#4e365f}.note .box{background:#f5f0e6;padding:12px 15px;border-radius:9px;font-size:13px}
@media screen{.slide,.note{margin:0 auto 20px}.preview .slide{margin:0}.preview{overflow:hidden}}
"""


STYLE += """
.cards3 .card{height:400px;padding:26px}
.cards3 .kicker{margin-bottom:12px}
.cards3 h2{font-size:30px;line-height:1.35;margin-bottom:18px}
.cards3 .card p{font-size:21px;line-height:1.6}
.cards3 .card p.fine{font-size:16px;line-height:1.5;margin-top:12px}
.note tr.selected td{color:#80551d}.note tr.new td{color:#326749}
.note .mini{overflow-wrap:anywhere}
"""


def foot(n):
    return f'<div class="footer"><span>까망이와 유령 저택 · PPO 적용</span><span>2026.11.01 &nbsp; / &nbsp; {n:02d}</span></div>'


def notefoot(n):
    return f'<div class="notef"><span>까망이와 유령 저택 · 발표 설명 및 근거</span><span>설명 {n} / 4</span></div>'


def build(metrics):
    def pct(n):
        return f'{100*n:.0f}%'
    rows = ''
    for key, title, cls in [('stage1_new','Stage 1 · 새 시드',''),('stage2_selection','Stage 2 · 선택 평가','selected'),('stage2_new','Stage 2 · 새 시드','new')]:
        s=metrics[key]
        rows += f'<tr class="{cls}"><td>{title}</td><td>{pct(s["success_rate"])}</td><td>{pct(s["three_star_rate"])}</td><td>{s["mean_stars"]:.2f}</td></tr>'
    s2new=metrics['stage2_new']
    slides = [f'''
<section class="slide"><p class="eyebrow">01 &nbsp; GAME × REINFORCEMENT LEARNING</p>
<h1>게임 상태를 읽고, 행동을 배우는 PPO</h1><p class="subtitle">실시간 탐사 환경에 상태 관측 · 행동 선택 · 보상 피드백을 연결했습니다.</p>
<div class="mainrow"><div class="card"><div class="big">640<small>차원 관측</small></div>
<div class="statrows"><div class="statrow"><span>지형 · 아이템 · 유령 · 시간</span><strong>630</strong></div><div class="statrow"><span>실제 이동 가능 방향</span><strong>4</strong></div><div class="statrow"><span>목표 단계 · 방향 · 온기 조건</span><strong>6</strong></div></div>
<p class="fine">환경과 목표는 별도 신경망 분기로 처리합니다.<br>이동 정보는 행동 마스크에 사용합니다.</p></div>
<div><img class="game" src="game_play.png"><p class="caption">실제 RealtimeMansionEnv · Stage 2 정책 재생 화면<br>행동: 상·하·좌·우·대기 / 한 번 선택한 행동을 최대 9프레임 유지</p></div></div>
<div class="flow"><div class="node">게임 상태<span>수치 관측</span></div><div class="arrow">→</div><div class="node">PPO 정책<span>행동 확률 계산</span></div><div class="arrow">→</div><div class="node">이동 · 대기<span>환경과 상호작용</span></div><div class="arrow">→</div><div class="node">보상 · 경험<span>정책 / 가치 갱신</span></div></div>{foot(1)}</section>''',f'''
<section class="slide"><p class="eyebrow">02 &nbsp; LEARNING DESIGN</p><h1>별 3개를 향해, 부분 행동을 연결하다</h1><p class="subtitle">성공 경험이 부족한 문제를 보상 설계와 단계별 복습으로 풀었습니다.</p>
<div class="cards3"><div class="card"><div class="kicker">REWARD</div><h2>무엇을 잘했다고<br>알려줄 것인가?</h2><p><span class="gold">수집 · 목표 접근 · 별 획득</span>에<br>긍정적인 피드백</p><p><span class="pink">시간 낭비 · 벽막힘 · 피해</span>에는<br>비용 부여</p><p class="fine">게임 점수와 학습 보상은 구분</p></div>
<div class="card"><div class="kicker">CURRICULUM</div><h2>쉬운 과제부터<br>전체 플레이로</h2><p>물건 획득을 짧게 연습한 뒤<br>자석 · 물건 연결 과제로 확장</p><p>기존 행동을 함께 복습하며<br><span class="gold">추가 수집 · 귀환까지 연결</span></p><p class="fine">중간 시작 상태와 실제 시작점 평가를 구분</p></div>
<div class="card"><div class="kicker">PPO / ACTOR–CRITIC</div><h2>경험으로 배우고<br>작게 갱신하기</h2><p>정책망: 다음 행동을 선택<br>가치망: 앞으로의 보상을 예측</p><p><span class="gold">Clipping</span>으로 한 번의 갱신에서<br>정책이 크게 바뀌는 것을 제한</p><p class="fine">Stable-Baselines3 PPO에 게임용 정책 연결</p></div></div>
<div class="route">초반 온기 · 자석<span class="sep">→</span>물건 획득<span class="sep">→</span>추가 온기<span class="sep">→</span>출구 귀환</div>
<p class="s2note">코드가 목표 순서와 경유지를 안내하고, 실제 이동·대기 행동은 PPO 정책이 선택합니다.</p>{foot(2)}</section>''',f'''
<section class="slide"><p class="eyebrow">03 &nbsp; EVALUATION & NEXT STEP</p><h1>실제 시작점에서 확인한 학습 성과</h1><p class="subtitle">확률적 플레이로 각 100판 평가 · 실패한 판도 별 0개로 평균에 포함</p>
<div class="results"><div class="card"><table><thead><tr><th>평가 대상</th><th>성공</th><th>별 3개</th><th>평균 별</th></tr></thead><tbody>{rows}</tbody></table>
<p class="table-note">Stage 1: 보관 모델의 새 시드 평가<br>Stage 2: 전체 학습 best_model의 선택 평가와 별도 새 시드 검증</p>
<div class="progress">Stage 2 전체 학습의 별 3개 획득률</div><div class="bars"><span class="label">학습 전</span><div class="barbg"><div class="bar" style="width:2%"></div></div><span class="v">2%</span></div><div class="bars"><span class="label">선택 모델</span><div class="barbg"><div class="bar goldbar" style="width:82%"></div></div><span class="v gold">82%</span></div>
<p class="table-note">위 두 값은 같은 선택 평가 시드 기준입니다.</p></div>
<div><img class="game" src="game_clear.png"><p class="caption">결정적 플레이 화면 예시 · 95점 · 별 3개 · 피해 1회<br>위 표의 통계는 확률적 플레이 결과입니다.</p>
<div class="callout">새 시드 자석 활용 <strong class="gold">{pct(s2new['magnet_use_rate'])}</strong><br>평균 피해 <strong class="pink">{s2new['mean_damage']:.2f}회</strong> → 유령 회피 보완</div></div></div>
<p class="bottom-note">검증한 성과: 자석 · 물건 · 추가 수집 · 귀환 연결 &nbsp; / &nbsp; 진행 중: 무피해 귀환 안정화<br>마무리 학습의 미확인 결과는 성과 수치에 포함하지 않았습니다.</p>{foot(3)}</section>''']

    notes=[f'''
<section class="note"><p class="eyebrow">PRESENTER NOTES · 01</p><h1>1장 설명 · 게임과 PPO의 연결</h1><p class="mini">발표용 PDF는 3장입니다. 이 뒤의 4쪽은 발표 준비를 위한 대본·상세 설명·근거입니다.</p>
<h2>그대로 읽을 수 있는 발표 대본</h2><div class="script">“까망이가 온기와 물건을 모아 귀환하는 행동을 PPO 강화학습으로 학습시켰습니다. 게임 내부 상태를 640개 숫자로 전달하고, AI가 네 방향 이동과 대기 중 다음 행동을 선택합니다. 환경 정보와 목표 정보를 별도 신경망 분기로 처리해 현재 상황과 가야 할 방향을 구분하도록 했습니다. 이동할 수 없는 방향과 효과 없는 상호작용은 마스크로 제외합니다. 선택한 행동을 최대 9프레임 유지한 뒤 다음 상태를 관찰하고, 플레이 경험과 보상을 이용해 정책을 갱신합니다.”</div>
<h2>640개 관측에는 무엇이 들어 있나?</h2><table><thead><tr><th>구성</th><th>수</th><th>정보</th></tr></thead><tbody><tr><td>실시간 환경</td><td>630</td><td>벽·출구, 캐릭터 위치·용기·보호·온기, 시간, 아이템, 유령 경로·상태·실제 위치, 물건·자석 상태 등</td></tr><tr><td>이동 가능 여부</td><td>4</td><td>실제 몸체 충돌과 닫힌 문을 반영한 네 방향 이동 가능 정보</td></tr><tr><td>목표 정보</td><td>6</td><td>물건·수집·귀환 단계 3개, 목표 상대 방향 2개, 기본 온기 조건 충족 여부 1개</td></tr></tbody></table>
<h2>신경망과 행동 마스크</h2><p>환경 630개는 32개 특징으로, 목표 6개도 32개 특징으로 변환합니다. 합친 64개 특징에서 정책망과 가치망으로 이어집니다. 각 망은 64개 노드의 은닉층 두 개를 사용합니다. 이동 정보 4개는 특징 분기에 섞지 않고 행동 마스크에 사용합니다.</p><p>행동 공간의 원래 슬롯은 6개지만, 현재 PPO 정책은 효과 없는 상호작용을 제외합니다. 발표에서는 “네 방향 이동과 대기”라고 설명하는 것이 정확합니다. 유령 위험 방향을 자동 차단하는 마스크는 사용하지 않습니다.</p>
<h2>프레임과 학습량을 설명할 때</h2><p>1틱은 18물리 프레임이고 행동 반복은 최대 9프레임입니다. 100만 물리 프레임은 약 11만 번의 행동 선택에 해당하며, 종료 프레임에 따라 달라집니다. 물리 프레임, 행동 선택 수, 게임 판수는 서로 다른 단위입니다.</p>
<p class="mini">구현 근거: kkamang/env.py · rl/ppo_observation.py · rl/ppo_features.py · rl/ppo_policy.py · rl/ppo_action_repeat.py</p>{notefoot(1)}</section>''',f'''
<section class="note"><p class="eyebrow">PRESENTER NOTES · 02</p><h1>2장 설명 · 어떻게 배우게 했나?</h1>
<h2>발표 대본</h2><div class="script">“처음부터 전체 게임을 반복하는 방식만으로는 성공 경험이 부족했습니다. 그래서 물건 획득처럼 짧은 과제를 먼저 연습하고, 초반 온기와 자석 수집을 함께 연결했습니다. 한 행동을 새로 익히면서 기존 행동을 잊지 않도록 복습 구간도 섞었습니다. 온기와 물건 획득, 목표 접근, 높은 점수와 별에는 보상을 주고, 시간 낭비와 벽막힘, 유령 피해에는 비용을 줍니다. PPO는 정책망과 가치망을 함께 학습하며, clipping으로 정책 갱신 폭을 제한합니다.”</div>
<h2>PPO를 선택한 이유와 실제 구현</h2><p>우리 게임은 다음 행동을 확률로 선택하는 이산 행동 문제입니다. Stable-Baselines3의 PPO를 기반으로 게임용 특징 추출기와 행동 마스크를 연결했습니다. Actor는 행동 확률을 계산하고 Critic은 상태의 향후 보상을 추정합니다. 수집한 경험을 미니배치로 나누어 반복 학습합니다.</p><p>Clipping은 새 정책과 이전 정책의 행동 확률 비율을 제한하는 목적함수 구성입니다. “성공을 보장한다” 또는 “절대 성능이 떨어지지 않는다”는 의미는 아닙니다. 다른 알고리즘보다 우수하다는 비교 실험을 수행한 것으로 발표하지 않습니다.</p>
<h2>보상과 게임 점수의 차이</h2><p>게임의 별은 온기·속도·남은 용기 점수를 합쳐 85점 이상일 때 3개입니다. 학습 보상은 그 목표를 배우도록 만든 별도의 피드백입니다. 기본 클리어 온기는 4점이며, 학습 안내는 추가 수집을 위해 14점을 목표로 합니다. 시간·용기 여유가 부족하면 기본 조건 확보와 귀환을 우선하도록 안내합니다.</p>
<table><thead><tr><th>학습 피드백</th><th>역할</th></tr></thead><tbody><tr><td>온기·물건·자석 획득</td><td>필수 수집 행동과 아이템 활용을 장려</td></tr><tr><td>목표 거리 감소 / 증가</td><td>올바른 방향에 빠른 피드백. 경유지·벽을 고려해 계산</td></tr><tr><td>최종 성공·점수·별</td><td>전체 과제와 고득점 귀환을 연결</td></tr><tr><td>시간·벽막힘·정체·피해 비용</td><td>무의미한 행동과 위험한 이동을 줄이는 신호</td></tr></tbody></table>
<h2>커리큘럼과 자석 안내</h2><p>쉬운 중간 상태에서 시작하는 복습과 실제 시작점 플레이를 혼합합니다. 중간 상태는 실제 게임 이동으로 준비하며 그동안 경과한 게임 시간·획득 상태를 유지합니다. 자석은 1점 온기를 주변 3×3 범위에서 흡수하고, 3점 온기는 직접 밟아야 합니다. 목표 안내에도 이 규칙과 남은 효과 시간을 반영했습니다.</p>
<p class="mini">PPO 근거: <a href="https://arxiv.org/abs/1707.06347">Schulman et al., 2017</a> · <a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html">Stable-Baselines3 공식 PPO 문서</a><br>구현: rl/train_ppo.py · rl/ppo_curriculum.py · rl/ppo_goal.py · rl/ppo_reward.py</p>{notefoot(2)}</section>''',f'''
<section class="note"><p class="eyebrow">PRESENTER NOTES · 03</p><h1>3장 설명 · 성과를 어떻게 해석하나?</h1>
<h2>발표 대본</h2><div class="script">“평가는 준비된 중간 상태가 아닌 실제 시작점에서 진행했습니다. Stage 1 보관 모델은 새 시드 100판에서 성공률 96%, 별 3개 획득률 95%였습니다. Stage 2는 모델 선택 평가에서 성공률 96%, 별 3개 82%였고, 별도 새 시드 100판에서도 성공률 97%, 별 3개 78%를 확인했습니다. 자석 활용은 98%로 높았지만 평균 피해는 1.17회여서 유령 회피가 남은 과제입니다. 현재 무피해 추가 수집·귀환을 보강하고 있으며, 미확인 결과는 성과에서 제외했습니다.”</div>
<h2>표에 사용한 실제 평가 수치</h2><table><thead><tr><th>평가</th><th>성공</th><th>별 3개</th><th>평균 별</th></tr></thead><tbody>{rows}</tbody></table>
<p>모두 실제 시작점의 확률적 100판입니다. Stage 1은 보관한 새 시드 평가, Stage 2 선택 평가는 최적 모델을 고르는 데 사용한 평가입니다. Stage 2 새 시드 평가는 선택 이후 다른 행동 난수 시드로 확인한 결과입니다. 새 맵에 대한 일반화 성능을 뜻하지 않습니다.</p>
<h2>결정적 플레이와 확률적 플레이</h2><p>결정적 플레이는 확률이 가장 높은 행동을 선택합니다. 확률적 플레이는 정책이 출력한 분포에서 행동을 뽑습니다. 최종 사용 방식에 맞춰 확률적 평가를 주 기준으로 사용했고, 결정적 플레이와 유령 초기 위상별 평가는 보조 지표로 기록했습니다.</p><p>Stage 2 결정적 화면 예시는 95점·별 3개·피해 1회입니다. 그 한 장면의 성공을 확률적 성공률 100%로 표현하면 안 됩니다. 같은 고정 환경에서 다른 시드의 결정적 재생이 같은 경로를 반복할 수도 있습니다.</p>
<h2>이미 확인한 성과와 현재 목표</h2><p>확인한 성과는 자석·물건·추가 수집·귀환의 연결입니다. 무피해 플레이의 안정성은 남은 과제입니다. 현재 마무리 설정은 성공 90%, 별 3개 90%, 평균 별 2.7개 이상과 평균 피해 0.1회 이하를 목표로 합니다. 학습 종료 후 선택한 모델 하나를 새 시드 200판으로 검증하도록 준비했습니다.</p>
<div class="box"><strong>발표에서 구분할 것</strong><br>부분 과제의 True ≠ 전체 게임 별 3개 성공<br>학습 중 성공률 ≠ 실제 시작점 평가 성공률<br>목표로 정한 기준 ≠ 이미 달성한 성과</div>
<p class="mini">발표 시점의 성과는 위 표로 고정했습니다. 진행 중인 마무리 실험의 최종 보고서가 확인되면 별도 근거와 함께 갱신해야 합니다.</p>{notefoot(3)}</section>''',f'''
<section class="note"><p class="eyebrow">REFERENCE · Q&A</p><h1>예상 질문과 기술 근거</h1>
<div class="qa"><p><strong>Q. 경로를 전부 하드코딩한 것 아닌가요?</strong><br>A. 목표 순서와 최단 경로의 경유지는 코드가 안내합니다. 준비 경로는 커리큘럼 시작 상태를 만드는 데 사용합니다. 실제 이동·대기 행동과 행동 확률은 PPO 정책이 선택합니다. 따라서 목표 안내를 포함한 강화학습 구조라고 설명합니다.</p>
<p><strong>Q. 게임 화면을 보고 학습하나요?</strong><br>A. 현재 모델은 픽셀 이미지 대신 게임 내부 상태를 숫자로 입력받습니다. 학습된 정책의 행동을 기존 게임 렌더러로 재생합니다.</p>
<p><strong>Q. 자석을 배웠다는 근거는 무엇인가요?</strong><br>A. 자석을 먹었는지와 실제 원격 수집에 사용했는지를 구분했습니다. Stage 2 새 시드 100판의 실제 활용률은 {pct(s2new['magnet_use_rate'])}입니다.</p>
<p><strong>Q. Stage 1과 Stage 2 모델을 합쳤나요?</strong><br>A. 기존 가중치를 다음 학습의 출발점으로 가져와 미세 조정했습니다. 여러 ZIP의 행동을 합산하는 방식은 아닙니다. 현재 실험은 새 최적화기로 이어 학습합니다.</p>
<p><strong>Q. 병렬 환경 4개면 CPU 코어도 4개인가요?</strong><br>A. 현재는 DummyVecEnv 4개와 Torch 스레드 1개 설정입니다. 환경 4개에서 경험을 모은다는 뜻이며, 4개 프로세스로 동시에 실행한다는 뜻은 아닙니다.</p></div>
<h2>성과 모델과 마무리 설정을 혼동하지 않기</h2><table><thead><tr><th>항목</th><th>표의 Stage 2 모델</th><th>현재 무피해 마무리 설정</th></tr></thead><tbody><tr><td>학습률 / PPO clip</td><td>0.00005 / 0.2</td><td>0.00002 / 0.1</td></tr><tr><td>반복 / 엔트로피</td><td>10회 / 0.01</td><td>10회 / 0.003</td></tr><tr><td>학습 예산</td><td>약 50만 물리 프레임</td><td>추가 최대 약 100만 물리 프레임</td></tr><tr><td>피해 / 무피해 보상</td><td>-1 / 완료 시 +2</td><td>-6 / 무피해 별 3개 완료 시 +8</td></tr></tbody></table>
<h2>프로젝트 내 근거 파일</h2><p class="mini">① Stage 1: reports/local/stage1/evaluation/heldout_seeds/evaluation.json<br>② Stage 2 선택: stage2_magnet_connection_full/quick/<br>seed_11_20261101_032919_437907/evaluation/step_451765/evaluation.json<br>②의 경로 앞에는 reports/local/experiments/를 붙입니다.<br>③ Stage 2 새 시드: 이 산출물 폴더의 stage2_new_seeds_evidence.json<br>④ 현재 설정: rl/ppo_config.py · 최종 절차: rl/PPO_STAGE2_SAFE_RETURN_FINAL.md</p>
<h2>공식 문헌</h2><p class="mini"><a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms — Schulman et al. (2017)</a><br><a href="https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html">Stable-Baselines3: PPO 공식 문서</a></p>
<p class="mini">제작 기준일: 2026-11-01 (Asia/Seoul). Canva 편집 화면에 접근할 수 없어, 실제 게임 화면과 코드·평가 기록으로 독립 구성했습니다. 학습과 게임 소스 편집은 수행하지 않았습니다.</p>{notefoot(4)}</section>''']
    def document(parts, preview=False):
        return '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>까망이와 유령 저택 · PPO 적용</title><style>'+STYLE+'</style></head><body'+(' class="preview"' if preview else '')+'>'+''.join(parts)+'</body></html>'
    (OUT/'ppo_slides.html').write_text(document(slides),encoding='utf-8')
    (OUT/'ppo_complete.html').write_text(document(slides+notes),encoding='utf-8')
    for n,slide in enumerate(slides,1):
        (OUT/f'preview_{n:02}.html').write_text(document([slide],True),encoding='utf-8')
    for n,note in enumerate(notes,1):
        (OUT/f'note_preview_{n:02}.html').write_text(document([note],True),encoding='utf-8')


def browser_render():
    browser=Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
    if not browser.exists():
        browser=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
    common=[str(browser),'--headless','--disable-gpu','--no-first-run','--no-default-browser-check',
            '--allow-file-access-from-files','--no-pdf-header-footer', '--hide-scrollbars',
            '--user-data-dir='+str(OUT/'browser_profile')]
    def run(extra, file):
        p=subprocess.run(common+extra+[file.as_uri()],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=90)
        (OUT/'browser_render.log').write_bytes(p.stderr)
        return p
    for source,target in [('ppo_slides.html','PPO_발표자료_3장.pdf'),('ppo_complete.html','PPO_발표자료와_상세설명_전체.pdf')]:
        path=OUT/target
        run(['--print-to-pdf='+str(path),'--virtual-time-budget=4000'],OUT/source)
        assert path.exists() and path.stat().st_size>30000, target
        count=len(re.findall(rb'/Type\s*/Page\b',path.read_bytes()))
        print(target,'pages',count,'bytes',path.stat().st_size,flush=True)
    for n in range(1,4):
        run(['--screenshot='+str(OUT/f'slide_{n:02}.png'),'--window-size=1600,900','--virtual-time-budget=1500'],OUT/f'preview_{n:02}.html')
    for n in range(1,5):
        run(['--screenshot='+str(OUT/f'note_{n:02}.png'),'--window-size=794,1122','--virtual-time-budget=1500'],OUT/f'note_preview_{n:02}.html')


if __name__=='__main__':
    source=RUN/'models/best_model.zip'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    print('Gathering verified metrics; no PPO training.',flush=True)
    metrics=gather()
    print('Rendering real game screenshots.',flush=True)
    screenshots()
    build(metrics)
    print('Printing presentation and full explanation PDFs.',flush=True)
    browser_render()
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps({'training_performed':False,'source_model_unchanged':True,
         'source_sha256':before,'presentation_pages':3,'explanation_pages':4,'metrics':metrics},ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',str(OUT),flush=True)
