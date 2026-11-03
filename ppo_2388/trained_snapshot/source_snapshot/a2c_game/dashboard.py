"""표준 라이브러리만 사용하는 읽기 전용 로컬 학습 대시보드."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from .display_units import load_training_monitor, settings_groups


def read_json(path, default=None):
    """학습기의 원자적 교체 또는 쓰기 중인 파일 때문에 화면을 중단하지 않습니다."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, OSError, ValueError):
        return {} if default is None else default


def read_jsonl(path, *, max_bytes=16_000_000, max_rows=20_000):
    """완성된 최근 레코드만 읽습니다. 잘린 마지막 줄은 다음 갱신 때 읽습니다."""
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - max_bytes))
            if handle.tell():
                handle.readline()
            lines = handle.read().splitlines()
    except OSError:
        return []
    result = []
    for line in lines[-max_rows:]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return result


def dashboard_state(run):
    """설정·학습 로그·평가·재생 경로를 묶어 브라우저에 전달합니다."""
    run = Path(run)
    history = read_jsonl(run / 'evaluation/history.jsonl')
    known = {str(row['replay']) for row in history if row.get('replay')}
    replays = []
    folder = run / 'replays'
    if folder.is_dir():
        for path in sorted(folder.rglob('*.json'))[-200:]:
            try:
                relative = str(path.relative_to(run))
            except ValueError:
                continue
            if relative not in known:
                replays.append(relative)
    config = read_json(run / 'config.json')
    metadata = read_json(run / 'metadata.json')
    metadata['progress_guard'] = (read_json(run / 'fresh_protocol.json')
                                  or read_json(run / 'all_stage_protocol.json')
                                  or read_json(run / 'progress_guard.json'))
    groups = (settings_groups(config, metadata=metadata)
              if all(key in config for key in ('a2c', 'environment', 'reward',
                                                'training', 'evaluation', 'policy')) else [])
    selected = read_json(run / 'selected_evaluation.json')
    monitor = load_training_monitor(run)
    monitor['guard'] = {'status': metadata.get('status')}
    if selected:
        rows = [{**row, 'stage': int(stage), 'mode': mode, 'steps': selected['steps']}
                for mode, stages in selected['metrics'].items() for stage, row in stages.items()]
        monitor['evaluation'] = {'steps': selected['steps'], 'complete': True, 'pending': False, 'rows': rows}
        for row in monitor['stages']:
            for mode in ('greedy', 'stochastic'):
                row[mode] = selected['metrics'][mode][str(row['stage'])]
    return {
        'selected_model': selected,
        'final_results': read_json(run / 'results.json'),
        'status': read_json(run / 'status.json'),
        'progress': read_jsonl(run / 'train/progress.jsonl'),
        'evaluation': history,
        'exposure': read_json(run / 'stageexposure.json'),
        'extra_replays': replays,
        'monitor': monitor,
        'settings': groups,
        'experiment': {'input_dimensions': 2388, 'initialization': metadata.get('initialization'),
                       'run_name': run.name, 'fresh': True},
        'references': read_json(Path(__file__).resolve().parents[1] / 'references/existing_models.json').get('models', []),
    }


HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PPO Navigation 2,388 · 새 공통 모델</title><style>
:root{color-scheme:light;--ink:#172437;--muted:#66758a;--line:#e1e7ef;--blue:#2962e7;--green:#13845a;--gold:#c67912;--purple:#8750c5}*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:#fff;border-bottom:1px solid var(--line);padding:22px max(24px,calc((100vw - 1450px)/2))}header>div{display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:23px;margin:0 0 5px;letter-spacing:-.7px}h2{font-size:16px;margin:0 0 12px}p{margin:0}.muted{color:var(--muted)}.badge{border-radius:20px;padding:5px 12px;background:#e7efff;color:var(--blue);white-space:nowrap}main{max-width:1498px;padding:22px 24px 40px;margin:auto}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}.metric,.panel{border:1px solid var(--line);border-radius:12px;background:#fff;padding:16px}.metric strong{display:block;font-size:24px;letter-spacing:-.5px;margin-top:6px}.layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,.85fr);gap:18px}.panel{margin-bottom:18px}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}button,select,input{font:inherit}button,select{border:1px solid #cad4e3;border-radius:7px;background:white;padding:7px 10px;color:var(--ink)}button{cursor:pointer}button.active{background:#e9efff;border-color:var(--blue);color:var(--blue)}button:hover{border-color:var(--blue)}select{max-width:100%}.plot{height:205px;width:100%;display:block}.legend{display:flex;gap:14px;font-size:12px;color:var(--muted);flex-wrap:wrap}.legend i{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:4px}.game{width:100%;display:block;background:#edf0f6;border-radius:8px}.gamehud{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;font-variant-numeric:tabular-nums}.replaylabel{min-height:38px;margin-bottom:10px;font-size:12px;color:var(--muted);overflow-wrap:anywhere}.scrub{width:100%;accent-color:var(--blue)}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:9px 7px;text-align:right;border-bottom:1px solid #edf0f5;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected{background:#edf3ff}.empty{color:#8a96a7}.success{color:var(--green)}.warning{color:var(--gold)}.error{color:#b53636}.notice{font-size:12px;color:var(--muted);padding-top:10px}.checkpoints{overflow-wrap:anywhere;font-size:12px}.bar{height:7px;border-radius:8px;background:#edf0f6;overflow:hidden;margin-top:10px}.bar span{height:100%;display:block;background:var(--blue)}.tablewrap{overflow-x:auto}.statusline{margin-top:7px;font-size:12px;color:var(--muted)}@media(max-width:1000px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:600px){main{padding:14px}header{padding:18px 14px}.metrics{grid-template-columns:repeat(2,1fr)}.metric strong{font-size:19px}}
</style></head><body>
<header><div><div><h1>PPO 2,388 · 새 공통 모델</h1><p class="muted">원본과 같은 전12맵 환경 · 임의 가중치에서 새 학습 · 학습 곡선과 최종 검증</p></div><span id="status" class="badge">기록 연결 중</span></div><div class="statusline" id="runName"></div></header>
<main><div class="metrics"><div class="metric"><span class="muted">학습 행동</span><strong id="steps">—</strong><div class="bar"><span id="budgetBar"></span></div></div><div class="metric"><span class="muted">PPO Adam 업데이트 / epochs</span><strong id="updates">—</strong></div><div class="metric"><span class="muted">경험 중인 스테이지</span><strong id="pool">—</strong></div><div class="metric"><span class="muted">선택 Stage 평균 점수</span><strong id="score">—</strong></div><div class="metric"><span class="muted">선택 Stage 85점 귀환</span><strong id="highscore">—</strong></div></div>
<div class="layout"><section>
<div class="panel"><div class="toolbar"><h2 style="margin:0">스테이지별 실제 평가</h2><select id="stage"></select><select id="mode"><option value="greedy">최대확률 실행</option><option value="stochastic">확률 실행</option></select></div><canvas id="scorePlot" class="plot"></canvas><div class="legend"><span><i style="background:#2962e7"></i>실제 평균 점수</span><span><i style="background:#8750c5"></i>85점 기준</span></div><canvas id="ratePlot" class="plot"></canvas><div class="legend"><span><i style="background:#13845a"></i>귀환</span><span><i style="background:#c67912"></i>45점 이상</span><span><i style="background:#2962e7"></i>65점 이상</span><span><i style="background:#8750c5"></i>85점 이상</span></div><p class="notice">가로축은 4개 환경을 합친 학습 행동 수입니다. 실제 평가값을 모두 표시하며, 성능이 항상 상승한다고 가정하지 않습니다.</p></div>
<div class="panel"><h2>학습 중 탐색과 처리량</h2><canvas id="entropyPlot" class="plot"></canvas><div class="legend"><span><i style="background:#c67912"></i>행동 엔트로피</span><span id="fps" class="muted"></span></div><p class="notice">엔트로피가 낮으면 정책이 소수 행동을 강하게 선호합니다. 학습 보상과 게임의 실제 점수는 구분해 평가합니다.</p></div>
<div class="panel"><h2>학습 과정과 실제 변경</h2><div class="tablewrap"><table><tbody id="monitorRows"></tbody></table></div><p class="notice">평가 점수와 학습 보상은 별개입니다. 기록된 조정만 표시합니다.</p><div id="changeHistory" class="checkpoints muted"></div></div>
<div class="panel"><details><summary style="cursor:pointer;font-weight:600">전체 학습·환경·보상·차원 설정</summary><div id="allSettings"></div></details></div>
<div class="panel"><h2>최종 검증 결과</h2><div class="tablewrap"><table><thead><tr><th>모델</th><th>입력</th><th>학습량</th><th>기준 통과 맵</th><th>선택 Stage 평균</th><th>선택 Stage 85점률</th></tr></thead><tbody id="comparisonTable"></tbody></table></div><p class="notice">동일 선정 모델의 별도 두 seed 검증입니다. G는 맵당1회, S는 맵당40회입니다. 전맵90% 기준은9·12번에서 미달했습니다. 아래 곡선은 전체 학습 과정, 성능표와 재생은 선정 모델 기준입니다.</p></div><div class="panel"><h2>현재 체크포인트</h2><div id="checkpoint" class="checkpoints muted">저장을 기다리고 있습니다.</div><div id="bestCheckpoint" class="checkpoints muted" style="margin-top:8px"></div></div>
</section><section>
<div class="panel"><h2>실제 플레이</h2><div class="toolbar"><button id="liveButton" hidden>실시간 맵</button><label hidden>환경 <select id="liveEnv"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label><button id="replayButton" class="active">선정 모델 평가 재생</button></div><div id="replayTools"><select id="replaySelect" style="width:100%;margin-bottom:10px"></select><div class="toolbar"><button id="play">일시정지</button><button id="restart">처음부터</button><label>속도 <select id="speed"><option value="3">느리게</option><option value="6" selected>보통</option><option value="12">빠르게</option></select></label></div><input id="scrub" class="scrub" type="range" min="0" max="0" value="0"></div><div id="playLabel" class="replaylabel">실제 플레이 기록을 기다리고 있습니다.</div><canvas id="game" class="game"></canvas><div id="gameHud" class="gamehud"></div><p class="notice">검정: 까망이 · 금색: 온기(+3은 숫자 표시) · 파랑: 퀘스트 · 초록: 귀환점 · 보라/빨강: 유령. 학습 중 화면은 주기적으로 갱신한 실제 상태입니다. 평가 재생은 저장된 실제 행동 전체입니다.</p></div>
<div class="panel"><h2>선정 시 개발평가 · Stage1–12 G1/S20</h2><div class="tablewrap"><table><thead><tr><th>Stage</th><th>학습 스텝</th><th>학습 평균</th><th>G 점수</th><th>G 85점</th><th>S 점수</th><th>S 귀환</th><th>S 85점</th><th>퀘스트</th><th>온기</th><th>실패 사유</th></tr></thead><tbody id="stageTable"></tbody></table></div><p class="notice">G=최대확률, S=확률 행동. 완료된 동일 체크포인트의 12맵 평가를 표시합니다. 다음 평가 중에는 이전 표를 유지합니다. 미학습은 성공0%와 다릅니다.</p></div>
</section></div><p id="connection" class="muted" style="font-size:12px">로컬 기록을 읽고 있습니다.</p></main>
<script>
const $=id=>document.getElementById(id);const N=x=>x!==null&&x!==''&&typeof x!=='boolean'&&Number.isFinite(Number(x))?Number(x):null;const fmt=x=>N(x)===null?'—':Number(x).toLocaleString('ko-KR');const pct=x=>N(x)===null?'—':(100*Number(x)).toFixed(0)+'%';let data={},selectedStage=1,selectedMode='greedy',live=false,replay=null,replayPath='',frameIndex=0,playing=true,lastAnimation=0,currentFrame=null;
for(let i=1;i<=12;i++){let o=document.createElement('option');o.value=i;o.textContent='Stage '+i;$('stage').appendChild(o)}
function series(){let rows=(data.evaluation||[]).filter(r=>Number(r.stage)===selectedStage&&r.mode===selectedMode).sort((a,b)=>a.steps-b.steps),last=rows.at(-1);if(!last)return[];return rows.filter(r=>['seed','episodes','randomize_phase'].every(k=>r[k]===last[k]))}

function teamSteps(x){let n=N(x);return n===null?'—':n>=10000?Number((n/10000).toFixed(2)).toLocaleString('ko-KR')+'만 스텝':fmt(n)+' 스텝'}
function decimal(x,d=1){return N(x)===null?'—':Number(x).toFixed(d)}
function renderMonitor(){let m=data.monitor||{},p=m.progress||{},c=m.evaluation||{},guard=m.guard||{},layout=m.layout||{},t=$('monitorRows');t.replaceChildren();let rows=[['평가 모델',teamSteps(c.steps)+' · '+(c.complete?'12맵 G/S 완료':'평가 미완료')+(c.pending?' · 다음 평가 진행 중':'')],['실제 엔트로피 계수',decimal(m.entropy_coef,6)],['정책 엔트로피',decimal(p.entropy,4)],['방문 보상 현재 계수',decimal(p.novelty_coefficient,6)],['정책 / 가치 손실',decimal(p.policy_loss,6)+' / '+decimal(p.value_loss,6)],['처리량',decimal(p.fps,0)+' 스텝/초'],['마지막 집중 맵',(layout.environment_stage_pools?.[2]?.length===12?'전12맵 균등':layout.environment_stage_pools?.slice(2).flat().join(' / '))??'기록 대기'],['4환경 맵 배분',JSON.stringify(layout.environment_stage_pools??layout.layout??layout.env_pools??layout.pools??'기록 대기')],['보상 프로필',m.reward_profile??'기록 대기'],['정체 검토 상태',guard.status??'기록 대기']];for(let [a,b]of rows){let tr=document.createElement('tr');for(let v of[a,b]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}t.appendChild(tr)}let changes=[...(m.entropy_changes||[]).map(r=>({steps:r.steps,text:'엔트로피 '+decimal(r.before,5)+' → '+decimal(r.after,5)})),...(m.adjustment_history||[]).map(r=>({steps:r.steps,text:(r.kind||'조정')+' '+JSON.stringify(r.before)+' → '+JSON.stringify(r.after)+' · '+(r.reason||'')}))].sort((a,b)=>a.steps-b.steps);$('changeHistory').replaceChildren();for(let r of changes.slice(-6)){let p=document.createElement('p');p.textContent=teamSteps(r.steps)+' · '+r.text;$('changeHistory').appendChild(p)}if(!changes.length)$('changeHistory').textContent='기록된 변경 없음';$('allSettings').replaceChildren();for(let group of data.settings||[]){let h=document.createElement('h2');h.textContent=group.title;h.style.marginTop='18px';$('allSettings').appendChild(h);let table=document.createElement('table');for(let row of group.rows||[]){let tr=document.createElement('tr');for(let value of row){let td=document.createElement('td');td.textContent=value;td.style.whiteSpace='normal';tr.appendChild(td)}table.appendChild(tr)}$('allSettings').appendChild(table)}}
function setupCanvas(canvas,height){let rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,rect.width*ratio);canvas.height=height*ratio;canvas.style.height=height+'px';let c=canvas.getContext('2d');c.scale(ratio,ratio);return[c,rect.width,height]}
function plot(id,rows,lines,maxY,label,threshold=null){let[c,w,h]=setupCanvas($(id),205),m={l:42,r:15,t:22,b:30},pw=w-m.l-m.r,ph=h-m.t-m.b;c.font='11px -apple-system,sans-serif';c.fillStyle='#66758a';c.fillText(label,m.l,13);let xmax=Math.max(1,...rows.map(r=>Number(r.steps)||0)),xmin=0;for(let n=0;n<=4;n++){let y=m.t+ph*(1-n/4);c.strokeStyle='#e9edf3';c.beginPath();c.moveTo(m.l,y);c.lineTo(w-m.r,y);c.stroke();c.fillStyle='#8490a1';c.textAlign='right';c.fillText((maxY*n/4).toFixed(maxY<=2?1:0),m.l-7,y+4)}for(let n=0;n<=4;n++){let v=xmax*n/4;c.textAlign='center';c.fillText(v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?Math.round(v/1000)+'k':Math.round(v),m.l+pw*n/4,h-8)}if(threshold!==null){c.strokeStyle='#8750c5';c.setLineDash([4,4]);let y=m.t+ph*(1-threshold/maxY);c.beginPath();c.moveTo(m.l,y);c.lineTo(w-m.r,y);c.stroke();c.setLineDash([])}let any=false;for(let line of lines){c.strokeStyle=line.color;c.fillStyle=line.color;c.lineWidth=2;c.beginPath();let started=false;for(let r of rows){let v=N(r[line.key]);if(v===null)continue;any=true;let x=m.l+pw*(Number(r.steps)-xmin)/(xmax-xmin),y=m.t+ph*(1-v/maxY);if(!started){c.moveTo(x,y);started=true}else c.lineTo(x,y)}c.stroke();for(let r of rows){let v=N(r[line.key]);if(v===null)continue;c.beginPath();c.arc(m.l+pw*Number(r.steps)/xmax,m.t+ph*(1-v/maxY),2.5,0,Math.PI*2);c.fill()}}if(!any){c.fillStyle='#8490a1';c.textAlign='center';c.fillText('평가 기록을 기다리고 있습니다.',w/2,h/2)}}
function render(){let s=data.status||{},p=(data.progress||[]).at(-1)||{},rows=series(),last=data.selected_model?.metrics?.[selectedMode]?.[String(selectedStage)]||rows.at(-1)||{};$('status').textContent=({training:'학습 중',running:'학습 중',paused:'저장 후 대기',interrupted:'저장 후 대기',complete:'학습 종료',completed:'학습 종료',failed:'실행 확인 필요',prepared:'학습 시작 대기 · 0스텝',reviewed:'평가 완료',time_limit:'시간 제한 · 저장 완료',final_evaluation:'최종 평가 중',goal_achieved:'전12맵 검증 통과',complete_not_passed:'학습 종료 · 목표 미달'})[s.status]||s.status||'기록 대기';$('runName').textContent=data.experiment?.run_name||s.run_name||'';$('steps').textContent=teamSteps(s.steps);$('updates').textContent=fmt(s.updates);$('pool').textContent=(s.stage_pool||p.stage_pool||[]).join(' · ')||'—';$('score').textContent=N(last.mean_score)===null?'—':Number(last.mean_score).toFixed(1);$('highscore').textContent=pct(last.star3_rate);$('budgetBar').style.width=(s.total_steps?Math.min(100,100*s.steps/s.total_steps):0)+'%';$('fps').textContent=p.fps?Math.round(p.fps)+' 학습 행동/초':'';$('checkpoint').textContent='현재: '+(s.latest_checkpoint||'저장 대기');$('bestCheckpoint').textContent=s.best_checkpoint?'최고 평가: '+s.best_checkpoint:'';plot('scorePlot',rows,[{key:'mean_score',color:'#2962e7'}],100,'실제 게임 평균 점수',85);plot('ratePlot',rows,[{key:'return_rate',color:'#13845a'},{key:'star1_rate',color:'#c67912'},{key:'star2_rate',color:'#2962e7'},{key:'star3_rate',color:'#8750c5'}],1,'실제 평가 비율');let prog=data.progress||[];plot('entropyPlot',prog,[{key:'entropy',color:'#c67912'}],Math.max(.1,...prog.map(r=>Number(r.entropy)||0))*1.1,'정책 엔트로피');renderTable();renderComparison();renderReplayOptions();if(currentFrame)drawGame(currentFrame)}
function renderComparison(){let box=$('comparisonTable');box.replaceChildren();let f=data.final_results?.final_validation;for(let [key,rows]of Object.entries(f?.results||{})){let values=Object.values(rows),row=rows[String(selectedStage)]||{},tr=document.createElement('tr'),passed=values.filter(x=>x.star3_rate>=.9&&x.return_rate>=.9).length;for(let value of[key.includes('greedy')?'최대확률 G · '+key.split('_')[0]:'확률40회 S · '+key.split('_')[0],'2,388',teamSteps(f.steps),passed+'/12',decimal(row.mean_score),pct(row.star3_rate)]){let td=document.createElement('td');td.textContent=value;tr.appendChild(td)}box.appendChild(tr)}}
function renderTable(){let table=$('stageTable');table.replaceChildren();for(let r of data.monitor?.stages||[]){let g=r.greedy||{},s=r.stochastic||{},tr=document.createElement('tr');if(r.stage===selectedStage)tr.className='selected';let reasons=Object.entries(r.failures||{}).map(([k,v])=>({timeout:'시간',courage:'용기'}[k]||k)+' '+v).join(' · '),values=[r.stage,r.trained?teamSteps(r.learner_steps):'미학습 / 기록 대기',decimal(r.training?.mean_score),decimal(g.mean_score),pct(g.star3_rate),decimal(s.mean_score),pct(s.return_rate),pct(s.star3_rate),pct(r.quest_rate),decimal(r.mean_warmth),reasons||(r.greedy||r.stochastic?'실패 없음':'—')];for(let v of values){let td=document.createElement('td');td.textContent=v;if(v==='미학습 / 기록 대기'||v==='—')td.className='empty';tr.appendChild(td)}tr.onclick=()=>{selectedStage=r.stage;$('stage').value=r.stage;render()};table.appendChild(tr)}renderMonitor()}
function renderReplayOptions(){let choices=(data.evaluation||[]).filter(r=>Number(r.stage)===selectedStage&&r.mode===selectedMode&&r.replay).map(r=>({path:r.replay,label:fmt(r.steps)+'행동 · Stage'+r.stage+' · '+(r.mode==='greedy'?'최대확률':'확률')+' · '+Number(r.mean_score).toFixed(1)+'점'}));for(let path of data.extra_replays||[])choices.push({path,label:path});choices.reverse();let old=$('replaySelect').value;$('replaySelect').replaceChildren();let seen=new Set();for(let choice of choices){if(seen.has(choice.path))continue;seen.add(choice.path);let o=document.createElement('option');o.value=choice.path;o.textContent=choice.label;$('replaySelect').appendChild(o)}if(seen.has(old))$('replaySelect').value=old;if(!live&&$('replaySelect').value&&$('replaySelect').value!==replayPath)loadReplay($('replaySelect').value)}
function coord(c,p,x,y,size){return[(p[0]+.5)*size+x,(p[1]+.5)*size+y]}
function drawGame(frame){currentFrame=frame;let grid=frame.grid||[],height=grid.length||9,width=(grid[0]||'').length||16,[c,w,h]=setupCanvas($('game'),Math.max(180,$('game').getBoundingClientRect().width*height/width)),size=Math.min(w/width,h/height),ox=(w-size*width)/2,oy=(h-size*height)/2;c.fillStyle='#edf1f7';c.fillRect(0,0,w,h);if(!grid.length){c.fillStyle='#66758a';c.textAlign='center';c.fillText('실제 게임 상태를 기다리고 있습니다.',w/2,h/2);return}for(let y=0;y<height;y++)for(let x=0;x<width;x++){c.fillStyle=grid[y][x]==='#'?'#526079':'#f7f8fc';c.fillRect(ox+x*size,oy+y*size,size-.5,size-.5)}function dot(p,color,r=.22){if(!p)return;c.fillStyle=color;c.beginPath();let[x,y]=coord(c,p,ox,oy,size);c.arc(x,y,size*r,0,Math.PI*2);c.fill()}function rect(p,color,r=.3){if(!p)return;let[x,y]=coord(c,p,ox,oy,size);c.fillStyle=color;c.fillRect(x-r*size,y-r*size,r*2*size,r*2*size)}rect(frame.exit,'#87d3b0',.4);for(let door of frame.shortcuts||[]){if(!door.opened)rect(door.gate,'#b88a4d',.48)}for(let item of frame.items||[])if(item.active){dot(item.position,'#edb844',item.value>=3?.25:.15);if(item.value>=3){let[x,y]=coord(c,item.position,ox,oy,size);c.fillStyle='#6c4305';c.font='bold '+Math.max(9,size*.3)+'px sans-serif';c.textAlign='center';c.fillText(String(item.value),x,y+size*.1)}}if(frame.quest&&!frame.quest.collected)rect(frame.quest.position,'#399cf2',.25);if(frame.magnet&&!frame.magnet.collected)dot(frame.magnet.position,'#eb7eae',.2);if(frame.invincibility&&!frame.invincibility.collected)dot(frame.invincibility.position,'#7a8edb',.2);for(let ghost of frame.ghosts||[])dot(ghost.position,ghost.kind==='chaser'?'#d44f65':ghost.id==='A'?'#9663ce':'#9676dc',.29);dot(frame.player,'#172437',.29);if(frame.player){let[x,y]=coord(c,frame.player,ox,oy,size);c.fillStyle='#172437';for(let sign of[-1,1]){c.beginPath();c.moveTo(x+sign*size*.25,y-size*.1);c.lineTo(x+sign*size*.18,y-size*.4);c.lineTo(x+sign*size*.04,y-size*.2);c.fill()}c.fillStyle='#fff';c.beginPath();c.arc(x-size*.09,y-size*.03,size*.035,0,7);c.arc(x+size*.09,y-size*.03,size*.035,0,7);c.fill()}let hud=$('gameHud');hud.replaceChildren();let items=['온기 '+frame.carried,'용기 '+frame.courage,'남은 시간 '+frame.remaining_ticks,'퀘스트 '+(frame.quest&&frame.quest.collected?'수집':'탐색 중')];if(frame.score!==null&&frame.score!==undefined)items.push('실제 점수 '+frame.score);for(let t of items){let e=document.createElement('span');e.textContent=t;hud.appendChild(e)}}
async function loadReplay(path){if(!path)return;try{let response=await fetch('/api/replay?path='+encodeURIComponent(path));if(!response.ok)throw Error('읽기 실패');replay=await response.json();replayPath=path;frameIndex=0;playing=true;$('play').textContent='일시정지';$('scrub').max=Math.max(0,(replay.frames||[]).length-1);showReplayFrame()}catch(e){$('playLabel').textContent='아직 완성되지 않은 재생 기록입니다. 다음 갱신 때 다시 선택해 주세요.'}}
function showReplayFrame(){if(!replay||!replay.frames?.length)return;frameIndex=Math.min(frameIndex,replay.frames.length-1);drawGame(replay.frames[frameIndex]);$('scrub').value=frameIndex;$('playLabel').textContent='Stage '+replay.stage+' · '+(replay.mode||'평가')+' · '+(frameIndex+1)+'/'+replay.frames.length+' · '+(replay.checkpoint||replayPath)}
function changeMode(isLive){live=isLive;$('liveButton').classList.toggle('active',live);$('replayButton').classList.toggle('active',!live);$('replayTools').hidden=live;if(live){$('playLabel').textContent='학습 중 실제 상태 · 주기적 갱신';pollLive()}else{renderReplayOptions();loadReplay($('replaySelect').value)}}
async function pollLive(){if(!live)return;try{let r=await fetch('/api/live');let x=await r.json(),entry=x.environments?.[Number($('liveEnv').value)]||x;if(entry.frame){drawGame(entry.frame);$('playLabel').textContent=(x.mode==='prepared'?'학습 시작 대기 · 초기 맵':data.status?.status==='training'?'학습 중 실제 맵':'마지막으로 저장한 실제 맵')+' · 환경 '+(Number($('liveEnv').value)+1)+' · Stage '+(entry.stage||entry.frame.stage)+' · '+fmt(x.steps)+'행동'}}catch(e){}}
async function poll(){try{let response=await fetch('/api/state');if(!response.ok)throw Error();data=await response.json();render();$('connection').textContent='마지막 갱신 '+new Date().toLocaleTimeString('ko-KR')+' · '+fmt((data.evaluation||[]).length)+'개 평가 기록'}catch(e){$('connection').textContent='연결을 기다리고 있습니다. 저장된 화면은 유지합니다.'}await pollLive()}
$('stage').onchange=()=>{selectedStage=Number($('stage').value);render()};$('mode').onchange=()=>{selectedMode=$('mode').value;render()};$('liveButton').onclick=()=>changeMode(true);$('liveEnv').onchange=()=>pollLive();$('replayButton').onclick=()=>changeMode(false);$('replaySelect').onchange=()=>loadReplay($('replaySelect').value);$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'일시정지':'재생'};$('restart').onclick=()=>{frameIndex=0;playing=true;$('play').textContent='일시정지';showReplayFrame()};$('scrub').oninput=()=>{frameIndex=Number($('scrub').value);playing=false;$('play').textContent='재생';showReplayFrame()};window.addEventListener('resize',render);
function animate(t){if(!live&&replay&&playing&&t-lastAnimation>1000/Number($('speed').value)){if(frameIndex<replay.frames.length-1){frameIndex++;showReplayFrame()}else{playing=false;$('play').textContent='재생'}lastAnimation=t}requestAnimationFrame(animate)}
poll();setInterval(poll,2000);setInterval(pollLive,500);requestAnimationFrame(animate);
</script></body></html>'''


def _handler(run):
    root = Path(run).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_payload(self, value, *, status=200, content_type='application/json; charset=utf-8'):
            payload = value if isinstance(value, bytes) else json.dumps(
                value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                return self.send_payload(HTML.encode('utf-8'), content_type='text/html; charset=utf-8')
            if url.path == '/api/state':
                return self.send_payload(dashboard_state(root))
            if url.path == '/api/live':
                return self.send_payload(read_json(root / 'live.json'))
            if url.path == '/health':
                return self.send_payload({'ok': True, 'input_dimensions': 2388, 'experiment': root.name})
            if url.path == '/api/replay':
                raw = parse_qs(url.query).get('path', [''])[0]
                try:
                    path = (root / raw).resolve()
                except (OSError, ValueError):
                    return self.send_payload({'error': 'invalid replay path'}, status=403)
                if not raw or not path.is_relative_to(root) or path.suffix != '.json':
                    return self.send_payload({'error': 'invalid replay path'}, status=403)
                value = read_json(path)
                if not isinstance(value, dict) or not isinstance(value.get('frames'), list):
                    return self.send_payload({'error': 'replay not ready'}, status=404)
                return self.send_payload(value)
            self.send_payload({'error': 'not found'}, status=404)

    return Handler


def start_dashboard(run, port=8765, host='127.0.0.1'):
    """기존 run만 읽는 서버를 시작합니다. 호출자는 shutdown/server_close할 수 있습니다."""
    run = Path(run).resolve(strict=True)
    if not run.is_dir() or host not in ('127.0.0.1', 'localhost'):
        raise ValueError('대시보드는 기존 실행 폴더와 로컬 호스트만 지원합니다.')
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('port는0..65535 정수여야 합니다.')
    # 서버는 로컬에서 기록을 읽기만 하며 별도 스레드로 학습과 함께 실행됩니다.
    server = ThreadingHTTPServer((host, port), _handler(run))
    server.daemon_threads = True
    server.dashboard_url = f'http://127.0.0.1:{server.server_address[1]}'
    server.dashboard_thread = threading.Thread(target=server.serve_forever, daemon=True,
                                               name='ppo-live-dashboard')
    server.dashboard_thread.start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = start_dashboard(args.run, args.port)
    print(f'PPO 학습 화면: {server.dashboard_url}', flush=True)
    try:
        server.dashboard_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
