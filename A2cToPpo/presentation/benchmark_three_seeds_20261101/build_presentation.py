"""원본 비교 JSON에서 발표 표·대본·3장 PDF를 생성합니다. 학습은 실행하지 않습니다."""

import csv
from functools import lru_cache
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
STAGES = [1, 2, 5, 6, 7, 8, 11, 12]
SEEDS = [11, 22, 33]
METRICS = ['success_rate', 'three_star_rate', 'mean_score', 'mean_stars', 'mean_damage']
COLORS = {'bg':'#0B1220','panel':'#142036','line':'#2A3B54','text':'#F2F5FB',
          'muted':'#A5B6CE','a2c':'#4DD5BC','ppo':'#FFCC79'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return format(Decimal(str(value)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP),'.2f')


def collect():
    rows,stage_rows,manifest,configs=[],[],{},[]
    for seed in SEEDS:
        matches=[]
        for path in (PROJECT/'A2cToPpo/runs/comparisons').glob(f'seed_{seed}_*.json'):
            data=read(path)
            if data.get('profile','benchmark')=='benchmark':matches.append((path,data))
        if len(matches)!=1:raise ValueError(f'시드 {seed}의 benchmark 비교 파일을 하나로 지정해야 합니다.')
        path,data=matches[0]
        manifest[str(path.relative_to(PROJECT))]=sha(path)
        for algorithm in ['A2C','PPO']:
            run=Path(data['runs'][algorithm]);config=read(run/'config.json');summary=read(run/'training_summary.json')
            if config['training']['physical_frames']!=300000:raise ValueError('학습 예산이 다릅니다.')
            configs.append(config)
            manifest[str((run/'config.json').relative_to(PROJECT))]=sha(run/'config.json')
            manifest[str((run/'training_summary.json').relative_to(PROJECT))]=sha(run/'training_summary.json')
            for role in ['baseline','last','candidate']:
                per_stage=data['results'][algorithm]['per_stage'][role]
                selected=[per_stage[str(stage)] for stage in STAGES]
                episodes=sum(value['episodes'] for value in selected)
                if episodes!=800 or any(value['episodes']!=100 for value in selected):
                    raise ValueError('스테이지별 확률적 평가 판 수가 다릅니다.')
                row={'seed':seed,'algorithm':algorithm,'model_role':role,
                     'episodes':episodes,'requested_physical_frames':300000,
                     'actual_training_frames':data['results'][algorithm]['actual_physical_frames'],
                     'decision_steps':data['results'][algorithm]['decision_steps'],
                     'candidate_training_frames':summary['candidate_model_physical_frames']}
                row.update({metric:sum(value[metric]*value['episodes'] for value in selected)/episodes for metric in METRICS})
                rows.append(row)
                for stage in STAGES:
                    value=per_stage[str(stage)]
                    stage_rows.append({'seed':seed,'algorithm':algorithm,'model_role':role,
                                       'stage':stage,'episodes':value['episodes'],
                                       **{metric:value[metric] for metric in METRICS}})
    reference=configs[0]
    for config in configs:
        for key in ['source_environment_config','common_updates','evaluation']:
            if config[key]!=reference[key]:raise ValueError('공통 비교 조건이 다릅니다: '+key)
        if config['source']['source_sha256']!=reference['source']['source_sha256']:
            raise ValueError('초기 가중치가 다릅니다.')
    stats_rows=[]
    for role in ['last','candidate']:
        for algorithm in ['A2C','PPO']:
            values=[r for r in rows if r['algorithm']==algorithm and r['model_role']==role]
            for metric in METRICS:
                xs=[r[metric] for r in values]
                stats_rows.append({'algorithm':algorithm,'model_role':role,'metric':metric,
                                   'training_seed_count':len(xs),'mean':mean(xs),'sample_std':stdev(xs)})
    pairs=[]
    for seed in SEEDS:
        a=next(r for r in rows if r['seed']==seed and r['algorithm']=='A2C' and r['model_role']=='last')
        p=next(r for r in rows if r['seed']==seed and r['algorithm']=='PPO' and r['model_role']=='last')
        pairs.append({'seed':seed,**{metric:p[metric]-a[metric] for metric in METRICS}})
    selected=next(r for r in rows if r['algorithm']=='PPO' and r['seed']==11 and r['model_role']=='candidate')
    model=PROJECT/'A2cToPpo/final/ppo_benchmark_seed11/models/candidate_model.zip'
    original=PROJECT/'A2cToPpo/runs/ppo_seed_11_20261101_164722_768756/models/candidate_model.zip'
    if sha(model)!=sha(original):raise ValueError('보관한 최종 모델이 원본 후보와 다릅니다.')
    manifest[str(model.relative_to(PROJECT))]=sha(model)
    return rows,stage_rows,stats_rows,pairs,selected,manifest,reference


@lru_cache(maxsize=48)
def font(size,bold=False):
    return ImageFont.truetype(str(Path('C:/Windows/Fonts')/('malgunbd.ttf' if bold else 'malgun.ttf')),size)


def text(draw,xy,value,size=30,color='text',bold=False,anchor=None):
    draw.text(xy,value,font=font(size,bold),fill=COLORS.get(color,color),anchor=anchor)


def paragraph(draw,xy,value,width,size=30,color='text',line_height=None,bold=False):
    x,y=xy;line_height=line_height or int(size*1.55)
    for line in value.split('\n'):
        current=''
        for char in line:
            trial=current+char
            if draw.textlength(trial,font=font(size,bold))>width and current:
                text(draw,(x,y),current,size,color,bold);y+=line_height;current=char
            else:current=trial
        text(draw,(x,y),current,size,color,bold);y+=line_height
    return y


def page(number,title,subtitle):
    image=Image.new('RGB',(1920,1080),COLORS['bg']);draw=ImageDraw.Draw(image)
    text(draw,(80,45),'GAME AI  /  BENCHMARK',23,'muted',True)
    text(draw,(80,105),title,57,'text',True)
    text(draw,(80,190),subtitle,29,'muted')
    draw.line((80,1000,1840,1000),fill=COLORS['line'],width=2)
    text(draw,(80,1022),'집계: 선택한 8개 스테이지 · 확률적 100판씩 · 추가 학습 시드 11 / 22 / 33',22,'muted')
    text(draw,(1840,1022),f'{number:02d} / 03',22,'muted',anchor='ra')
    return image,draw


def metric(stats_rows,algorithm,metric_name):
    return next(r for r in stats_rows if r['algorithm']==algorithm and r['model_role']=='last' and r['metric']==metric_name)


def render(rows,stats_rows,selected):
    pages=[]
    image,draw=page(1,'같은 출발점에서 A2C와 PPO를 비교했습니다',
                    'A2C 사전 학습 가중치를 고정하고, 이후 업데이트 알고리즘을 나누어 추가 학습했습니다.')
    draw.rounded_rectangle((80,270,1840,455),radius=22,fill=COLORS['panel'])
    text(draw,(130,305),'동일한 A2C 사전 학습 모델',36,'text',True)
    text(draw,(130,363),'정책망 · 가치망 가중치 복사 / 양쪽 모두 새 Adam 최적화기',26,'muted')
    text(draw,(1035,303),'A2C 추가 학습',34,'a2c',True)
    text(draw,(1035,364),'PPO 추가 학습',34,'ppo',True)
    text(draw,(1760,305),'30만 F',29,'a2c',True,anchor='ra')
    text(draw,(1760,366),'30만 F',29,'ppo',True,anchor='ra')
    conditions=[('환경·입력','같은 실시간 게임 / 관측 2,240개 / 5행동 / 18프레임 유지'),
                ('공통 학습 설정','학습률 0.00005 · 엔트로피 0.005 · 환경 4개 · 환경당 128행동 수집'),
                ('반복과 평가','학습 시드 3개 / 알고리즘·시드당 800판 / 마지막 모델을 주 비교'),
                ('PPO 갱신','배치 128 · 최대 4회 반복 · clip 0.1 · target KL 0.02')]
    for i,(label,value) in enumerate(conditions):
        y=505+i*87
        text(draw,(100,y),label,29,'muted',True)
        text(draw,(440,y),value,29)
        draw.line((100,y+61,1820,y+61),fill=COLORS['line'],width=1)
    paragraph(draw,(100,890),'비교 질문: “같은 사전 학습 정책을 어느 방식으로 추가 학습할 때 성능이 유지되는가?”',1720,29,'text',bold=True)
    pages.append(image)

    image,draw=page(2,'시드에 따라 성능 차이가 달라졌습니다',
                    '같은 추가 예산까지 학습한 last_model 기준 · 각 수치는 동일한 최종 평가 시드에서 측정했습니다.')
    widths=[220,250,250,250,250,250,250];xs=[80]
    for width in widths:xs.append(xs[-1]+width)
    draw.rounded_rectangle((80,280,1800,850),radius=20,fill=COLORS['panel'])
    for label,start,end in [('귀환 성공률',xs[1],xs[3]),('별 3개 비율',xs[3],xs[5]),('평균 점수',xs[5],xs[7])]:
        text(draw,((start+end)/2,305),label,30,'text',True,anchor='ma')
    text(draw,((xs[0]+xs[1])/2,350),'학습 시드',28,'muted',True,anchor='ma')
    for col,algorithm in enumerate(['A2C','PPO','A2C','PPO','A2C','PPO'],1):
        text(draw,((xs[col]+xs[col+1])/2,365),algorithm,28,algorithm.lower(),True,anchor='ma')
    draw.line((100,420,1780,420),fill=COLORS['line'],width=2)
    for index,seed in enumerate(SEEDS):
        y=468+index*105
        text(draw,((xs[0]+xs[1])/2,y),str(seed),33,'text',True,anchor='ma')
        for group,metric_name in enumerate(['success_rate','three_star_rate','mean_score']):
            for offset,algorithm in enumerate(['A2C','PPO']):
                row=next(r for r in rows if r['seed']==seed and r['algorithm']==algorithm and r['model_role']=='last')
                col=1+group*2+offset
                value=row[metric_name]*100 if group<2 else row[metric_name]
                text(draw,((xs[col]+xs[col+1])/2,y),number(value)+('%' if group<2 else ''),33,
                     algorithm.lower(),True,anchor='ma')
    draw.line((100,750,1780,750),fill=COLORS['line'],width=2)
    text(draw,((xs[0]+xs[1])/2,782),'평균 ± 표준편차',24,'text',True,anchor='ma')
    for group,metric_name in enumerate(['success_rate','three_star_rate','mean_score']):
        for offset,algorithm in enumerate(['A2C','PPO']):
            m=metric(stats_rows,algorithm,metric_name);scale=100 if group<2 else 1
            col=1+group*2+offset
            text(draw,((xs[col]+xs[col+1])/2,782),f"{number(m['mean']*scale)} ± {number(m['sample_std']*scale)}",27,
                 algorithm.lower(),True,anchor='ma')
    text(draw,(100,886),'시드 11에서는 PPO가 높았고, 시드 22·33에서는 A2C가 높았습니다.',32,'text',True)
    text(draw,(100,942),'표준편차는 3개 학습 시드 간 값입니다. 비율의 표준편차 단위는 %p입니다.',25,'muted')
    pages.append(image)

    image,draw=page(3,'알고리즘 평균과 시연 모델 선택을 구분했습니다',
                    '평균 성능의 해석과 특정 모델 한 개의 채택은 서로 다른 판단입니다.')
    draw.rounded_rectangle((80,280,940,875),radius=22,fill=COLORS['panel'])
    draw.rounded_rectangle((980,280,1840,875),radius=22,fill=COLORS['panel'])
    text(draw,(125,320),'3개 시드에서 관찰한 결과',34,'text',True)
    text(draw,(125,398),'별 3개 평균',27,'muted')
    a=metric(stats_rows,'A2C','three_star_rate');p=metric(stats_rows,'PPO','three_star_rate')
    text(draw,(125,445),f"A2C  {number(a['mean']*100)}%",45,'a2c',True)
    text(draw,(525,445),f"PPO  {number(p['mean']*100)}%",45,'ppo',True)
    text(draw,(125,535),'별 3개 비율의 시드 간 표준편차',27,'muted')
    text(draw,(125,580),f"A2C  {number(a['sample_std']*100)}%p",36,'a2c',True)
    text(draw,(525,580),f"PPO  {number(p['sample_std']*100)}%p",36,'ppo',True)
    paragraph(draw,(125,680),'이번 조건에서는 A2C의 평균 수치가\n조금 높고, 시드 간 변동은 작았습니다.\n3개 시드의 관찰 결과로 보편적 우위를\n확정하지 않습니다.',760,29,'text',line_height=46)
    text(draw,(1025,320),'게임 시연용으로 선택한 모델',34,'text',True)
    text(draw,(1025,398),'PPO · seed 11 · benchmark 후보',31,'ppo',True)
    text(draw,(1025,470),f"귀환 성공률  {number(selected['success_rate']*100)}%",43,'text',True)
    text(draw,(1025,545),f"별 3개 비율  {number(selected['three_star_rate']*100)}%",43,'text',True)
    paragraph(draw,(1025,660),'성능이 높았던 특정 시드의 모델을\n보관했습니다.\n개별 모델의 결과와 알고리즘의\n평균 성능을 구분해 제시합니다.',760,29,'text',line_height=46)
    paragraph(draw,(100,905),'발표 결론: 같은 사전 학습 정책의 추가 학습을 비교했고, 시드 변동까지 확인한 뒤 시연 모델을 선택했습니다.',1720,28,'text',bold=True)
    pages.append(image)
    for i,image in enumerate(pages,1):image.save(OUT/f'slide_{i:02d}.png')
    pages[0].save(OUT/'A2C_PPO_3개시드_비교발표.pdf',save_all=True,append_images=pages[1:],
                  resolution=150.0,quality=95,title='A2C·PPO 추가 학습: 3개 시드 비교',author='game_project')


def notes(rows,stats_rows,selected,manifest):
    lines=['# A2C·PPO 3개 시드 비교: 발표 자료와 대본','',
           '## 표를 읽는 기준','',
           '- 비교 유형: 같은 A2C 사전 학습 가중치에서 A2C·PPO로 추가 학습한 비교.',
           '- 프로필: benchmark. return_practice 결과는 이 표에 섞지 않았습니다.',
           '- 주 표: 같은 추가 예산까지 학습한 last_model. 후보 선택 시점 차이를 주 비교에서 제거했습니다.',
           '- 집계 스테이지: '+', '.join(map(str,STAGES))+'. 스테이지별 확률적 100판, 시드·알고리즘당 800판.',
           '- 학습 시드: 11·22·33. 초기 A2C 사전 학습 가중치는 하나로 고정했습니다.',
           '- 요청 예산: 각 실행 추가 300,000 물리 프레임. 실제 묶음 종료 시점은 CSV에 별도 기록했습니다.',
           '- 평균·표본 표준편차(ddof=1)는 3개 학습 시드의 값입니다. 알고리즘당 2,400판을 독립 학습 2,400회로 세지 않습니다.',
           '- 원본 평가 설정·원본 보고서는 수정하지 않았습니다. 이 자료는 발표용 파생 집계입니다.','',
           '## 1장: 같은 출발점에서 비교','',
           '저희는 같은 A2C 사전 학습 모델을 출발점으로, A2C와 PPO를 각각 추가 학습했습니다. '
           '환경, 관측, 신경망, 보상, 추가 프레임 예산은 맞췄고, 양쪽 모두 새 Adam 최적화기로 시작했습니다. '
           '학습 시드는 11, 22, 33으로 반복했습니다. 따라서 처음부터 무작위 가중치로 학습한 알고리즘 비교가 아니라, '
           '동일한 사전 학습 정책을 어떻게 보완하는지에 대한 비교입니다.','',
           '## 2장: 시드별 결과','',
           '| 시드 | A2C 성공률 | PPO 성공률 | A2C 별 3개 | PPO 별 3개 | A2C 평균 점수 | PPO 평균 점수 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        a=next(r for r in rows if r['seed']==seed and r['algorithm']=='A2C' and r['model_role']=='last')
        p=next(r for r in rows if r['seed']==seed and r['algorithm']=='PPO' and r['model_role']=='last')
        lines.append(f"| {seed} | {number(a['success_rate']*100)}% | {number(p['success_rate']*100)}% | "
                     f"{number(a['three_star_rate']*100)}% | {number(p['three_star_rate']*100)}% | "
                     f"{number(a['mean_score'])} | {number(p['mean_score'])} |")
    lines += ['', '| 항목 | A2C 평균 ± 표준편차 | PPO 평균 ± 표준편차 |','|---|---:|---:|']
    for key,label,scale in [('success_rate','귀환 성공률',100),('three_star_rate','별 3개 비율',100),
                             ('mean_score','평균 점수',1)]:
        a=metric(stats_rows,'A2C',key);p=metric(stats_rows,'PPO',key)
        lines.append(f"| {label} | {number(a['mean']*scale)} ± {number(a['sample_std']*scale)} | "
                     f"{number(p['mean']*scale)} ± {number(p['sample_std']*scale)} |")
    lines += ['', '비율의 평균 단위는 %, 표준편차 단위는 %p입니다. 점수 단위는 점입니다.','',
              '**발표 대본:** 시드 11에서는 PPO가 더 높았지만, 시드 22와 33에서는 A2C가 더 높았습니다. '
              '세 시드의 별 3개 비율 평균은 A2C 97.58%, PPO 96.83%였습니다. '
              '시드 간 표준편차는 각각 0.19%p와 1.46%p로, 이번 조건에서는 A2C가 더 작은 변동을 보였습니다. '
              '다만 학습 시드가 세 개인 소규모 비교이므로, 이 차이를 통계적으로 확정된 우열로 주장하지 않습니다.','',
              '## 3장: 해석과 모델 선택','',
              f"시연 모델은 PPO seed 11의 benchmark 후보입니다. 귀환 성공률 {number(selected['success_rate']*100)}%, "
              f"별 3개 비율 {number(selected['three_star_rate']*100)}%였습니다.",
              '보관 위치: A2cToPpo/final/ppo_benchmark_seed11/models/candidate_model.zip','',
              '**발표 대본:** 알고리즘의 평균 성능과 게임에 사용할 모델 한 개의 선택은 구분했습니다. '
              '이번 세 시드 평균에서는 A2C가 조금 높고 안정적이었고, PPO도 높은 플레이 성능을 보였습니다. '
              '게임 시연에는 개별 검증 결과가 좋았던 PPO 시드 11 후보를 보관했습니다. '
              '이 특정 모델의 성능을 PPO 전체의 평균 성능이라고 표현하지는 않습니다.','',
              '## 발표에서 사용할 수 있는 문장','',
              '> 같은 A2C 사전 학습 정책을 두 알고리즘으로 추가 학습하고, 세 개의 학습 시드로 성능과 변동을 비교했습니다.',
              '> 이번 조건에서는 A2C의 평균 수치가 조금 높고 시드 변동이 작았으며, 게임 시연에는 PPO 시드 11 후보를 선택했습니다.','',
              '피할 표현: “처음부터 PPO로만 학습했다”, “PPO가 모든 시드에서 우수했다”, '
              '“세 번의 실험으로 알고리즘 우열이 증명됐다”, “선택한 모델의 결과가 PPO 평균이다”.','',
              '## 예상 질문과 답변','',
              '**왜 마지막 모델을 주로 비교했나요?** 같은 추가 예산을 사용한 상태를 비교하려는 목적입니다. '
              '정기 평가로 고른 후보의 결과는 CSV에 별도로 남겼습니다.','',
              '**평균은 A2C가 높은데 왜 PPO를 시연하나요?** 알고리즘 평균과 실제 사용할 모델 한 개의 선택은 다릅니다. '
              '시연에는 검증 결과가 좋았던 특정 PPO 모델을 골랐으며, 이를 평균 우위라고 주장하지 않습니다.','',
              '**같은 프레임 예산이면 계산량도 같은가요?** 아닙니다. 새로 수집하는 물리 프레임 예산을 맞춘 것입니다. '
              'PPO는 수집한 경험을 여러 번 갱신하므로 같은 실행 시간이나 같은 연산량을 보장하는 비교가 아닙니다.','',
              '**표준편차는 무엇인가요?** 3개 추가 학습 시드 사이의 변동입니다. 신뢰구간이나 유의확률이 아닙니다. '
              '원래 A2C 사전 학습 모델은 하나로 고정했습니다.','',
              '## 원본과 파생 자료','',
              '- seed_results.csv: 기준·마지막·후보의 시드별 집계와 실제 프레임 수.',
              '- summary_stats.csv: 마지막·후보를 구분한 세 시드 평균·표준편차.',
              '- stage_results.csv: 스테이지별 원래 수치.',
              '- paired_differences.csv: 같은 시드에서 PPO−A2C 차이. 비율 값은 1.0=100%p.',
              '- analysis.json: 위 집계와 최종 보관 모델 수치.',
              '- sources_manifest.json: 원본 JSON·설정·선택 모델의 SHA-256.',
              '- 후보 표에는 후보를 찾지 못해 기준 모델이 남은 경우도 포함될 수 있습니다. candidate_training_frames=0이면 이 경우입니다. '
              '주 비교에는 이런 선택 정책의 영향을 줄이기 위해 마지막 모델을 사용했습니다.','']
    lines += ['원본 파일:']+['- '+path for path in manifest if '/comparisons/' in path]
    (OUT/'발표대본과_결과정리.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    rows,stages,summary,pairs,selected,sources,config=collect()
    write_csv('seed_results.csv',rows);write_csv('stage_results.csv',stages)
    write_csv('summary_stats.csv',summary);write_csv('paired_differences.csv',pairs)
    write_json('sources_manifest.json',sources)
    write_json('analysis.json',{'profile':'benchmark','stages':STAGES,'training_seeds':SEEDS,
               'primary_model_role':'last','initialization':'A2C_pretrained_weights',
               'statistics':summary,'selected_demo_model':selected,
               'common_updates':config['common_updates']})
    notes(rows,summary,selected,sources)
    render(rows,summary,selected)
    print('생성 완료:',OUT)
    print('PDF 3장 / 발표 대본 / 원본 근거와 집계 CSV')


if __name__=='__main__':
    main()
