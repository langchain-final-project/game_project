"""로컬 글꼴과 문서 레이아웃으로 PDF를 생성합니다. 학습은 하지 않습니다."""

from pathlib import Path
from html.parser import HTMLParser
from decimal import Decimal, ROUND_HALF_UP
import json
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FINAL = ROOT / 'reports/local/experiments/stage2_safe_return_final/quick/seed_11_20261101_040335_220183'
S = 2
REG = 'C:/Windows/Fonts/malgun.ttf'
BOLD = 'C:/Windows/Fonts/malgunbd.ttf'
BG = '#1c192b'
PAPER = '#fffdf8'
FG = '#f5f0e6'
GOLD = '#e9c382'
MUTED = '#c5b9cf'
SAGE = '#a8c8b0'
PINK = '#efb1b4'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REG, round(size * S))


def wrap(text, size, width, bold=False):
    f = font(size, bold)
    result = []
    for paragraph in text.split('\n'):
        line = ''
        for ch in paragraph:
            if line and f.getlength(line + ch) > width * S:
                result.append(line.rstrip())
                line = ch.lstrip()
            else:
                line += ch
        result.append(line)
    return result


class Page:
    def __init__(self, width=1600, height=900, bg=BG):
        self.w, self.h = width, height
        self.im = Image.new('RGB', (width*S, height*S), bg)
        self.d = ImageDraw.Draw(self.im)

    def box(self, x, y, w, h, fill='#292337', edge='#4c3d59', radius=16):
        self.d.rounded_rectangle((x*S, y*S, (x+w)*S, (y+h)*S), radius*S,
                                 fill=fill, outline=edge, width=S)

    def line(self, x1, y1, x2, y2, color='#483b55'):
        self.d.line((x1*S,y1*S,x2*S,y2*S), fill=color, width=S)

    def text(self, value, x, y, size=24, color=FG, bold=False, width=None, spacing=1.5):
        lines = wrap(value, size, width, bold) if width else value.split('\n')
        for line in lines:
            self.d.text((x*S,y*S), line, font=font(size,bold), fill=color)
            y += size*spacing
        return y

    def image(self, path, x, y, w, h):
        source = Image.open(path).convert('RGB')
        source.thumbnail((round(w*S),round(h*S)), Image.Resampling.LANCZOS)
        self.im.paste(source,(round(x*S),round(y*S)))

    def footer(self, n):
        self.line(68,843,1532,843)
        self.text('까망이와 유령 저택 · PPO 적용',68,855,14,MUTED)
        self.text(f'2026.11.01  /  {n:02}',1374,855,14,MUTED)

    def header(self,n,english,title,subtitle):
        self.text(f'{n:02}  {english}',68,39,17,GOLD,True)
        self.text(title,68,87,49,FG,True)
        self.text(subtitle,68,168,22,MUTED)


def pct(value):
    return f'{value*100:g}%'


def number(value):
    return str(Decimal(str(value)).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def slides(metrics, final):
    pages=[]
    p=Page();p.header(1,'GAME × REINFORCEMENT LEARNING','게임 상태를 읽고, 행동을 배우는 PPO',
                     '실시간 탐사 환경에 관측 · 행동 선택 · 보상 피드백을 연결했습니다.')
    p.box(68,227,565,438)
    p.text('640',95,244,91,GOLD,True);p.text('차원 관측',306,309,26,FG,True)
    for y,label,value in [(370,'지형 · 아이템 · 유령 · 시간','630'),(430,'실제 이동 가능 방향','4'),(490,'목표 단계 · 방향 · 온기 조건','6')]:
        p.text(label,96,y,23);p.text(value,530,y,25,SAGE,True);p.line(96,y+48,603,y+48)
    p.text('환경과 목표는 별도 신경망 분기로 처리합니다.\n이동 정보는 행동 마스크에 사용합니다.',96,563,19,MUTED,width=500)
    p.image(OUT/'game_play.png',666,227,866,487)
    p.text('실제 Stage 2 정책 재생 화면 · 이동 4방향과 대기\n선택한 행동을 최대 9프레임 유지',680,722,17,MUTED)
    for i,(title,desc) in enumerate([('게임 상태','수치 관측'),('PPO 정책','행동 확률'),('이동 · 대기','환경 변화'),('보상 · 경험','정책 / 가치 갱신')]):
        x=68+i*147
        p.box(x,698,130,105,fill='#352a42')
        p.text(title,x+13,715,19,FG,True);p.text(desc,x+13,752,13,MUTED)
        if i<3:p.text('→',x+132,735,17,GOLD)
    p.footer(1);pages.append(p)

    p=Page();p.header(2,'LEARNING DESIGN','별 3개를 향해, 부분 행동을 연결하다',
                     '성공 경험이 부족한 문제를 보상 설계와 단계별 복습으로 풀었습니다.')
    cards=[('REWARD','무엇을 잘했다고\n알려줄 것인가?', [('수집 · 목표 접근 · 별 획득','긍정적인 피드백'),('시간 낭비 · 벽막힘 · 피해','비용 부여')], '게임 점수와 학습 보상은 구분'),
           ('CURRICULUM','쉬운 과제부터\n전체 플레이로', [('물건 획득 → 자석과 연결','짧은 부분 과제부터 연습'),('기존 행동을 함께 복습','추가 수집 · 귀환으로 확장')], '중간 시작 상태와 실제 시작점 평가를 구분'),
           ('PPO / ACTOR–CRITIC','경험으로 배우고\n작게 갱신하기', [('정책망: 다음 행동 선택','가치망: 향후 보상 예측'),('Clipping으로 갱신 폭 제한','Stable-Baselines3 PPO 적용')], '목표 안내를 입력받고 행동은 정책이 선택')]
    for i,(label,title,items,note) in enumerate(cards):
        x=68+i*496
        p.box(x,235,472,397)
        p.text(label,x+26,256,16,SAGE,True)
        p.text(title,x+26,297,30,FG,True)
        y=410
        for a,b in items:
            p.text(a,x+26,y,21,GOLD,True,width=420)
            p.text(b,x+26,y+36,20,MUTED,width=420)
            y+=87
        p.text(note,x+26,592,15,MUTED,width=420)
    p.box(68,663,1464,89,fill='#efe5ce',edge='#efe5ce')
    p.text('초반 온기 · 자석   →   물건 획득   →   추가 온기   →   출구 귀환',139,690,27,'#382c43',True)
    p.text('코드가 목표 순서와 경유지를 안내하고, 실제 이동·대기 행동은 PPO 정책이 선택합니다.',78,783,20,MUTED)
    p.footer(2);pages.append(p)

    p=Page();p.header(3,'VERIFIED RESULTS & LIMITATIONS','클리어는 안정적, 유령 회피는 남은 과제',
                     '실제 시작점의 확률적 플레이로 평가 · 실패한 판도 평균 별에 포함')
    p.box(68,232,793,392)
    for x,word in [(92,'평가 대상'),(439,'성공'),(578,'별 3개'),(718,'평균 별')]:p.text(word,x,252,18,MUTED,True)
    p.line(90,293,840,293)
    data=[('Stage 1 · 새 시드 100판',metrics['stage1_new'],FG),('Stage 2 · 선택 평가 100판',metrics['stage2_selection'],GOLD),('Stage 2 · 최종 검증 200판',final,SAGE)]
    for i,(label,m,color) in enumerate(data):
        y=315+i*81
        p.text(label,92,y,20,color,True)
        p.text(pct(m['success_rate']),439,y,24,color,True)
        p.text(pct(m['three_star_rate']),578,y,24,color,True)
        p.text(number(m['mean_stars']),718,y,24,color,True)
        p.line(90,y+58,840,y+58)
    p.text('Stage 2 선택 모델은 약 45만 프레임 시점입니다.\n최종 검증은 개선 후보가 없어 보존한 같은 모델의 결과입니다.',94,566,16,MUTED,width=740)
    p.image(OUT/'game_clear.png',898,232,634,357)
    p.text('결정적 화면 예시: 95점 · 별 3개 · 피해 1회\n표의 통계는 확률적 플레이입니다.',908,603,17,MUTED)
    p.box(68,652,793,145,fill='#352a42')
    p.text('무피해 보완 실험의 판정: 목표 미달',92,671,24,PINK,True)
    p.text('약 15.4만 프레임에서 성능 보존 하한으로 중단\n새 후보는 채택하지 않고 기존 모델을 보관했습니다.',92,714,21,FG,width=740)
    p.box(898,681,634,116,fill='#352a42')
    p.text(f'최종 자석 활용 {pct(final["magnet_use_rate"])}',924,695,25,GOLD,True)
    p.text(f'평균 피해 {final["mean_damage"]:.3f}회 · 무피해 성공 {pct(final["damage_free_success_rate"])}',924,744,20,PINK)
    p.footer(3);pages.append(p)
    return pages


class NoteParser(HTMLParser):
    def __init__(self):
        super().__init__();self.blocks=[];self.active=None;self.row=[];self.cell=None;self.footer=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='div' and a.get('class')=='notef':self.footer=True
        if self.footer:return
        if tag in ['h1','h2','p','li'] and self.active is None:
            self.active=['mini' if tag=='p' and a.get('class')=='mini' else tag,'']
        if tag=='div' and a.get('class')=='script':self.active=['quote','']
        if tag=='tr':self.row=[]
        if tag in ['th','td']:self.cell=''
        if tag=='br' and self.active:self.active[1]+='\n'
    def handle_data(self,data):
        if self.footer:return
        if self.cell is not None:self.cell+=data
        elif self.active:self.active[1]+=data
    def handle_endtag(self,tag):
        if self.footer:return
        if tag in ['th','td'] and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        if tag=='tr' and self.row:self.blocks.append(('row',' | '.join(self.row)));self.row=[]
        if self.active and (tag==self.active[0] or tag=='div' and self.active[0]=='quote'
                            or tag=='p' and self.active[0]=='mini'):
            kind,text=self.active;self.active=None
            if text.strip() and not text.startswith('PRESENTER NOTES') and not text.startswith('REFERENCE'):
                self.blocks.append((kind,text.strip()))


def guide(final):
    pages=[]
    replacements={
        '別':'별',
        '별도 새 시드 100판에서도 성공률 97%, 별 3개 78%를 확인했습니다.':'최종 새 시드 200판에서 성공률 97.5%, 별 3개 74%를 확인했습니다.',
        '평균 피해는 1.17회':'평균 피해는 1.155회',
        '현재는 무피해 추가 수집·귀환을 보강하고 있으며, 미확인 결과는 성과에서 제외했습니다.':'무피해 보완은 성능 보존 하한 미달로 중단됐고, 개선 후보 대신 기존 모델을 보존했습니다.',
        'Stage 2 새 시드 평가는 선택 이후 다른 행동 난수 시드로 확인한 결과입니다.':'최종 검증은 보존한 기존 모델을 60,000~60,199번 새 시드로 확인한 결과입니다.',
        '현재 마무리 설정은':'무피해 보완 실험의 목표는',
        '학습 종료 후 선택한 모델 하나를 새 시드 200판으로 검증하도록 준비했습니다.':'학습은 약 154,308프레임에서 중단됐고 최종 200판 검증도 목표를 통과하지 못했습니다.',
        '발표 시점의 성과는 위 표로 고정했습니다. 진행 중인 마무리 실험의 최종 보고서가 확인되면 별도 근거와 함께 갱신해야 합니다.':'최종 보고서에서 final_stage2_passed는 false입니다. 추가 보완의 성공으로 발표하지 않고 현재 성과와 한계를 함께 설명합니다.',
        '현재 무피해 마무리 설정':'무피해 보완 실험 설정',
        '제작 기준일: 2026-11-01 (Asia/Seoul).':'제작 기준일: 2026-11-01 (Asia/Seoul). 최종 보완 실험 결과까지 반영했습니다.'
    }
    replacements.update({
        '이 뒤의 4쪽은 발표 준비를 위한 대본·상세 설명·근거입니다.':'뒤의 설명 자료는 발표 준비를 위한 대본·상세 설명·근거입니다.',
        '모두 실제 시작점의 확률적 100판입니다.':'모두 실제 시작점 평가입니다. Stage 1과 Stage 2 선택 평가는 확률적 100판, 최종 검증은 확률적 200판입니다.',
        'Stage 2 새 시드 100판의 실제 활용률은':'Stage 2 최종 새 시드 200판의 실제 활용률은',
    })
    allblocks=[]
    for n in range(1,5):
        source=(OUT/f'note_preview_{n:02}.html').read_text(encoding='utf-8')
        for old,new in replacements.items():source=source.replace(old,new)
        parser=NoteParser();parser.feed(source)
        blocks=[]
        for kind,text in parser.blocks:
            if '공식 PPO 문서' in text and 'Proximal Policy Optimization Algorithms' in text:
                text += '\nhttps://arxiv.org/abs/1707.06347\nhttps://stable-baselines3.readthedocs.io/en/master/modules/ppo.html'
            if '③ Stage 2 새 시드:' in text:
                text += '\n최종 검증 200판: 이 산출물 폴더의 final_stage2_evidence.json'
            if n==3 and kind=='quote':
                text='“평가는 준비된 중간 상태가 아닌 실제 시작점에서 진행했습니다. Stage 1 보관 모델은 새 시드 100판에서 성공률 96%, 별 3개 획득률 95%였습니다. Stage 2는 선택 평가에서 성공률 96%, 별 3개 82%였고, 최종 새 시드 200판에서 성공률 97.5%, 별 3개 74%였습니다. 자석 활용은 98%였지만 평균 피해는 1.155회로 유령 회피가 남은 과제입니다. 무피해 보완 실험은 성능 보존 하한 미달로 중단됐고, 개선된 후보 대신 기존 모델을 보관했습니다.”'
            if kind=='row' and text.startswith('Stage 2 · 새 시드'):
                text=f'Stage 2 · 최종 검증 200판 | {pct(final["success_rate"])} | {pct(final["three_star_rate"])} | {number(final["mean_stars"])}'
            blocks.append((kind,text))
        allblocks.append(blocks)
    allblocks.append([('h1','최종 보완 실험 · 결과와 해석'),
        ('p','실행: seed_11_20261101_040335_220183 / 종료: performance_retention_stopped'),
        ('h2','같은 선택 평가 시드에서 비교'),
        ('row','시점 | 클리어 | 별 3개 | 평균 별 | 평균 피해'),
        ('row','출발 모델 | 96% | 82% | 2.63 | 1.16'),
        ('row','54,372F | 98% | 74% | 2.45 | 1.21'),
        ('row','104,398F | 97% | 65% | 2.30 | 1.27'),
        ('row','154,308F | 98% | 73% | 2.49 | 1.17'),
        ('p','별 3개 80%·평균 별 2.5개 하한을 세 번 연속 충족하지 못해 중단했습니다. 클리어 성공률은 높았지만 별 획득이 줄었고 피해도 개선되지 않았습니다.'),
        ('h2','최종 200판의 의미'),
        ('p',f'선택 시점은 0프레임이며, 보존한 best_model의 신경망 가중치는 이전 모델과 동일함을 확인했습니다. 따라서 성공 {pct(final["success_rate"])}·별 3개 {pct(final["three_star_rate"])}·평균 별 {final["mean_stars"]:.3f}개·평균 피해 {final["mean_damage"]:.3f}회는 새로 개선된 정책의 결과가 아닙니다.'),
        ('p','이전 선택 평가 100판의 별 3개 82%와 최종 새 시드 200판의 74%는 서로 다른 평가 표본입니다. 같은 모델의 수치를 비교해 정책이 퇴보했다고 결론 내리지 않습니다. 추가 학습 후보의 하락은 위의 같은 시드 비교로 판단합니다.'),
        ('h2','원인에 대해 말할 수 있는 범위'),
        ('p','반복적으로 피해 1회를 받는 기존 행동을 보상 강화와 위상 복습만으로 안전한 행동으로 바꾸지 못했습니다. 정확히 어떤 요소가 원인인지 분리한 대조 실험은 수행하지 않았으므로 특정 보상 계수 하나를 확정 원인으로 주장하지 않습니다.'),
        ('h2','프로젝트를 여기서 정리할 때'),
        ('p','검증된 기존 모델을 채택하고 현재 성공률·별 획득률·피해량을 그대로 발표합니다. 성능 보존 장치가 열화된 후보를 채택하지 않았다는 점과 유령 회피가 남은 과제라는 점을 함께 제시합니다.'),
        ('p','근거: reports/local/experiments/stage2_safe_return_final/quick/seed_11_20261101_040335_220183/training_summary.json 및 final_verification/evaluation.json')])
    for index,blocks in enumerate(allblocks,1):
        p=Page(794,1122,PAPER);y=52
        p.text(f'발표 설명 / {index}',54,24,10,'#836440',True)
        for kind,text in blocks:
            size={'h1':28,'h2':17,'quote':14,'row':12,'mini':11}.get(kind,14)
            bold=kind in ['h1','h2']
            gap={'h1':22,'h2':13,'row':7,'quote':15}.get(kind,10)
            width=650 if kind=='quote' else 686
            lines=wrap(text,size,width,bold)
            needed=len(lines)*size*1.65+gap+(24 if kind=='quote' else 0)
            if y+needed>1055:
                p.text('까망이와 유령 저택 · 상세 설명',54,1081,10,'#75677e');pages.append(p)
                p=Page(794,1122,PAPER);y=55
                p.text('상세 설명 (계속)',54,24,10,'#836440',True)
            if kind=='quote':
                p.box(54,y,686,needed-5,fill='#f0e9f4',edge='#f0e9f4',radius=9)
                p.text(text,72,y+11,size,'#403048',width=650,spacing=1.65)
            else:
                color='#563d67' if bold else '#30283a'
                p.text(text,54,y,size,color,bold,width=width,spacing=1.65)
            y+=needed
        p.text('까망이와 유령 저택 · 상세 설명',54,1081,10,'#75677e');pages.append(p)
    return pages


if __name__=='__main__':
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    evidence=json.loads((FINAL/'final_verification/evaluation.json').read_text(encoding='utf-8'))
    final=evidence['results']['stochastic']['summary']
    metrics['stage2_final']=final
    (OUT/'final_stage2_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    deck=slides(metrics,final)
    notes=guide(final)
    targets=[('PPO_발표자료_3장.pdf',deck),('PPO_발표자료와_상세설명_전체.pdf',deck+notes)]
    for name,pages in targets:
        pages[0].im.save(OUT/name,'PDF',save_all=True,append_images=[p.im for p in pages[1:]],
                         resolution=192,quality=96,title='까망이와 유령 저택: PPO 적용',author='프로젝트 발표 자료')
        print(name,'pages',len(pages),'bytes',(OUT/name).stat().st_size)
    for i,p in enumerate(deck,1):
        p.im.resize((1600,900),Image.Resampling.LANCZOS).save(OUT/f'slide_{i:02}.png')
    for i,p in enumerate(notes,1):
        p.im.resize((794,1122),Image.Resampling.LANCZOS).save(OUT/f'note_{i:02}.png')
    (OUT/'manifest.json').write_text(json.dumps({'training_performed':False,'rl_source_modified':False,
        'presentation_pages':3,'explanation_pages':len(notes),'latest_result_included':True,
        'pdf_rendering':'Pillow, Korean Malgun font, 192 dpi'},ensure_ascii=False,indent=2),encoding='utf-8')
