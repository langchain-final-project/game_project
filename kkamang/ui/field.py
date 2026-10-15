"""Readable, layered room rendering. Art never changes collision or timing."""
import math
import pygame
from .theme import text, wrapped, panel, CREAM, INK, NIGHT, PAPER, GOLD, SAGE
from .illustration import cat, ghost, prop, glow


def draw_field(surface,assets,snapshot,rect,elapsed=0.,previous=None,interpolation=1.,cosmetics=None,facing='down',mood='idle'):
    s=snapshot;r=pygame.Rect(rect);tile=min(r.w/16,r.h/9);ox=r.x+(r.w-tile*16)/2;oy=r.y+(r.h-tile*9)/2
    cosmetics=cosmetics or {};d=pygame.draw
    def box(p):return pygame.Rect(round(ox+p[0]*tile),round(oy+p[1]*tile),math.ceil(tile),math.ceil(tile))
    def foot(p):return (ox+(p[0]+.5)*tile,oy+(p[1]+.95)*tile)
    tasks=[]
    # Thin floor seams, broad continuous walls; collision grid is still exact.
    for y,row in enumerate(s['grid']):
        for x,c in enumerate(row):
            rr=box((x,y))
            if c=='.':
                color=(130+(x+y)%3*3,107+(x+y)%3*2,103+(x+y)%3*2)
                d.rect(surface,color,rr);d.line(surface,(145,121,113),rr.topleft,rr.topright,1)
                if (x+y)%4==0:d.line(surface,(121,99,98),(rr.x+8,rr.bottom-12),(rr.right-7,rr.bottom-12),1)
            else:
                d.rect(surface,(78,69,87),rr)
                if y in (0,8):
                    d.line(surface,(100,82,98),(rr.x+tile/2,rr.y+6),(rr.x+tile/2,rr.bottom-6),1)
                    d.line(surface,(167,127,106),(rr.x,rr.bottom-5),(rr.right,rr.bottom-5),3)
                # Furnish wall cells only: never decorate an empty walkable cell as blocked.
                if y>0 and y<8 and x not in (0,15):
                    kind=('shelf','plant','table')[(x+y+s['stage'])%3]
                    tasks.append((rr.bottom,lambda rr=rr,kind=kind:prop(surface,kind,rr.inflate(-2,8))))
                elif y==0 and x in (3,8,12):prop(surface,'window',rr.inflate(-5,-4))
                elif y==0 and x in (5,10):prop(surface,'lantern',rr.inflate(-10,-9));glow(surface,rr.center,65)
    # Rugs are flat decorative floor coverings, away from every object.
    for y,row in enumerate(s['grid']):
        for x,c in enumerate(row):
            if c=='.' and (x+y*3+s['stage'])%23==0:
                rr=box((x,y)).inflate(-8,-12);d.rect(surface,(149,117,139),rr,border_radius=9);d.rect(surface,(191,158,163),rr.inflate(-5,-5),1,border_radius=6)
                for dx in range(5,rr.w-4,6):
                    d.line(surface,(191,158,163),(rr.x+dx,rr.y-3),(rr.x+dx,rr.y+2),1)
                    d.line(surface,(191,158,163),(rr.x+dx,rr.bottom-2),(rr.x+dx,rr.bottom+3),1)
    ex=box(s['exit']);d.rect(surface,(145,174,151),ex.inflate(-7,-7),border_radius=12)
    prop(surface,'garden',ex.inflate(-9,-4))
    for sc in s['shortcuts']:
        gate=box(sc['gate']);term=box(sc['terminal'])
        if not sc['opened']:
            d.rect(surface,(91,72,82),gate.inflate(-7,-2),border_radius=10)
            for k in range(3):d.line(surface,GOLD,(gate.x+15+k*12,gate.y+8),(gate.x+15+k*12,gate.bottom-8),2)
        else:d.rect(surface,SAGE,gate.inflate(-7,-7),2,border_radius=10)
        d.circle(surface,SAGE,term.center,int(tile*.29));text(surface,assets,sc['id'],term.center,18,INK,True,'center')
    veil=pygame.Surface(r.size,pygame.SRCALPHA);veil.fill((28,25,42,20));surface.blit(veil,r)
    # Next tile and route remain visible: the art change is not a visibility nerf.
    for g in s['ghosts']:
        points=[box(p).center for p in g['route']]
        d.lines(surface,(169,148,184),True,points,1)
        nextbox=box(g['next_position']).inflate(-7,-7);d.rect(surface,(210,185,228),nextbox,2,border_radius=9)
        for k in range(g['countdown']):d.circle(surface,(235,207,237),(nextbox.centerx-5+k*10,nextbox.centery),3)
    for item in s['items']:
        if item['active']:
            rr=box(item['position']);cx,cy=rr.center;cy+=math.sin(elapsed*3+cx)*2
            glow(surface,(cx,cy),int(tile*.6))
            d.circle(surface,GOLD,(int(cx),int(cy)),int(tile*(.16 if item['value']==1 else .22)))
            d.circle(surface,CREAM,(int(cx-2),int(cy-3)),int(tile*.09))
            if item['value']==3:text(surface,assets,'+3',(cx,cy+12),14,CREAM,True,'midtop')
    q=s['quest']
    if not q['collected']:
        rr=box(q['position']);d.rect(surface,(238,213,164),rr.inflate(-4,-4),2,border_radius=12)
        prop(surface,q['kind'],rr.inflate(-10,-10));glow(surface,rr.center,60)
        d.circle(surface,GOLD,(rr.right-9,rr.y+8),7);text(surface,assets,'!',(rr.right-9,rr.y+7),11,INK,True,'center')
    for i,g in enumerate(s['ghosts']):
        pos=list(g['position'])
        if previous and i<len(previous['ghosts']):
            old=previous['ghosts'][i]['position'];pos=[old[k]+(pos[k]-old[k])*interpolation for k in (0,1)]
        pt=foot(pos)
        def draw_patrol(pt=pt,ident=g['id']):
            ghost(surface,pt,int(tile*.75),elapsed)
            text(surface,assets,ident,(pt[0],pt[1]-tile*.9),12,CREAM,True,'midbottom')
        tasks.append((pt[1],draw_patrol))
    pos=list(s['position']);moving=False
    if previous:
        old=previous['position'];moving=old!=pos
        pos=[old[k]+(pos[k]-old[k])*interpolation for k in (0,1)]
    pt=foot(pos)
    cm='surprised' if s['protection'] else 'walk' if moving and mood=='idle' else mood
    tasks.append((pt[1],lambda:cat(surface,pt,int(tile*.87),facing,elapsed,cm,cosmetics.get('scarf','default'))))
    for _,draw in sorted(tasks,key=lambda t:t[0]):draw()
    if s['protection']:d.ellipse(surface,(225,210,240),box(s['position']).inflate(-4,-5),2)
    # A compact contextual interaction prompt, drawn above scenery.
    label=None
    if s['position']==s['exit']:label='E 귀환' if q['collected'] and s['carried']>=s['target'] else '물건 + 온기 필요'
    elif s['position']==q['position'] and not q['collected']:label='E '+q['name']
    elif any(i['active'] and i['position']==s['position'] for i in s['items']):label='E 온기 회수'
    elif any(not g['opened'] and g['terminal']==s['position'] for g in s['shortcuts']):label='E 샛문 열기'
    if label:
        p=foot(s['position']);ww=assets.font(14).size(label)[0]+20;rr=pygame.Rect(p[0]-ww/2,p[1]-tile-23,ww,26)
        rr.clamp_ip(r);d.rect(surface,NIGHT,rr,border_radius=9);text(surface,assets,label,rr.center,14,CREAM,False,'center')


def draw_game(surface,assets,snapshot,stage_story,elapsed=0.,previous=None,interpolation=1.,toast='',cosmetics=None,facing='down',mood='idle'):
    s=snapshot;surface.fill((36,34,49));d=pygame.draw
    text(surface,assets,'THE SLEEPLESS MANOR',(26,16),12,(177,160,177))
    text(surface,assets,f'{s["stage"]:02d}  {s["name"]}',(25,38),27,CREAM,True)
    for i in range(3):
        c=(208,148,157) if i<s['courage'] else (81,71,88)
        d.circle(surface,c,(350+i*31,52),9);d.circle(surface,c,(362+i*31,52),9);d.polygon(surface,c,[(341+i*31,53),(371+i*31,53),(356+i*31,71)])
    q=s['quest'];panel(surface,(492,21,405,58),fill=(69,61,78),border=(118,102,113),radius=14)
    prop(surface,q['kind'],(505,29,42,42));text(surface,assets,q['name']+('  회수 완료' if q['collected'] else '  찾아오기'),(558,39),19,CREAM)
    text(surface,assets,f'온기  {s["carried"]} / {s["target"]}',(925,40),22,GOLD,True)
    remaining=s['remaining_ticks']*.3;text(surface,assets,f'{remaining:04.1f} 초',(1245,40),25,ROSE if remaining<6 else CREAM,True,'topright')
    panel(surface,(19,94,1002,575),fill=(77,63,73),border=(149,118,103),radius=18)
    draw_field(surface,assets,s,(31,106,978,550),elapsed,previous,interpolation,cosmetics,facing,mood)
    panel(surface,(1039,94,221,575),fill=PAPER,border=(182,148,116),radius=18)
    text(surface,assets,'오늘 되찾을 기억',(1058,115),18,INK,True)
    prop(surface,q['kind'],(1098,154,102,102),q['collected'])
    text(surface,assets,q['name'],(1149,269),21,INK,True,'center')
    wrapped(surface,assets,q['restoration'],(1060,303,178,104),17,INK,5)
    d.line(surface,(203,177,144),(1059,416),(1240,416))
    text(surface,assets,'조금 더 탐험하면',(1060,433),17,INK,True)
    wrapped(surface,assets,'온기 10점 귀환\n수집 배지와 꾸미기 해금\n\n무피격 완료\n조용한 발걸음 배지',(1060,465,175,139),15,INK,5)
    text(surface,assets,'큰 온기는 순찰을 조심!',(1060,631),13,(131,91,100))
    text(surface,assets,toast or '방향키 / WASD 이동   ·   E 회수 / 귀환   ·   Space 대기   ·   Esc 일시정지',(28,689),16,CREAM)

ROSE=(235,151,156)
