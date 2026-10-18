"""Original code-native picture-book assets. No simulation state is mutated here.
All characters use one rig; legs, tail, gaze and scarf are independent layers.
"""
import math
from functools import lru_cache
import pygame

INK=(43,39,57); FUR=(53,48,66); LIGHT=(77,69,88); CREAM=(248,232,202)
SAGE=(151,181,157); GOLD=(240,194,111); ROSE=(218,151,155)

@lru_cache(maxsize=2048)
def cat_frame(size=100, direction='down', phase=0, mood='idle', scarf='default', *, walking=None):
    s=pygame.Surface((160,180),pygame.SRCALPHA)
    shadow=pygame.Surface((160,180),pygame.SRCALPHA)
    pygame.draw.ellipse(shadow,(20,18,30,45),(24,153,111,16));s.blit(shadow,(0,0))
    is_walking = mood == 'walk' if walking is None else walking
    walk=math.sin(phase*math.pi/4)*9 if is_walking else 0
    bob=abs(walk)*.28
    tail=[(107,128),(133,126),(140,106+math.sin(phase*math.pi/4)*6),(131,90)]
    pygame.draw.lines(s,INK,False,tail,19);pygame.draw.lines(s,LIGHT,False,tail,11)
    # Four independently placed paws: alternating pairs move in opposite phases.
    for x,y,shift,col in [(52,133,walk,INK),(98,133,-walk,INK),(61,150,-walk,FUR),(94,150,walk,FUR)]:
        if mood=='collect' and x==94:y-=24;shift=6
        pygame.draw.ellipse(s,col,(x-10,y+shift,23,20))
        if x in (61,94):
            for a in (0,5):pygame.draw.line(s,LIGHT,(x+a,y+shift+11),(x+a,y+shift+15),1)
    pygame.draw.ellipse(s,INK,(40,82-bob,80,77));pygame.draw.ellipse(s,FUR,(44,86-bob,72,68))
    pygame.draw.ellipse(s,(88,79,95),(65,113-bob,28,28))
    if direction in ('left','right'):
        offset=-10 if direction=='left' else 10
    else:offset=0
    # Ears and head, same proportions across portraits and field sprites.
    hx=offset; hy=-bob
    pygame.draw.polygon(s,INK,[(29+hx,63+hy),(28+hx,13+hy),(64+hx,34+hy)])
    pygame.draw.polygon(s,INK,[(94+hx,34+hy),(133+hx,13+hy),(130+hx,66+hy)])
    pygame.draw.polygon(s,ROSE,[(36+hx,43+hy),(36+hx,24+hy),(56+hx,37+hy)])
    pygame.draw.polygon(s,ROSE,[(104+hx,37+hy),(125+hx,24+hy),(122+hx,48+hy)])
    pygame.draw.ellipse(s,INK,(21+hx,29+hy,118,86));pygame.draw.ellipse(s,FUR,(25+hx,31+hy,110,79))
    pygame.draw.arc(s,LIGHT,(39+hx,35+hy,79,55),.3,2.9,3)
    if direction!='up':
        eyes=[(56+hx,70+hy),(103+hx,70+hy)]
        if direction=='left':eyes=[(43+hx,70+hy),(83+hx,70+hy)]
        if direction=='right':eyes=[(70+hx,70+hy),(110+hx,70+hy)]
        for x,y in eyes:
            if mood in ('sleep','happy') or (mood=='idle' and phase==7):
                pygame.draw.arc(s,GOLD,(x-10,y-5,20,12),0,math.pi,3)
            else:
                r=13 if mood=='surprised' else 11
                pygame.draw.ellipse(s,GOLD,(x-r,y-r,r*2,r*2+3))
                pygame.draw.ellipse(s,INK,(x-4,y-9,8,21))
                pygame.draw.circle(s,CREAM,(int(x+4),int(y-6)),3)
        pygame.draw.polygon(s,ROSE,[(76+hx,82+hy),(84+hx,82+hy),(80+hx,87+hy)])
        pygame.draw.arc(s,(204,186,183),(72+hx,86+hy,8,8),math.pi,2*math.pi,2)
        pygame.draw.arc(s,(204,186,183),(80+hx,86+hy,8,8),math.pi,2*math.pi,2)
        for side in (-1,1):
            for dy in (-3,4):pygame.draw.line(s,(149,134,151),(80+hx+side*35,86+hy+dy),(80+hx+side*52,84+hy+dy),2)
    else:
        for x in (60,80,100):pygame.draw.arc(s,LIGHT,(x-6,57+hy,12,24),.2,2.9,2)
    cloth={'default':SAGE,'peach':ROSE,'lavender':(181,160,210)}.get(scarf,SAGE)
    pygame.draw.polygon(s,cloth,[(47,108-bob),(110,108-bob),(98,120-bob),(56,117-bob)])
    pygame.draw.polygon(s,cloth,[(98,114-bob),(112,116-bob),(120,139-bob),(102,133-bob)])
    pygame.draw.line(s,(205,220,191),(56,111-bob),(98,113-bob),2)
    return pygame.transform.smoothscale(s,(int(size),int(size*180/160)))

def cat(surface, center, size=80, direction='down', time=0., mood='idle', scarf='default',
        *, walking=None, walk_progress=None):
    is_walking = mood == 'walk' if walking is None else walking
    if is_walking and walk_progress is not None:
        # 32 cached poses per cycle, driven by distance rather than wall time.
        phase = (int(walk_progress * 32) % 32) / 4
    else:
        phase=int(time*(12 if is_walking else 3))%8
    im=cat_frame(int(size),direction,phase,mood,scarf,walking=walking)
    surface.blit(im,im.get_rect(midbottom=(round(center[0]),round(center[1]))))

def ghost(surface, center, size=50, time=0., resident=False, variant=0):
    s=pygame.Surface((100,110),pygame.SRCALPHA)
    col=((246,225,190),(190,217,182),(202,190,222))[variant%3] if resident else (170,161,200)
    pygame.draw.ellipse(s,(31,26,44,35),(15,91,70,12))
    pygame.draw.circle(s,col,(50,43),31)
    pygame.draw.polygon(s,col,[(19,43),(81,43),(86,88),(72,82),(60,92),(48,83),(35,92),(18,86)])
    for x in (38,63):
        pygame.draw.ellipse(s,INK,(x-3,41,7,12))
        if resident:pygame.draw.circle(s,(226,169,154),(x-6,59),5)
    pygame.draw.arc(s,INK,(44,53,13,10),math.pi,2*math.pi,2)
    if resident and variant==1:
        pygame.draw.line(s,SAGE,(50,18),(52,2),3);pygame.draw.ellipse(s,SAGE,(51,0,18,9))
    s=pygame.transform.smoothscale(s,(size,int(size*1.1)))
    surface.blit(s,s.get_rect(midbottom=(int(center[0]),int(center[1]+math.sin(time*2+variant)*3))))

@lru_cache(maxsize=256)
def prop_image(kind,size=64,lit=True):
    s=pygame.Surface((100,100),pygame.SRCALPHA);d=pygame.draw
    brown=(121,85,76);wood=(173,124,95);edge=(83,63,69)
    if kind in ('book','map','music'):
        d.rect(s,edge,(16,19,70,65),border_radius=5);d.rect(s,(239,220,183),(20,17,62,60),border_radius=4)
        d.line(s,brown,(49,20),(49,73),2)
        for y in (30,40,50,60):d.line(s,(180,150,126),(26,y),(43,y),2);d.line(s,(180,150,126),(55,y),(75,y),2)
        if kind=='music':d.circle(s,INK,(62,56),6);d.line(s,INK,(67,56),(67,30),3)
    elif kind in ('towel','scarf','cushion'):
        d.rect(s,edge,(14,27,75,52),border_radius=13);d.rect(s,(195,172,203),(17,24,69,49),border_radius=11)
        for x in range(24,83,9):d.line(s,(229,207,216),(x,29),(x,66),2)
        d.line(s,CREAM,(21,59),(81,59),3)
    elif kind=='plant':
        d.line(s,(104,135,104),(50,60),(50,17),4)
        for x,y in [(27,20),(50,12),(29,38),(50,32)]:d.ellipse(s,SAGE,(x,y,25,15))
        d.polygon(s,brown,[(29,56),(75,56),(68,85),(37,85)]);d.rect(s,(199,151,117),(27,53,50,8),border_radius=3)
    elif kind in ('tea','cup'):
        d.ellipse(s,wood,(14,76,74,9));d.arc(s,CREAM,(60,36,25,30),-1.6,1.6,6)
        d.rect(s,(237,210,169),(25,35,41,41),border_radius=11);d.ellipse(s,(111,80,76),(27,32,37,10))
        d.arc(s,(220,207,189),(36,7,20,23),0,2.8,3)
    elif kind in ('frame','window'):
        d.rect(s,wood,(13,11,75,79),border_radius=7);d.rect(s,(65,69,94),(20,18,61,65),border_radius=3)
        d.circle(s,GOLD,(61,37),12)
        if kind=='window':d.line(s,wood,(50,18),(50,84),5);d.line(s,wood,(20,51),(82,51),5)
        else:
            d.ellipse(s,CREAM,(33,49,30,23))
            for x,y in [(29,43),(40,36),(53,36),(64,43)]:d.circle(s,CREAM,(x,y),5)
    elif kind=='garden':
        d.rect(s,wood,(23,12,55,78),border_radius=24);d.rect(s,(59,79,72),(30,20,40,64),border_radius=18);d.circle(s,GOLD,(63,57),4)
    elif kind=='shelf':
        d.rect(s,edge,(8,9,84,82),border_radius=5);d.rect(s,wood,(12,13,76,73),border_radius=3)
        for row in (20,54):
            for x,c in zip(range(18,81,11),[SAGE,ROSE,GOLD,(153,157,190),SAGE,ROSE]):d.rect(s,c,(x,row,8,25),border_radius=1)
            d.rect(s,brown,(13,row+26,74,5))
    elif kind=='table':
        d.rect(s,edge,(19,57,10,32),border_radius=3);d.rect(s,edge,(72,57,10,32),border_radius=3)
        d.rect(s,brown,(9,27,84,40),border_radius=9);d.rect(s,wood,(9,20,84,38),border_radius=9)
        d.line(s,(206,158,117),(18,28),(81,28),2)
    else:
        c=GOLD if lit else (104,94,105)
        d.arc(s,c,(35,10,30,35),0,math.pi,5);d.polygon(s,brown,[(29,35),(71,35),(80,85),(20,85)])
        d.polygon(s,c,[(35,40),(65,40),(71,78),(29,78)])
        if lit:d.ellipse(s,CREAM,(43,45,17,27))
    return pygame.transform.smoothscale(s,(int(size),int(size)))

def prop(surface,kind,rect,lit=True):
    rect=pygame.Rect(rect);im=prop_image(kind,max(rect.size),lit)
    surface.blit(pygame.transform.smoothscale(im,rect.size),rect)

@lru_cache(maxsize=128)
def halo(radius,color=(239,192,110)):
    s=pygame.Surface((radius*2,radius*2),pygame.SRCALPHA)
    for r in range(radius,0,-4):pygame.draw.circle(s,(*color,int(20*(1-r/radius)**1.5)),(radius,radius),r)
    return s

def glow(surface,pos,radius=75):
    im=halo(radius);surface.blit(im,(int(pos[0]-radius),int(pos[1]-radius)))

# Home is constructed from the same props and character rig as the field.
def shelter(surface, completed=(), cosmetics=None,time=0.,before=False,focus_stage=None,show_actors=True):
    cosmetics=cosmetics or {};done=set() if before else set(completed);d=pygame.draw
    surface.fill((40,39,57))
    for y in range(200,720,40):
        d.rect(surface,(75+(y//40%2)*3,60,69),(0,y,1280,39))
        for x in range(-80+(y//40%2)*80,1280,160):d.line(surface,(87,68,73),(x,y),(x,y+38),1)
    d.rect(surface,(68,61,81),(0,0,1280,275))
    for x in range(0,1280,64):d.line(surface,(79,70,90),(x,0),(x,275),2)
    d.rect(surface,(124,90,80),(0,269,1280,12))
    for x in range(32,1280,64):
        for y in (40,125,210):
            d.line(surface,(91,77,93),(x-4,y),(x+4,y),1)
            d.line(surface,(91,77,93),(x,y-4),(x,y+4),1)
    cord=[(x,285+int(math.sin(x/1280*math.pi)*27)) for x in range(0,1281,40)]
    d.lines(surface,(167,132,111),False,cord,2)
    for k,x in enumerate(range(60,1260,100)):
        y=285+int(math.sin(x/1280*math.pi)*27)
        col=GOLD if k+1 in done else (107,91,102)
        d.line(surface,(167,132,111),(x,y),(x,y+7),1);d.ellipse(surface,col,(x-4,y+7,8,12))
        if k+1 in done:glow(surface,(x,y+12),35)

    # Rain outside a framed window.
    d.rect(surface,(165,123,103),(500,62,280,211),border_radius=78)
    d.rect(surface,(55,67,91),(513,74,254,186),border_radius=68)
    d.circle(surface,(233,218,179),(693,116),26)
    for i in range(22):
        x=525+(i*73)%222;y=85+(i*47+int(time*45))%160
        d.line(surface,(108,126,155),(x,y),(x-4,y+12),1)
    d.line(surface,(165,123,103),(640,73),(640,262),8);d.line(surface,(165,123,103),(514,175),(767,175),7)
    rug={'default':(145,111,135),'stars':(128,114,158),'leaves':(113,145,129)}.get(cosmetics.get('rug'),(145,111,135))
    d.ellipse(surface,(40,33,47),(290,426,700,182));d.ellipse(surface,rug,(300,420,680,175));d.ellipse(surface,(202,176,164),(315,433,650,148),3)
    if cosmetics.get('rug')=='stars':
        for x in range(370,970,85):d.circle(surface,CREAM,(x,500),3)
    elif cosmetics.get('rug')=='leaves':
        for x in range(370,970,85):d.ellipse(surface,SAGE,(x,492,17,8))
    # A permanent visible object per successful chapter, not a changing baked background.
    kinds=['towel','map','frame','book','scarf','plant','music','tea','window','cushion','music','garden']
    places=[(135,300),(280,300),(390,300),(890,310),(1050,300),(148,448),(1008,445),(800,453),(1090,153),(434,477),(731,324),(60,160)]
    for n,(kind,(x,y)) in enumerate(zip(kinds,places),1):
        if n in done:
            if n in (1,4,6,8,10,11):prop(surface,'table',(x-17,y+26,112,90))
            prop(surface,kind,(x,y,76,76))
            if kind=='frame' and cosmetics.get('frame','default')!='default':
                color=SAGE if cosmetics['frame']=='leaves' else (181,160,210)
                d.rect(surface,color,(x+10,y+8,58,61),3,border_radius=5)
            glow(surface,(x+38,y+30),90)
            if n==focus_stage:
                for k in range(5):
                    px=x+38+math.sin(time*2+k)*43;py=y+36+math.cos(time*2+k)*35
                    d.circle(surface,GOLD,(int(px),int(py)),2)
        else:
            d.ellipse(surface,(74,65,81),(x+10,y+53,55,9))
    if show_actors:
        # Resident activities persist after restoration and remain presentation-only.
        if 1 in done:
            d.line(surface,CREAM,(474,420),(457+math.sin(time*2)*5,422),5)
        if 4 in done:
            prop(surface,'book',(455,385,60,55))
            d.line(surface,CREAM,(481,397),(493+math.sin(time*1.8)*4,400),2)
        if 6 in done:
            prop(surface,'cup',(239,452,35,35))
            for k in range(3):
                yy=473+(time*22+k*7)%27;d.circle(surface,(171,204,212),(237-k*4,int(yy)),2)
        if 8 in done:
            for k in range(3):d.arc(surface,(213,191,182),(822+k*7,420-int(time*8+k*5)%20,12,18),0,2.5,1)
        if 11 in done:
            for xx in (477,218,977):d.circle(surface,GOLD,(xx,370+int(math.sin(time*3)*5)),3)
        if 12 in done:
            d.rect(surface,(202,205,175),(83,175,29,45),border_radius=12)
    prop(surface,'shelf',(18,292,123,145));prop(surface,'shelf',(1135,302,128,149))
    if show_actors:
        cat(surface,(615,532),132,time=time,mood='happy' if 10 in done else 'idle',scarf=cosmetics.get('scarf','default'))
        ghost(surface,(490,440),97,time,True,0)
        if 6 in done:ghost(surface,(220,488),84,time,True,1)
        if 7 in done:ghost(surface,(978,473),90,time,True,2)
        if 7 in done:
            for i in range(3):
                yy=400-(time*18+i*22)%75;xx=1000+math.sin(time+i)*17
                d.circle(surface,GOLD,(int(xx),int(yy)),3);d.line(surface,GOLD,(xx+3,yy),(xx+3,yy-13),2)
    # Consistent warm light; restoration progressively brightens the scene.
    overlay=pygame.Surface((1280,720),pygame.SRCALPHA);overlay.fill((27,29,45,max(0,40-len(done)*3)));surface.blit(overlay,(0,0))
    glow(surface,(615,440),230)
