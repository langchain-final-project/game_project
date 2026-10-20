"""PNG 캐릭터·소품과 정원·해방 장면용 도형·조명을 그린다.

일반 쉼터와 엔딩의 PNG 표시는 각각 lobby와 menus가 담당한다.
"""
import math
from functools import lru_cache
import pygame
from ..paths import ASSET_DIR
from .cat_sprites import sprite_frame, WALK_FRAMES
from .ghost_sprites import ghost_frame, RESIDENT_IDS

INK=(43,39,57); FUR=(53,48,66); LIGHT=(77,69,88); CREAM=(248,232,202)
SAGE=(151,181,157); GOLD=(240,194,111); ROSE=(218,151,155)

# 가구와 물건의 종류가 겹치므로 필수 물건은 ID로 고른다. 수건은 담요 원화를 재사용한다.
QUEST_RECTS = {
    'memory_01': (25, 738, 375, 337),
    'memory_02': (30, 60, 385, 310),
    'memory_03': (435, 60, 305, 310),
    'memory_04': (775, 50, 310, 325),
    'memory_05': (1100, 80, 325, 290),
    'memory_06': (50, 385, 310, 345),
    'memory_07': (388, 385, 330, 335),
    'memory_08': (740, 430, 355, 275),
    'memory_09': (1110, 375, 305, 355),
    'memory_10': (25, 738, 375, 337),
    'memory_11': (410, 728, 377, 325),
    'memory_12': (850, 705, 190, 365),
}

# 가구 시트의 잘라낼 영역은 왼쪽·위·너비·높이 순서다.
FURNITURE_RECTS = {
    'table': (35, 365, 285, 230),
    'plant': (995, 40, 215, 285),
    'shelf': (650, 40, 280, 285),
}


@lru_cache(maxsize=4)
def _png_file(filename):
    image = pygame.image.load(str(ASSET_DIR / 'art' / filename))
    return image.convert_alpha() if pygame.display.get_surface() else image

@lru_cache(maxsize=16)
def _png_source(filename, crop=None):
    image = _png_file(filename)
    if crop is not None:
        image = image.subsurface(crop)
    # 거의 투명한 외곽 흔적을 제외해 실제 그림의 크기를 맞춘다.
    bounds = image.get_bounding_rect(min_alpha=10)
    if not bounds.width or not bounds.height:
        raise ValueError(f'Empty PNG: {filename}, {crop}')
    return image.subsurface(bounds).copy()

@lru_cache(maxsize=256)
def _png_image(filename, size, crop=None):
    source = _png_source(filename, crop)
    width, height = size
    scale = min(width / source.get_width(), height / source.get_height())
    image = pygame.transform.smoothscale(source, (
        max(1, round(source.get_width() * scale)),
        max(1, round(source.get_height() * scale)),
    ))
    canvas = pygame.Surface(size, pygame.SRCALPHA)
    canvas.blit(image, image.get_rect(midbottom=(width // 2, height)))
    return canvas

def quest_prop(surface, quest, rect):
    """방 장식과 혼동하지 않도록 물건 ID에 대응하는 필수 원화를 그린다."""
    rect = pygame.Rect(rect)
    image = _png_image('Essential_item.png', rect.size, QUEST_RECTS[quest['id']])
    surface.blit(image, rect)

def cat_frame(size=100, direction='down', phase=0, mood='idle', scarf='default', *, walking=None):
    # 표정·스카프 인자는 호출 호환용이다. 걷기는 별도로 판정해 피격 중에도 유지한다.
    is_walking = mood == 'walk' if walking is None else walking
    index = int(phase * WALK_FRAMES / 8 + 1e-9) % WALK_FRAMES if is_walking else 0
    return sprite_frame(int(size), direction, is_walking, index)

@lru_cache(maxsize=2048)
def _resident_cat_frame(size=100, direction='down', phase=0, mood='idle', scarf='default', palette=None, *, walking=None, folded_ears=False):
    # 정원·해방 연출용 몸 색은 까망이 PNG 시트와 별개다.
    fur,light,belly,eye_color,rear = palette or (FUR,LIGHT,(88,79,95),GOLD,INK)
    s=pygame.Surface((160,180),pygame.SRCALPHA)
    shadow=pygame.Surface((160,180),pygame.SRCALPHA)
    pygame.draw.ellipse(shadow,(20,18,30,45),(24,153,111,16));s.blit(shadow,(0,0))
    is_walking = mood == 'walk' if walking is None else walking
    walk=math.sin(phase*math.pi/4)*9 if is_walking else 0
    bob=abs(walk)*.28
    tail=[(107,128),(133,126),(140,106+math.sin(phase*math.pi/4)*6),(131,90)]
    pygame.draw.lines(s,INK,False,tail,19);pygame.draw.lines(s,light,False,tail,11)

    for x,y,shift,col in [(52,133,walk,rear),(98,133,-walk,rear),(61,150,-walk,fur),(94,150,walk,fur)]:
        if mood=='collect' and x==94:y-=24;shift=6
        pygame.draw.ellipse(s,col,(x-10,y+shift,23,20))
        if x in (61,94):
            for a in (0,5):pygame.draw.line(s,light,(x+a,y+shift+11),(x+a,y+shift+15),1)
    pygame.draw.ellipse(s,INK,(40,82-bob,80,77));pygame.draw.ellipse(s,fur,(44,86-bob,72,68))
    pygame.draw.ellipse(s,belly,(65,113-bob,28,28))
    if direction in ('left','right'):
        offset=-10 if direction=='left' else 10
    else:offset=0

    hx=offset; hy=-bob
    if not folded_ears:
        pygame.draw.polygon(s,INK,[(29+hx,63+hy),(28+hx,13+hy),(64+hx,34+hy)])
        pygame.draw.polygon(s,INK,[(94+hx,34+hy),(133+hx,13+hy),(130+hx,66+hy)])
        pygame.draw.polygon(s,ROSE,[(36+hx,43+hy),(36+hx,24+hy),(56+hx,37+hy)])
        pygame.draw.polygon(s,ROSE,[(104+hx,37+hy),(125+hx,24+hy),(122+hx,48+hy)])
    pygame.draw.ellipse(s,INK,(21+hx,29+hy,118,86));pygame.draw.ellipse(s,fur,(25+hx,31+hy,110,79))
    pygame.draw.arc(s,light,(39+hx,35+hy,79,55),.3,2.9,3)
    if folded_ears:
        for x in (26,99):
            pygame.draw.ellipse(s,INK,(x+hx,32+hy,36,29))
            pygame.draw.ellipse(s,fur,(x+3+hx,33+hy,30,25))
            pygame.draw.arc(s,light,(x+7+hx,36+hy,23,18),math.pi,math.tau,2)
    if direction!='up':
        eyes=[(56+hx,70+hy),(103+hx,70+hy)]
        if direction=='left':eyes=[(43+hx,70+hy),(83+hx,70+hy)]
        if direction=='right':eyes=[(70+hx,70+hy),(110+hx,70+hy)]
        for x,y in eyes:
            if mood in ('sleep','happy') or (mood=='idle' and phase==7):
                pygame.draw.arc(s,eye_color,(x-10,y-5,20,12),0,math.pi,3)
            else:
                r=13 if mood=='surprised' else 11
                pygame.draw.ellipse(s,eye_color,(x-r,y-r,r*2,r*2+3))
                pygame.draw.ellipse(s,INK,(x-4,y-9,8,21))
                pygame.draw.circle(s,CREAM,(int(x+4),int(y-6)),3)
        pygame.draw.polygon(s,ROSE,[(76+hx,82+hy),(84+hx,82+hy),(80+hx,87+hy)])
        pygame.draw.arc(s,(204,186,183),(72+hx,86+hy,8,8),math.pi,2*math.pi,2)
        pygame.draw.arc(s,(204,186,183),(80+hx,86+hy,8,8),math.pi,2*math.pi,2)
        for side in (-1,1):
            for dy in (-3,4):pygame.draw.line(s,(149,134,151),(80+hx+side*35,86+hy+dy),(80+hx+side*52,84+hy+dy),2)
    else:
        for x in (60,80,100):pygame.draw.arc(s,light,(x-6,57+hy,12,24),.2,2.9,2)
    cloth={'default':SAGE,'peach':ROSE,'lavender':(181,160,210)}.get(scarf,SAGE)
    pygame.draw.polygon(s,cloth,[(47,108-bob),(110,108-bob),(98,120-bob),(56,117-bob)])
    pygame.draw.polygon(s,cloth,[(98,114-bob),(112,116-bob),(120,139-bob),(102,133-bob)])
    pygame.draw.line(s,(205,220,191),(56,111-bob),(98,113-bob),2)
    return pygame.transform.smoothscale(s,(int(size),int(size*180/160)))

def cat(surface, center, size=80, direction='down', time=0., mood='idle', scarf='default',
        *, walking=None, walk_progress=None):
    is_walking = mood == 'walk' if walking is None else walking
    if is_walking and walk_progress is not None:
        # 이동 거리를 기준으로 여섯 걷기 자세를 한 주기로 재생한다.
        phase = (int(walk_progress * WALK_FRAMES) % WALK_FRAMES) * 8 / WALK_FRAMES
    else:
        phase=(int(time * 10) % WALK_FRAMES) * 8 / WALK_FRAMES if is_walking else 0
    im=cat_frame(int(size),direction,phase,mood,scarf,walking=walking)
    surface.blit(im,im.get_rect(midbottom=(round(center[0]),round(center[1]))))

def ghost(surface, center, size=50, time=0., resident=False, variant=0, *,
          ghost_id=None, direction='down'):
    # 변형 번호는 보송·이파리·도레 순서다.
    ident = ghost_id if ghost_id is not None else RESIDENT_IDS[variant % 3]
    image = ghost_frame(int(size), ident, direction)
    surface.blit(image, image.get_rect(midbottom=(
        round(center[0]), round(center[1] + math.sin(time * 2 + variant) * 3))))

def resident_cat(surface, center, size=80, time=0., variant=0):
    """정원·해방 연출에서 고양이로 돌아온 친구를 도형으로 그린다."""
    variant%=3
    palettes=[
        ((245,239,226),(216,207,211),(255,248,233),(119,141,150),(208,199,199)),
        ((195,155,111),(224,189,141),(244,220,176),(85,121,98),(152,113,83)),
        ((161,153,184),(193,182,210),(222,208,222),(124,104,160),(122,113,145)),
    ]
    scarf=('peach','default','lavender')[variant]
    im=_resident_cat_frame(160,'down',int(time*3)%8,'happy',scarf,palettes[variant],folded_ears=variant==0).copy()
    d=pygame.draw
    if variant==0:
        for points in [[(28,71),(17,76),(27,83)],[(131,71),(143,76),(133,83)],[(65,34),(73,23),(81,33)]]:
            d.polygon(im,palettes[0][0],points)
    elif variant==1:
        d.line(im,(84,120,86),(115,37),(124,15),3)
        d.ellipse(im,SAGE,(120,12,24,12));d.line(im,(99,140,103),(124,20),(139,16),1)
        d.ellipse(im,(110,148,107),(105,19,18,10))
    else:
        for x in (67,80,93):d.line(im,(130,121,157),(x,35),(x,47),3)
        d.circle(im,GOLD,(81,125),6);d.line(im,GOLD,(86,125),(86,110),2)
    im=pygame.transform.smoothscale(im,(int(size),int(size*180/160)))
    surface.blit(im,im.get_rect(midbottom=(round(center[0]),round(center[1]))))

@lru_cache(maxsize=256)
def prop_image(kind,size=64,lit=True):
    s=pygame.Surface((100,100),pygame.SRCALPHA);d=pygame.draw
    brown=(121,85,76);wood=(173,124,95);edge=(83,63,69)
    if kind=='map':
        d.polygon(s,edge,[(12,23),(35,15),(64,22),(87,14),(87,78),(65,86),(35,79),(12,87)])
        d.polygon(s,(237,218,180),[(16,26),(35,20),(64,27),(83,20),(83,75),(65,81),(35,74),(16,81)])
        for x in (35,64):d.line(s,(188,160,128),(x,24),(x,76),2)
        for a,b in [((24,65),(31,50)),((34,46),(44,52)),((48,53),(58,42)),((62,39),(73,36))]:
            d.line(s,(153,121,117),a,b,3)
        d.circle(s,ROSE,(73,35),7);d.circle(s,CREAM,(73,35),3)
        d.line(s,(142,128,109),(22,31),(29,31),2);d.line(s,(142,128,109),(25,28),(25,35),2)
    elif kind in ('book','music','music_final'):
        d.rect(s,edge,(16,19,70,65),border_radius=5);d.rect(s,(239,220,183),(20,17,62,60),border_radius=4)
        d.line(s,brown,(49,20),(49,73),2)
        for y in (30,40,50,60):d.line(s,(180,150,126),(26,y),(43,y),2);d.line(s,(180,150,126),(55,y),(75,y),2)
        if kind in ('music','music_final'):
            d.circle(s,INK,(62,56),6);d.line(s,INK,(67,56),(67,30),3)
            d.polygon(s,(205,182,145),[(70,17),(82,29),(70,29)])
            d.circle(s,(190,147,114),(31,63),5)
            if kind=='music_final':
                d.line(s,INK,(55,66),(75,66),2);d.line(s,INK,(72,62),(72,71),2)
        else:
            d.lines(s,SAGE,False,[(57,60),(65,48),(74,52)],3)
    elif kind in ('towel','scarf','cushion'):
        cloth={'towel':(229,215,185),'scarf':SAGE,'cushion':(195,172,203)}[kind]
        d.rect(s,edge,(14,27,75,52),border_radius=13);d.rect(s,cloth,(17,24,69,49),border_radius=11)
        for x in range(24,83,9):d.line(s,(246,231,213),(x,29),(x,66),2)
        d.line(s,CREAM,(21,59),(81,59),3)
    elif kind=='footprint_photo':
        d.rect(s,edge,(10,14,80,73),border_radius=5)
        d.rect(s,(230,211,175),(14,11,72,72),border_radius=4)
        d.rect(s,(148,136,129),(21,19,58,56))
        for x,y in ((37,59),(57,36)):
            d.ellipse(s,(77,70,77),(x-5,y-3,12,10))
            for dx,dy in ((-6,-7),(0,-10),(6,-8)):d.circle(s,(77,70,77),(x+dx,y+dy),3)
    elif kind=='scarf_hook':
        d.rect(s,brown,(33,10,34,28),border_radius=7)
        d.circle(s,GOLD,(50,20),4)
        d.arc(s,GOLD,(39,22,25,31),math.pi,math.pi*2.7,6)
        d.polygon(s,ROSE,[(32,44),(66,42),(73,89),(53,81),(41,94)])
        d.line(s,CREAM,(41,49),(58,78),3)
        for x in (43,49,55):d.line(s,CREAM,(x,83),(x+4,91),1)
    elif kind=='saucer':
        d.ellipse(s,edge,(9,30,82,46));d.ellipse(s,(239,222,193),(11,26,78,43))
        d.ellipse(s,(184,153,137),(23,32,54,29),2)
        d.ellipse(s,(250,237,213),(34,39,32,17))
        for x,y in ((24,47),(77,48),(49,31),(50,63)):
            d.circle(s,ROSE,(x,y),3);d.circle(s,GOLD,(x+4,y),2)
    elif kind=='photo':
        d.rect(s,edge,(10,14,80,73),border_radius=6);d.rect(s,wood,(13,11,74,72),border_radius=5)
        d.rect(s,(116,133,147),(20,18,60,57),border_radius=2)
        d.ellipse(s,(159,173,161),(20,53,60,22))

        for x,y,col in [(34,46,(245,239,226)),(51,42,(195,155,111)),(66,49,(161,153,184))]:
            d.ellipse(s,col,(x-9,y+6,18,18))
            d.polygon(s,col,[(x-10,y),(x-9,y-12),(x-2,y-5),(x+3,y-5),(x+10,y-12),(x+11,y)])
            d.ellipse(s,col,(x-10,y-8,21,19))
            for ex in (x-4,x+4):d.line(s,INK,(ex,y-1),(ex,y+1),1)
            d.circle(s,ROSE,(x,y+4),1)
        d.line(s,(218,177,137),(20,78),(79,78),2)
    elif kind=='collar':
        d.ellipse(s,edge,(12,23,76,52),15);d.ellipse(s,SAGE,(16,22,68,47),10)
        d.rect(s,GOLD,(57,21,16,16),border_radius=3);d.rect(s,edge,(61,25,8,8),2,border_radius=1)
        d.line(s,GOLD,(49,61),(49,70),3)
        d.circle(s,brown,(49,77),12);d.circle(s,GOLD,(49,75),11)
        d.arc(s,(255,224,154),(42,69,14,15),0,math.pi,2)
        d.line(s,brown,(42,79),(57,79),2);d.circle(s,brown,(49,82),2)
        d.polygon(s,ROSE,[(25,61),(35,64),(36,79),(22,76)])
        d.line(s,CREAM,(27,67),(31,68),2)
    elif kind=='seeds':
        d.polygon(s,edge,[(32,20),(67,20),(63,35),(82,70),(75,87),(25,87),(18,70),(37,35)])
        d.polygon(s,(192,157,114),[(35,23),(64,23),(59,36),(77,70),(71,82),(29,82),(23,70),(41,36)])
        d.line(s,brown,(33,36),(67,36),4);d.line(s,brown,(48,36),(40,48),2);d.line(s,brown,(51,36),(63,46),2)
        d.ellipse(s,(236,214,167),(31,48,38,29))
        d.line(s,(94,132,86),(50,70),(50,54),3)
        d.ellipse(s,SAGE,(49,51,15,9));d.ellipse(s,(118,156,105),(36,55,15,9))
        for x,y in ((76,85),(87,79),(83,91)):d.ellipse(s,(165,118,71),(x-3,y-2,7,4))
    elif kind=='bowl':
        d.ellipse(s,(64,52,64),(12,72,77,13))
        d.polygon(s,edge,[(12,42),(88,42),(77,77),(23,77)])
        d.polygon(s,(204,176,160),[(17,44),(83,44),(73,73),(27,73)])
        d.ellipse(s,edge,(10,27,80,30));d.ellipse(s,(240,221,193),(15,30,70,22))
        d.ellipse(s,(126,100,88),(23,35,54,13))
        d.lines(s,(104,77,78),False,[(66,50),(61,59),(68,64),(62,73)],2)
        d.circle(s,CREAM,(42,62),4)
        for x,y in ((35,56),(42,53),(49,56)):d.circle(s,CREAM,(x,y),2)
    elif kind=='cord':
        d.arc(s,(229,199,151),(16,15,66,65),-.5,4.4,7)
        d.arc(s,(180,140,105),(27,28,42,43),-.6,4.3,4)
        d.lines(s,(229,199,151),False,[(74,37),(79,59),(63,73),(60,83)],6)
        d.circle(s,brown,(59,81),6)
        d.polygon(s,GOLD,[(54,81),(64,81),(69,96),(49,96)])
        for x in (53,58,63):d.line(s,(178,131,88),(x,85),(x-1,94),1)
    elif kind=='clock_hand':
        d.circle(s,(72,58,65),(48,60),12);d.circle(s,GOLD,(48,58),9)
        d.polygon(s,edge,[(43,58),(44,27),(48,9),(54,27),(53,58)])
        d.polygon(s,GOLD,[(46,55),(47,28),(49,18),(51,28),(50,55)])
        d.polygon(s,edge,[(51,55),(74,36),(88,33),(81,45),(55,63)])
        d.polygon(s,(214,167,96),[(53,57),(76,38),(82,36),(78,42),(54,60)])
        d.circle(s,CREAM,(48,58),3)
    elif kind=='handle':
        d.rect(s,edge,(24,10,32,81),border_radius=14);d.rect(s,(181,135,90),(28,13,24,75),border_radius=11)
        for y in (23,78):d.circle(s,edge,(40,y),3);d.line(s,(216,178,119),(38,y),(42,y),1)
        d.circle(s,edge,(40,43),10);d.circle(s,GOLD,(40,42),8)
        d.lines(s,edge,False,[(40,42),(47,52),(73,52),(81,45)],13)
        d.lines(s,GOLD,False,[(40,40),(48,49),(73,49),(79,44)],8)
        d.line(s,(252,220,155),(48,47),(73,47),2)
        d.circle(s,edge,(40,66),4);d.rect(s,edge,(38,65,4,7),border_radius=1)
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

def prop(surface, kind, rect, lit=True):
    rect = pygame.Rect(rect)
    if kind in FURNITURE_RECTS:
        image = _png_image('map/warmth.png', rect.size, FURNITURE_RECTS[kind])
        surface.blit(image, rect)
        return
    im=prop_image(kind,max(rect.size),lit)
    surface.blit(pygame.transform.smoothscale(im,rect.size),rect)

@lru_cache(maxsize=128)
def halo(radius,color=(239,192,110)):
    s=pygame.Surface((radius*2,radius*2),pygame.SRCALPHA)
    for r in range(radius,0,-4):pygame.draw.circle(s,(*color,int(20*(1-r/radius)**1.5)),(radius,radius),r)
    return s

def glow(surface,pos,radius=75):
    im=halo(radius);surface.blit(im,(int(pos[0]-radius),int(pos[1]-radius)))

def _released_residents(released=False,released_variants=None,*,before=False):
    """보송·이파리·도레 순서의 해방 대상 집합을 입력 변경 없이 반환한다."""
    if before:return frozenset()
    if released_variants is None:return frozenset((0,1,2) if released else ())
    return frozenset(variant for variant in released_variants if variant in (0,1,2))

def shelter(surface, completed=(), cosmetics=None,time=0.,before=False,show_actors=True,*,focus_stage=None,released=False,story_time=0.,released_variants=None,show_residents=True):
    cosmetics=cosmetics or {};done=set() if before else set(completed);d=pygame.draw
    living=_released_residents(released,released_variants,before=before)
    released=len(living)==3
    surface.fill((145,116,105) if released else (40,39,57))
    for y in range(200,720,40):
        floor=(143+(y//40%2)*3,110,100) if released else (75+(y//40%2)*3,60,69)
        d.rect(surface,floor,(0,y,1280,39))
        for x in range(-80+(y//40%2)*80,1280,160):d.line(surface,(160,125,111) if released else (87,68,73),(x,y),(x,y+38),1)
    d.rect(surface,(145,135,150) if released else (68,61,81),(0,0,1280,275))
    for x in range(0,1280,64):d.line(surface,(160,148,161) if released else (79,70,90),(x,0),(x,275),2)
    d.rect(surface,(169,128,104) if released else (124,90,80),(0,269,1280,12))
    for x in range(32,1280,64):
        for y in (40,125,210):
            wallpaper=(178,155,163) if released else (91,77,93)
            d.line(surface,wallpaper,(x-4,y),(x+4,y),1)
            d.line(surface,wallpaper,(x,y-4),(x,y+4),1)
    cord=[(x,285+int(math.sin(x/1280*math.pi)*27)) for x in range(0,1281,40)]
    d.lines(surface,(167,132,111),False,cord,2)
    for k,x in enumerate(range(60,1260,100)):
        y=285+int(math.sin(x/1280*math.pi)*27)
        col=GOLD if k+1 in done else (107,91,102)
        d.line(surface,(167,132,111),(x,y),(x,y+7),1);d.ellipse(surface,col,(x-4,y+7,8,12))
        if k+1 in done:glow(surface,(x,y+12),35)

    # 세 친구가 모두 해방된 경우에만 비 오는 밤을 새벽으로 바꾼다.
    d.rect(surface,(165,123,103),(500,62,280,211),border_radius=78)
    d.rect(surface,(181,204,207) if released else (55,67,91),(513,74,254,186),border_radius=68)
    if released:
        d.ellipse(surface,(230,199,176),(526,183,228,70))
        d.circle(surface,(255,225,165),(693,183),29)
        for x,y,w in ((555,114,59),(655,98,65)):
            d.ellipse(surface,(228,231,217),(x,y,w,13))
        for x,y in ((577,157),(609,137)):
            d.arc(surface,(116,143,151),(x,y,12,8),0,math.pi,2)
            d.arc(surface,(116,143,151),(x+10,y,12,8),0,math.pi,2)
        sunbeam=pygame.Surface((1280,720),pygame.SRCALPHA)
        d.polygon(sunbeam,(255,223,164,25),[(516,255),(766,255),(1100,620),(315,620)])
        surface.blit(sunbeam,(0,0))
    else:
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
    if show_actors:
        travel=max(0.,min(1.,story_time/3.));travel=travel*travel*(3-2*travel)
        if 8 in done:
            player=(445+170*travel,545)
            positions=((500-50*travel,450),(290+90*travel,510),(925-140*travel,510))
        elif 4 in done:
            player=(250+115*travel,545)
            positions=((490+30*travel,440),(220,488),(978,473))
        else:
            player=(190,545);positions=((490,440),(220,488),(978,473))

        towel=(445,484) if 8 in done else (390,462) if 4 in done else (345,442)
        prop(surface,'towel',(*towel,88,62))
        if show_residents and 6 in done:
            plant=(positions[1][0]+39,positions[1][1]-55)
            prop(surface,'plant',(*plant,48,59))
            for k in range(3):
                yy=plant[1]+10+(time*18+k*7)%27
                d.circle(surface,(171,204,212),(round(plant[0]-8-k*3),round(yy)),2)
        # 까망이의 앉은 PNG는 유지하고 모임에 합류하는 위치만 이동한다.
        seated_cat = _png_image('character/kkamang_wait.png', (132, 132))
        surface.blit(seated_cat, seated_cat.get_rect(midbottom=(round(player[0]), round(player[1]))))
        if show_residents:
            for variant,pos in enumerate(positions):
                if variant in living:resident_cat(surface,pos,97,time,variant)
                elif variant==0 or (variant==1 and 6 in done) or (variant==2 and 7 in done):
                    ghost(surface,pos,(97,110,90)[variant],time,True,variant)
        if show_residents and 7 in done:
            for i in range(3):
                yy=positions[2][1]-73-(time*18+i*22)%75
                xx=positions[2][0]+22+math.sin(time+i)*17
                d.circle(surface,GOLD,(int(xx),int(yy)),3);d.line(surface,GOLD,(xx+3,yy),(xx+3,yy-13),2)

    overlay=pygame.Surface((1280,720),pygame.SRCALPHA);overlay.fill((255,221,161,8) if released else (27,29,45,max(0,40-len(done)*3)));surface.blit(overlay,(0,0))
    glow(surface,(615,440),230)
