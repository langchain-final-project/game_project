"""PNG cat animation and decorative illustrations; simulation state is read-only."""
import math
from functools import lru_cache
import pygame
from ..paths import ASSET_DIR
from .cat_sprites import sprite_frame, WALK_FRAMES
from .ghost_sprites import ghost_frame, RESIDENT_IDS

INK=(43,39,57); FUR=(53,48,66); LIGHT=(77,69,88); CREAM=(248,232,202)
SAGE=(151,181,157); GOLD=(240,194,111); ROSE=(218,151,155)

# Quest art is selected by ID: furniture and quests can share a kind, and the
# two music quests have different drawings. The towel reuses the blanket art.
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

# warmth.png에서 가구를 잘라낼 영역입니다.
# 순서: 왼쪽 위치, 위쪽 위치, 가로 길이, 세로 길이
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
    # Ignore almost invisible export noise outside the painted object.
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
    """Draw a mandatory item without confusing its kind with room decoration."""
    rect = pygame.Rect(rect)
    image = _png_image('Essential_item.png', rect.size, QUEST_RECTS[quest['id']])
    surface.blit(image, rect)


def cat_frame(size=100, direction='down', phase=0, mood='idle', scarf='default', *, walking=None):
    # Keep the existing drawing API. Mood/scarf do not select art until those
    # variants exist; a hurt cat can still walk and retain its separate shield.
    is_walking = mood == 'walk' if walking is None else walking
    index = int(phase * WALK_FRAMES / 8 + 1e-9) % WALK_FRAMES if is_walking else 0
    return sprite_frame(int(size), direction, is_walking, index)

def cat(surface, center, size=80, direction='down', time=0., mood='idle', scarf='default',
        *, walking=None, walk_progress=None):
    is_walking = mood == 'walk' if walking is None else walking
    if is_walking and walk_progress is not None:
        # Six PNG poses per cycle, still driven by actual travel distance.
        phase = (int(walk_progress * WALK_FRAMES) % WALK_FRAMES) * 8 / WALK_FRAMES
    else:
        phase=(int(time * 10) % WALK_FRAMES) * 8 / WALK_FRAMES if is_walking else 0
    im=cat_frame(int(size),direction,phase,mood,scarf,walking=walking)
    surface.blit(im,im.get_rect(midbottom=(round(center[0]),round(center[1]))))

def ghost(surface, center, size=50, time=0., resident=False, variant=0, *,
          ghost_id=None, direction='down'):
    # Existing resident variants already follow Bosong/Ipari/Dore order.
    ident = ghost_id if ghost_id is not None else RESIDENT_IDS[variant % 3]
    image = ghost_frame(int(size), ident, direction)
    surface.blit(image, image.get_rect(midbottom=(
        round(center[0]), round(center[1] + math.sin(time * 2 + variant) * 3))))

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

def prop(surface, kind, rect, lit=True):
    rect = pygame.Rect(rect)
    if kind in FURNITURE_RECTS:
        # assets/art/map/warmth.png에서 해당 가구만 가져옵니다.
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

# Home keeps the room, lights and residents without furniture/item displays.
def shelter(surface, completed=(), cosmetics=None,time=0.,before=False,show_actors=True):
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
    if show_actors:
        # 로비에서는 앉아 있는 까망이 PNG를 발 위치에 맞춰 표시합니다.
        seated_cat = _png_image('character/kkamang_wait.png', (132, 132))
        surface.blit(seated_cat, seated_cat.get_rect(midbottom=(615, 532)))
        ghost(surface,(490,440),97,time,True,0)
        if 6 in done:
            ghost(surface,(220,488),110,time,True,1)  # B 유령만 기존보다 10픽셀 크게 표시합니다.
        if 7 in done:ghost(surface,(978,473),90,time,True,2)
        if 7 in done:
            for i in range(3):
                yy=400-(time*18+i*22)%75;xx=1000+math.sin(time+i)*17
                d.circle(surface,GOLD,(int(xx),int(yy)),3);d.line(surface,GOLD,(xx+3,yy),(xx+3,yy-13),2)
    # Consistent warm light; restoration progressively brightens the scene.
    overlay=pygame.Surface((1280,720),pygame.SRCALPHA);overlay.fill((27,29,45,max(0,40-len(done)*3)));surface.blit(overlay,(0,0))
    glow(surface,(615,440),230)
