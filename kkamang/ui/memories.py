"""Three immutable picture-book originals and their opaque reveal masks.

This module draws only. Completed rooms, unlocks and cinematic playback belong
to the app. In particular the third original never receives current ending state.
"""
from functools import lru_cache
import math
import pygame

from .illustration import (cat, _resident_cat_frame, resident_cat, ghost, prop, glow,
                           shelter, INK, CREAM, GOLD)
from ..paths import ASSET_DIR

SIZE = (1280, 720)
FOG = (67, 62, 78)
RESIDENT_COLORS = ((245,239,226), (195,155,111), (161,153,184))
MEMORY_ART_DIR = ASSET_DIR / 'art' / 'memories'


def _memory_id(value):
    value = int(value)
    if value not in (1, 2, 3):
        raise ValueError('memory_id must be 1, 2 or 3')
    return value


def _blanket(s,rect):
    r=pygame.Rect(rect);d=pygame.draw
    d.rect(s,(89,75,100),r.move(3,5),border_radius=24)
    d.rect(s,(195,172,203),r,border_radius=24)
    for x in range(r.x+15,r.right-10,18):d.line(s,(246,231,213),(x,r.y+6),(x,r.bottom-6),3)
    d.line(s,CREAM,(r.x+8,r.bottom-16),(r.right-8,r.bottom-16),4)


@lru_cache(maxsize=3)
def _original(memory_id):
    """Load the one immutable raster original for this memory, never a fallback."""
    memory_id=_memory_id(memory_id)
    path=MEMORY_ART_DIR / f'memory_{memory_id:02d}.png'
    if not path.is_file():
        raise FileNotFoundError(f'필수 기억 일러스트가 없습니다: {path}')
    try:
        image=pygame.image.load(str(path))
    except (OSError,pygame.error) as exc:
        raise RuntimeError(f'기억 일러스트를 읽을 수 없습니다: {path}') from exc
    width,height=image.get_size()
    # Image-generation exports can round the 16:9 height by one source pixel.
    # Keep the source PNG intact and normalise only this runtime surface.
    if abs(width*9-height*16)>16:
        raise ValueError(f'기억 일러스트는 16:9 비율이어야 합니다: {path} ({width}×{height})')
    if pygame.mask.from_surface(image,threshold=254).count()!=width*height:
        raise ValueError(f'기억 일러스트는 완전히 불투명해야 합니다: {path}')
    if image.get_size()!=SIZE:
        image=pygame.transform.smoothscale(image,SIZE)
    # An RGB backing keeps the immutable original independent of video mode.
    original=pygame.Surface(SIZE);original.blit(image,(0,0))
    return original


def memory_original(memory_id):
    """Return a copy so callers cannot alter the cached, fixed original."""
    return _original(_memory_id(memory_id)).copy()


@lru_cache(maxsize=3)
def _labels(memory_id):
    memory_id=_memory_id(memory_id);s=pygame.Surface(SIZE);s.fill((4,0,0));d=pygame.draw
    def poly(label,points):d.polygon(s,(label,0,0),points)
    if memory_id==1:
        # Towel and its warm patch; the kitten is explicitly removed below.
        poly(1,[(389,540),(470,504),(560,495),(700,495),(778,503),(862,525),
                (934,595),(988,642),(972,720),(295,720),(331,612)])
        # The open entrance and the wet path continue across the foreground.
        poly(2,[(0,0),(250,0),(255,330),(245,393),(359,480),(440,557),(535,636),(625,720),(0,720)])
        # Chair, with the grandmother's body and face deferred to the last part.
        poly(3,[(844,180),(1051,176),(1170,184),(1230,279),(1235,464),
                (1194,561),(951,567),(856,517),(830,394)])
        poly(4,[(888,94),(1050,85),(1136,161),(1133,292),(1118,351),(1079,386),
                (1110,528),(1020,553),(892,538),(855,477),(854,410),(882,338),
                (908,272),(901,200)])
        # Her open hand, and only the kitten's two little front paws.
        poly(3,[(877,355),(903,367),(893,402),(884,435),(869,448),(855,431),(858,399)])
        poly(4,[(525,402),(742,402),(746,547),(525,547)])
        poly(3,[(610,515),(631,512),(649,518),(674,515),(700,519),(700,540),(610,540)])
    elif memory_id==2:
        poly(1,[(0,0),(201,0),(221,158),(215,340),(289,402),(314,518),(344,596),(298,720),(0,720)])
        poly(2,[(209,0),(607,0),(615,163),(610,290),(584,381),(486,408),
                (371,404),(285,371),(227,316),(209,153)])
        poly(3,[(924,0),(1280,0),(1280,720),(937,720),(916,621),(905,503),
                (930,408),(934,293),(942,181)])
    else:
        poly(1,[(229,0),(1280,0),(1280,720),(1090,720),(1045,610),(1044,483),
                (969,448),(945,350),(900,252),(639,231),(610,320),(553,388),
                (334,404),(239,369)])
        poly(2,[(291,563),(460,533),(590,531),(787,540),(959,573),(1084,632),
                (1280,674),(1280,720),(0,720),(0,655),(198,619)])
        poly(3,[(600,0),(950,0),(935,120),(920,258),(911,361),(877,383),
                (610,387),(574,335),(581,241),(609,210)])
        # The painterly spectral edges extend well past the three faces. Their
        # faint tails, bright fur, orange particles and halos all wait for 12.
        poly(4,[(390,319),(1069,319),(1091,444),(1077,605),(1019,626),
                (394,620),(383,545)])
        # The final raster keeps the lower bodies and tails opaque, without
        # escaping glints. Reveal them with the blanket; all faces, upper
        # shoulders, spectral edges and floating warmth stay above this patch.
        poly(2,[(426,546),(488,532),(567,516),(622,503),(689,509),(735,509),
                (771,512),(819,514),(862,530),(933,550),(1009,575),(1052,591),
                (1043,622),(602,631),(442,616),(417,586)])
        # The ember glow inside the fireplace is distinct from escaping warmth.
        poly(3,[(720,310),(889,310),(898,345),(878,350),(729,348)])
        # Door, handle and the narrow garden opening are also final-stage clues.
        poly(4,[(0,0),(229,0),(241,335),(223,452),(176,486),(0,493)])
    return s


def memory_region_mask(memory_id):
    """RGB red channel stores label 1..4; every pixel has exactly one label."""
    return _labels(_memory_id(memory_id)).copy()


@lru_cache(maxsize=12)
def _region_mask(memory_id,region):
    # Mask and Surface are core Pygame APIs; surfarray would require NumPy even
    # in the ordinary game installation, whose only dependency is Pygame.
    return pygame.mask.from_threshold(_labels(memory_id),(region,0,0),
                                      threshold=(1,1,1,255))


@lru_cache(maxsize=12)
def _region_alpha(memory_id,region):
    alpha=pygame.Surface(SIZE,pygame.SRCALPHA)
    return _region_mask(memory_id,region).to_surface(
        surface=alpha,setcolor=(255,255,255,255),unsetcolor=(255,255,255,0))


@lru_cache(maxsize=12)
def _region(memory_id,region):
    im=pygame.Surface(SIZE,pygame.SRCALPHA);im.blit(_original(memory_id),(0,0))
    im.blit(_region_alpha(memory_id,region),(0,0),special_flags=pygame.BLEND_RGBA_MULT)
    return im


@lru_cache(maxsize=96)
def _scaled_visibility(memory_id,regions,size):
    alpha=pygame.Surface(size,pygame.SRCALPHA);alpha.fill((255,255,255,0))
    for region in regions:
        # Nearest-neighbour alpha keeps exactly the label of each target pixel.
        alpha.blit(pygame.transform.scale(_region_alpha(memory_id,region),size),(0,0))
    return alpha


@lru_cache(maxsize=1)
def _fog():
    s=pygame.Surface(SIZE);s.fill(FOG);d=pygame.draw
    # Marks are independent of the hidden original: no outlines or thumbnails.
    for y in range(13,720,37):
        for x in range(17,1280,43):
            c=(71,66,82) if (x+y)%3 else (62,58,74)
            d.arc(s,c,(x,y,29,17),0,2.6,1)
    return s


def draw_memory(surface,memory_id,regions,rect,*,previous_regions=None,
                reveal_region=None,progress=1.,item_kind=None,elapsed=0.):
    """Draw only unlocked regions. The backing fog is always fully opaque.

    ``regions`` and ``previous_regions`` are four booleans in stage order;
    ``reveal_region`` is a zero-based index. One-based label sets also work.
    Animation is local to the new region. Previously revealed pixels retain the
    same source values, and item/light effects are clipped to that new region.
    """
    def unlocked(value):
        value=list(value)
        if len(value)==4 and all(type(r) is bool for r in value):
            return {index+1 for index,opened in enumerate(value) if opened}
        return {int(r) for r in value if int(r) in (1,2,3,4)}
    memory_id=_memory_id(memory_id);regions=unlocked(regions)
    # Public UI indices are zero based; the mask's labels are one based.
    reveal_region=None if reveal_region is None else int(reveal_region)+1
    progress=max(0.,min(1.,float(progress)));scene=_fog().copy()
    old=regions if previous_regions is None else regions & unlocked(previous_regions)
    if reveal_region is None:old=regions
    for region in sorted(old):scene.blit(_region(memory_id,region),(0,0))
    pending=regions-old
    for region in sorted(pending):
        im=_region(memory_id,region)
        if progress<1.:
            im=im.copy();im.set_alpha(round(255*(progress*progress*(3-2*progress))))
        scene.blit(im,(0,0))
    if reveal_region in pending and 0.<progress<1.:
        center=_region_mask(memory_id,reveal_region).centroid()
        effect=pygame.Surface(SIZE,pygame.SRCALPHA)
        start=(640,629);end=center
        for k in range(22):
            t=(k/21.+progress*.7)%1.
            x=start[0]+(end[0]-start[0])*t;y=start[1]+(end[1]-start[1])*t-math.sin(t*math.pi)*72
            pygame.draw.circle(effect,(*GOLD,round(160*math.sin(progress*math.pi))),(round(x),round(y)),3)
        if item_kind:
            icon=pygame.Surface((88,88),pygame.SRCALPHA);prop(icon,item_kind,icon.get_rect())
            icon.set_alpha(round(220*(1-progress)));effect.blit(icon,(center[0]-44,center[1]-44))
        effect.blit(_region_alpha(memory_id,reveal_region),(0,0),special_flags=pygame.BLEND_RGBA_MULT)
        scene.blit(effect,(0,0))
    target=pygame.Rect(rect)
    if target.size!=SIZE:
        scaled=pygame.transform.smoothscale(scene,target.size)
        # Resampling must not smear an exposed edge into a locked target pixel.
        visible=old | (pending if progress>0. else set())
        clipped=pygame.Surface(target.size,pygame.SRCALPHA);clipped.blit(scaled,(0,0))
        clipped.blit(_scaled_visibility(memory_id,tuple(sorted(visible)),target.size),
                     (0,0),special_flags=pygame.BLEND_RGBA_MULT)
        scene=pygame.transform.smoothscale(_fog(),target.size)
        scene.blit(clipped,(0,0))
    surface.blit(scene,target)


def _ease(value):
    t=max(0.,min(1.,value));return t*t*(3-2*t)


def _sleeping(s,center,variant=None,scarf='default',time=0.):
    x,y=center;d=pygame.draw
    color=(53,48,66) if variant is None else RESIDENT_COLORS[variant]
    breath=math.sin(time*1.4+(variant or 0))*.7
    d.ellipse(s,INK,(x-76,y-36-breath,145,65))
    d.ellipse(s,color,(x-72,y-34-breath,137,58))
    source=pygame.Surface((160,180),pygame.SRCALPHA)
    if variant is None:
        # The playable PNG atlas has no sleeping pose. Keep this local to the
        # shared-blanket scene, with closed eyes and Kkamang's red neckerchief.
        source.blit(_resident_cat_frame(160,mood='sleep',scarf=scarf),(0,0))
        d.polygon(source,(190,74,54),[(47,108),(110,108),(98,120),(56,117)])
        d.polygon(source,(190,74,54),[(98,114),(112,116),(120,139),(102,133)])
        d.line(source,(239,135,96),(56,111),(98,113),2)
    else:resident_cat(source,(80,180),160,0.,variant)
    head=pygame.transform.smoothscale(source.subsurface((14,8,132,112)),(83,70))
    s.blit(head,(x-89,y-61))
    d.arc(s,INK,(x+12,y-15,54,42),-.8,2.3,13)
    d.arc(s,color,(x+12,y-15,54,42),-.8,2.3,8)


def _cinematic_fixtures(surface,elapsed,*,door_open=False,dawn=False):
    """Present-day story fixtures, separate from the simplified playable lobby."""
    d=pygame.draw
    # A full-height threshold aligns with the cat's existing garden walk.
    d.rect(surface,(78,61,71),(24,94,178,464),border_radius=62)
    d.rect(surface,(172,126,94),(31,101,164,450),border_radius=56)
    d.rect(surface,(57,70,83) if not dawn else (187,210,187),
           (43,113,140,427),border_radius=49)
    if door_open:
        d.rect(surface,(112,146,112) if dawn else (81,111,102),(44,451,138,89))
        d.polygon(surface,(137,97,79),[(179,116),(224,147),(224,507),(179,539)])
        d.line(surface,(196,151,107),(188,138),(188,517),3)
        d.circle(surface,GOLD,(211,350),5)
    else:
        d.rect(surface,(128,91,77),(47,117,132,420),border_radius=44)
        d.rect(surface,(162,117,89),(59,137,107,238),3,border_radius=37)
        d.rect(surface,(162,117,89),(59,393,107,127),3,border_radius=13)
        d.circle(surface,GOLD,(157,353),7)
    d.rect(surface,(198,157,113),(27,539,174,15),border_radius=4)

    # Reuse the game's stone, wood and warm light palette for the hearth.
    d.rect(surface,(130,105,105),(269,202,196,215),border_radius=13)
    for y in range(216,408,38):
        d.line(surface,(161,129,115),(275,y),(459,y),2)
    d.rect(surface,(193,147,111),(256,190,222,20),border_radius=5)
    d.rect(surface,(202,158,119),(256,403,222,24),border_radius=5)
    d.rect(surface,(49,39,49),(293,239,148,164),border_radius=62)
    d.line(surface,(112,74,66),(313,388),(418,377),12)
    d.line(surface,(142,89,68),(319,376),(413,390),9)
    d.polygon(surface,(219,129,74),[(310,385),(326,314),(349,350),(368,297),(422,385)])
    d.polygon(surface,(250,199,111),[(332,385),(346,345),(364,370),(376,331),(402,385)])
    glow(surface,(367,350),100)

    # No wall clock in the regular lobby is added or changed. This one is
    # frozen in every ordinary resolution and resumes only after liberation.
    center=(367,142)
    d.circle(surface,(83,62,69),center,35)
    d.circle(surface,(176,133,99),center,31)
    d.circle(surface,(235,217,181),center,26)
    for index in range(12):
        angle=math.tau*index/12-math.pi/2
        start=(round(center[0]+22*math.cos(angle)),round(center[1]+22*math.sin(angle)))
        end=(round(center[0]+24*math.cos(angle)),round(center[1]+24*math.sin(angle)))
        d.line(surface,INK,start,end,2)
    d.line(surface,INK,center,(354,133),3)
    d.line(surface,INK,center,(378,127),3)
    seconds=math.floor(max(0.,elapsed)) if dawn else 0
    angle=math.tau*seconds/60-math.pi/2
    tip=(round(center[0]+22*math.cos(angle)),round(center[1]+22*math.sin(angle)))
    d.line(surface,(166,91,85),center,tip,2)
    d.circle(surface,INK,center,3)


def draw_cinematic(surface,kind,elapsed,*,completed=(),cosmetics=None,released=False):
    """Silent present-day scenes; no calls can alter past memory originals."""
    allowed=('prologue','garden','liberation','basic_end','true_end')
    if kind not in allowed:raise ValueError('unknown cinematic kind')
    cosmetics=cosmetics or {};t=max(0.,float(elapsed));s=pygame.Surface(SIZE)
    all_done=range(1,13) if kind!='prologue' else ()
    dawn=kind=='true_end' or (kind=='liberation' and t>=2.2)
    shelter(s,all_done,cosmetics,time=t,before=kind=='prologue',show_actors=False,released=dawn,story_time=t)
    clock_time=max(0.,t-2.2) if kind=='liberation' else t
    _cinematic_fixtures(s,clock_time,door_open=kind!='prologue',dawn=dawn)
    d=pygame.draw;scarf=cosmetics.get('scarf','default')
    if kind=='prologue':
        walk=_ease(t/2.3);retreat=_ease((t-1.1)/1.8)
        cat(s,(80+164*walk,557),127,time=t,mood='surprised',scarf=scarf,walking=walk<1.,walk_progress=walk*2.)
        prop(s,'towel',(332,461,98,73))
        ghost(s,(429+89*retreat,455),104,t,True,0)
        glow(s,(333,514),35+round(30*walk))
    elif kind=='garden':
        # Open threshold, a path outside, and three friends leaving it clear.
        d.polygon(s,(145,164,132),[(0,554),(184,554),(420,720),(0,720)])
        step=_ease(t/2.2)
        cat(s,(297-128*step,544+24*step),130,time=t,scarf=scarf,walking=step<1.,walk_progress=step*2.)
        for variant,(x,y) in enumerate(((384,453),(476,510),(578,459))):
            ghost(s,(x+_ease(t/2.)*(variant-1)*26,y),96,t,True,variant)
    elif kind=='liberation':
        p=_ease((t-.5)/2.2)
        cat(s,(615,550),137,time=t,mood='happy',scarf=scarf)
        for variant,(x,y) in enumerate(((468,476),(801,523),(943,469))):
            body=pygame.Surface(SIZE,pygame.SRCALPHA)
            ghost(body,(x,y-8),106,t,True,variant);body.set_alpha(round(255*(1-p)));s.blit(body,(0,0))
            body=pygame.Surface(SIZE,pygame.SRCALPHA)
            resident_cat(body,(x,y),106,t,variant);body.set_alpha(round(255*p));s.blit(body,(0,0))
            for k in range(7):
                a=(k/7.+t*.22)%1.;xx=367+(x-367)*a;yy=350+(y-350)*a-math.sin(a*math.pi)*38
                if p<1.:d.circle(s,GOLD,(round(xx),round(yy)),3)
    elif kind=='basic_end':
        arrival=_ease(t/2.)
        cat(s,(225+380*arrival,546),133,time=t,mood='happy',scarf=scarf,walking=arrival<1.,walk_progress=arrival*3.)
        prop(s,'towel',(406,454,96,70))
        for variant,(x,y) in enumerate(((461,451),(785,511),(956,461))):ghost(s,(x,y),101,t,True,variant)
        prop(s,'plant',(799,429,68,82));prop(s,'music',(943,467,67,62))
    else:
        _blanket(s,(300,424,705,184))
        # Four physical bodies share the same blanket; Kkamang sleeps inside.
        _sleeping(s,(445,493),0,time=t)
        _sleeping(s,(773,493),2,time=t)
        _sleeping(s,(593,550),None,scarf,t)
        _sleeping(s,(889,553),1,time=t)
        d.arc(s,(161,153,184),(483,428,293,90),0,math.pi,11)
    if surface.get_size()!=SIZE:s=pygame.transform.smoothscale(s,surface.get_size())
    surface.blit(s,(0,0))
