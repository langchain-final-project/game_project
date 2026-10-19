"""Three immutable picture-book originals and their opaque reveal masks.

This module draws only. Completed rooms, unlocks and cinematic playback belong
to the app. In particular the third original never receives current ending state.
"""
from functools import lru_cache
import math
import pygame

from .illustration import (cat, cat_frame, resident_cat, ghost, prop, glow,
                           shelter, INK, CREAM, GOLD, SAGE, ROSE)

SIZE = (1280, 720)
FOG = (67, 62, 78)
RESIDENT_COLORS = ((245,239,226), (195,155,111), (161,153,184))


def _memory_id(value):
    value = int(value)
    if value not in (1, 2, 3):
        raise ValueError('memory_id must be 1, 2 or 3')
    return value


def _chair(s):
    d=pygame.draw
    for x in (706,823):d.line(s,(81,60,67),(x,463),(x-5,548),12)
    d.rect(s,(93,67,76),(688,260,161,236),border_radius=52)
    d.rect(s,(164,125,127),(701,271,135,203),border_radius=44)
    d.rect(s,(195,153,143),(705,441,130,37),border_radius=17)
    for x in (687,825):d.rect(s,(111,78,80),(x,374,26,104),border_radius=12)
    for x in (735,800):d.circle(s,(133,96,107),(x,325),4)


def _hearth(s, night=False):
    d=pygame.draw
    stone=(129,107,115) if night else (173,135,119)
    d.rect(s,stone,(363,110,252,310),border_radius=15)
    d.rect(s,(199,157,128),(348,102,282,26),border_radius=6)
    d.rect(s,(213,170,133),(345,401,291,27),border_radius=6)
    d.rect(s,(53,43,55),(400,189,178,214),border_radius=77)
    for x,y in ((378,153),(582,153),(379,282),(582,282)):
        d.line(s,(219,179,139),(x-8,y),(x+8,y),2)
        d.line(s,(219,179,139),(x,y-8),(x,y+8),2)
    d.circle(s,(229,188,124),(489,156),17,2)
    d.polygon(s,(229,188,124),[(489,139),(504,164),(474,164)],2)
    if night:
        d.lines(s,(82,65,85),False,[(482,136),(491,151),(484,158),(500,174)],4)
        d.lines(s,(233,181,118),False,[(471,156),(478,149),(482,156)],2)
        d.ellipse(s,(177,111,91),(434,370,107,19))
        d.polygon(s,(228,155,91),[(468,377),(481,346),(490,366),(507,350),(516,377)])
    else:
        d.polygon(s,(225,141,80),[(420,390),(439,304),(467,354),(495,278),(552,390)])
        d.polygon(s,(250,203,116),[(444,389),(462,342),(481,369),(495,322),(526,389)])
        glow(s,(486,348),133)


def _room(memory_id):
    """The chair, hearth, window and rug have identical positions in all three."""
    night=memory_id==3;afternoon=memory_id==2;d=pygame.draw
    s=pygame.Surface(SIZE)
    s.fill((66,66,86) if night else (146,119,128) if afternoon else (98,86,110))
    for x in range(30,1280,72):
        d.line(s,(90,82,102) if night else (172,143,144),(x,0),(x,370),2)
        for y in range(32,370,84):
            d.circle(s,(128,111,128),(x+32,y),2)
    d.rect(s,(85,70,78) if night else (139,104,91),(0,363,1280,357))
    for y in range(370,720,44):
        d.line(s,(111,88,91) if night else (166,126,103),(0,y),(1280,y),2)
        for x in range((y//44%2)*90,1280,180):d.line(s,(104,82,87),(x,y),(x,y+44),1)
    d.rect(s,(185,139,111),(0,352,1280,15))
    # Entrance, always in the same house and at the same angle.
    d.rect(s,(69,57,74),(44,65,177,325),border_radius=71)
    d.rect(s,(129,99,95),(54,75,157,303),border_radius=64)
    if memory_id in (1,3):
        d.polygon(s,(43,54,74),[(66,124),(150,93),(150,366),(66,366)])
        d.polygon(s,(160,120,91),[(150,93),(204,123),(204,366),(150,366)])
        prop(s,'handle',(156,232,40,65))
    else:prop(s,'handle',(157,244,40,65))
    # Framed window and the matching curtain tie.
    d.rect(s,(185,142,112),(989,47,249,274),border_radius=70)
    sky=(43,53,83) if night else (183,207,201) if afternoon else (69,80,109)
    d.rect(s,sky,(1002,60,223,248),border_radius=59)
    if afternoon:
        d.circle(s,(250,225,169),(1161,116),32)
        d.ellipse(s,(225,231,209),(1027,116,102,17))
    else:
        for i in range(28 if night else 14):
            x=1015+(i*47)%190;y=83+(i*59)%200
            d.line(s,(138,157,186),(x,y),(x-9,y+21),2)
        if night:d.lines(s,(231,224,193),False,[(1139,77),(1115,123),(1145,114),(1126,160)],4)
    d.line(s,(185,142,112),(1114,60),(1114,309),9)
    d.line(s,(185,142,112),(1003,184),(1225,184),8)
    for x in (970,1213):
        d.polygon(s,(135,112,142),[(x,56),(x+37,56),(x+23,233),(x+39,330),(x-10,330),(x+7,233)])
        d.line(s,GOLD,(x+4,231),(x+25,235),5)
    _hearth(s,night);_chair(s)
    d.ellipse(s,(55,45,60),(226,482,810,188))
    d.ellipse(s,(133,112,142),(238,475,787,183))
    d.ellipse(s,(205,169,159),(252,488,756,156),3)
    return s


def _grandmother(s, center, piano=False):
    """A small, open-handed seated pose; never a dialogue portrait."""
    x,y=center;d=pygame.draw
    d.polygon(s,(98,119,124),[(x-37,y-119),(x+31,y-119),(x+55,y),(x-63,y)])
    d.polygon(s,(186,154,139),[(x-29,y-116),(x+30,y-116),(x+39,y-41),(x-34,y-41)])
    d.ellipse(s,(230,220,197),(x-33,y-188,66,72))
    for dx,dy,r in ((-22,-162,18),(0,-179,26),(23,-160,18)):
        d.circle(s,(208,204,202),(x+dx,y+dy),r)
    d.ellipse(s,(225,181,151),(x-24,y-164,48,43))
    for dx in (-10,10):d.arc(s,(106,81,83),(x+dx-5,y-149,10,7),0,math.pi,2)
    d.arc(s,(150,103,99),(x-6,y-139,14,10),math.pi,math.pi*2,2)
    # Her hand rests beside her, leaving the kitten room to approach.
    hand=(x+75,y-67) if piano else (x-79,y-43)
    d.line(s,(130,143,140),(x+22 if piano else x-26,y-105),hand,17)
    d.ellipse(s,(229,186,153),(hand[0]-13,hand[1]-5,29,13))
    for dx in (-28,15):d.ellipse(s,(75,63,73),(x+dx,y-3,31,15))


def _piano(s):
    d=pygame.draw
    d.rect(s,(69,55,69),(866,309,344,187),border_radius=11)
    d.rect(s,(108,77,84),(877,318,322,156),border_radius=7)
    d.rect(s,(235,219,184),(885,418,303,43),border_radius=3)
    for x in range(887,1185,19):d.line(s,(109,88,95),(x,419),(x,460),2)
    for i,x in enumerate(range(895,1172,19)):
        if i%7 not in (2,6):d.rect(s,(58,49,64),(x,419,10,24))
    for x in (882,1175):d.rect(s,(69,55,69),(x,480,15,82),border_radius=5)
    prop(s,'music',(1015,309,98,87))


def _paw(s,x,y,color=(104,95,104),scale=1.):
    d=pygame.draw
    d.ellipse(s,color,(int(x-6*scale),int(y-3*scale),int(14*scale),int(11*scale)))
    for dx,dy in ((-7,-7),(0,-11),(7,-8)):
        d.circle(s,color,(int(x+dx*scale),int(y+dy*scale)),max(1,int(3*scale)))


def _blanket(s,rect):
    r=pygame.Rect(rect);d=pygame.draw
    d.rect(s,(89,75,100),r.move(3,5),border_radius=24)
    d.rect(s,(195,172,203),r,border_radius=24)
    for x in range(r.x+15,r.right-10,18):d.line(s,(246,231,213),(x,r.y+6),(x,r.bottom-6),3)
    d.line(s,CREAM,(r.x+8,r.bottom-16),(r.right-8,r.bottom-16),4)


@lru_cache(maxsize=3)
def _original(memory_id):
    memory_id=_memory_id(memory_id);s=_room(memory_id);d=pygame.draw
    if memory_id==1:
        for i in range(5):_paw(s,172+i*35,397+i*30,(90,87,103),.85)
        _grandmother(s,(770,483))
        prop(s,'towel',(279,531,196,126))
        glow(s,(373,601),93)
        resident_cat(s,(367,569),115,0.,0)
        prop(s,'book',(524,420,87,84))
        d.lines(s,SAGE,False,[(577,462),(591,449),(601,453)],2)
        # A low garden sign repeats the guidebook's route motif.
        d.line(s,(155,114,94),(251,349),(251,404),6)
        d.polygon(s,SAGE,[(224,333),(264,333),(278,345),(264,357),(224,357)])
    elif memory_id==2:
        prop(s,'scarf_hook',(265,258,94,110))
        resident_cat(s,(310,566),142,0.,0)
        _piano(s);_grandmother(s,(829,477),True)
        prop(s,'plant',(1140,221,96,105));prop(s,'seeds',(1210,307,62,61))
        resident_cat(s,(1086,315),125,0.,1)
        resident_cat(s,(1050,589),143,0.,2)
        prop(s,'table',(537,523,274,142))
        prop(s,'saucer',(602,527,105,52));prop(s,'cup',(623,494,68,68))
        sun=pygame.Surface(SIZE,pygame.SRCALPHA)
        d.polygon(sun,(255,232,174,21),[(1006,308),(1221,308),(1009,695),(719,695)])
        s.blit(sun,(0,0))
    else:
        # The original is the storm, even after the current-day true ending.
        d.polygon(s,(187,163,176),[(718,314),(748,320),(744,439),(717,447),(703,398)])
        for x in (467,499):d.line(s,(211,183,151),(x,69),(x,89),3)
        d.circle(s,(166,129,111),(489,69),34);d.circle(s,(215,196,172),(489,69),27)
        d.line(s,INK,(489,69),(489,48),3);d.line(s,INK,(489,69),(473,78),3)
        d.line(s,(126,105,113),(302,315),(302,395),6)
        d.line(s,(126,105,113),(281,397),(324,397),6)
        prop(s,'music_final',(259,255,87,88))
        for variant,(x,y) in enumerate(((320,593),(463,602),(606,593))):
            resident_cat(s,(x,y),154,0.,variant)
            # The spectral edge follows this body's own shoulder and tail.
            # There is no second ghost character beside the living cat.
            sheen=pygame.Surface(SIZE,pygame.SRCALPHA)
            d.polygon(sheen,(195,215,231,92),[(x+8,y-143),(x+52,y-117),(x+61,y-48),(x+44,y-34),(x+29,y-43),(x+12,y-31)])
            d.lines(sheen,(219,225,238,220),False,[(x+29,y-150),(x+52,y-128),(x+61,y-48),(x+44,y-34),(x+29,y-43)],3)
            s.blit(sheen,(0,0))
            for j in range(3):
                px=x+48+j*11;py=y-170-j*23
                d.lines(s,(238,177,100),False,[(x+35,y-98),(px-7,py+20),(px,py)],2)
                d.circle(s,GOLD,(px,py),7 if j==1 else 5)
                d.circle(s,CREAM,(px-1,py-2),2)
        _blanket(s,(236,535,481,118))
        for x in (222,717):d.arc(s,(184,158,186),(x-21,583,43,41),-.6,2.8,9)
    # A paper-book border belongs to the original, rather than a changing state.
    d.rect(s,(213,184,145),(13,13,1254,694),3,border_radius=13)
    return s


def memory_original(memory_id):
    """Return a copy so callers cannot alter the cached, fixed original."""
    return _original(_memory_id(memory_id)).copy()


@lru_cache(maxsize=3)
def _labels(memory_id):
    memory_id=_memory_id(memory_id);s=pygame.Surface(SIZE);s.fill((4,0,0));d=pygame.draw
    def poly(label,points):d.polygon(s,(label,0,0),points)
    if memory_id==1:
        poly(2,[(0,0),(283,0),(310,195),(272,330),(315,425),(302,603),(177,672),(0,720)])
        poly(3,[(532,235),(862,222),(913,391),(870,560),(612,598),(526,481)])
        poly(3,[(306,539),(433,536),(448,583),(347,602),(300,577)])
        poly(1,[(259,578),(473,574),(526,629),(841,673),(1280,657),(1280,720),(0,720),(0,680),(230,640)])
        # All of the young cat except the little front paws waits for region 4.
        poly(4,[(285,407),(440,403),(451,529),(428,548),(305,546),(277,486)])
    elif memory_id==2:
        poly(1,[(0,0),(394,0),(426,267),(399,411),(453,584),(395,720),(0,720)])
        poly(2,[(952,0),(1280,0),(1280,471),(1147,494),(1007,455),(969,316),(938,204)])
        poly(3,[(787,263),(928,278),(975,414),(1099,476),(1280,477),(1280,720),(852,720),(770,578),(804,446)])
    else:
        poly(1,[(692,0),(1280,0),(1280,720),(922,720),(857,549),(696,524),(671,329),(713,200)])
        poly(2,[(203,526),(348,520),(419,531),(535,522),(661,528),(748,567),(770,720),(119,720),(173,631)])
        poly(3,[(0,0),(692,0),(711,201),(669,327),(710,487),(633,539),(447,515),(208,525),(0,643)])
        # Faces, dissolving contours and every orange warmth particle are one
        # last-stage island. Even their soft silhouettes never appear early.
        poly(4,[(218,357),(279,332),(359,347),(429,327),(509,340),(581,326),(676,349),(699,419),(699,551),(637,563),(560,551),(486,561),(409,550),(334,561),(233,549),(218,466)])
        # The dying flame itself is a stage-11 clue, beside (not underneath)
        # the final warmth trails. This island contains none of those trails.
        poly(3,[(425,338),(505,333),(515,350),(512,394),(426,394)])
        # The garden threshold and handle also wait for the final stage.
        poly(4,[(30,58),(140,41),(233,84),(243,243),(222,402),(72,419),(27,362)])
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
    if variant is None:source.blit(cat_frame(160,mood='sleep',scarf=scarf),(0,0))
    else:resident_cat(source,(80,180),160,0.,variant)
    head=pygame.transform.smoothscale(source.subsurface((14,8,132,112)),(83,70))
    s.blit(head,(x-89,y-61))
    d.arc(s,INK,(x+12,y-15,54,42),-.8,2.3,13)
    d.arc(s,color,(x+12,y-15,54,42),-.8,2.3,8)


def draw_cinematic(surface,kind,elapsed,*,completed=(),cosmetics=None,released=False):
    """Silent present-day scenes; no calls can alter past memory originals."""
    allowed=('prologue','garden','liberation','basic_end','true_end')
    if kind not in allowed:raise ValueError('unknown cinematic kind')
    cosmetics=cosmetics or {};t=max(0.,float(elapsed));s=pygame.Surface(SIZE)
    all_done=range(1,13) if kind!='prologue' else ()
    dawn=kind=='true_end' or (kind=='liberation' and t>=2.2)
    shelter(s,all_done,cosmetics,time=t,before=kind=='prologue',show_actors=False,released=dawn,story_time=t)
    d=pygame.draw;scarf=cosmetics.get('scarf','default')
    if kind=='prologue':
        walk=_ease(t/2.3);retreat=_ease((t-1.1)/1.8)
        cat(s,(80+164*walk,557),127,time=t,mood='surprised',scarf=scarf,walking=walk<1.,walk_progress=walk*2.)
        prop(s,'towel',(332,461,98,73))
        ghost(s,(429+89*retreat,455),104,t,True,0)
        glow(s,(333,514),35+round(30*walk))
    elif kind=='garden':
        # Open threshold, a path outside, and three friends leaving it clear.
        d.polygon(s,(91,124,108),[(40,83),(123,79),(123,263),(40,263)])
        d.polygon(s,(145,164,132),[(0,465),(180,465),(420,720),(0,720)])
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
                a=(k/7.+t*.22)%1.;xx=367+(x-367)*a;yy=253+(y-253)*a-math.sin(a*math.pi)*38
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
