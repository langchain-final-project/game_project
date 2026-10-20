"""기억 원화 3장의 분할 공개와 온기 복원 효과를 그린다.

현재 엔딩 상태를 과거 기억 원화에 섞지 않으며 진행 상태는 앱이 관리한다.
"""
from functools import lru_cache
import math
import pygame

from .illustration import prop, GOLD
from ..paths import ASSET_DIR

SIZE = (1280, 720)
FOG = (67, 62, 78)
MEMORY_ART_DIR = ASSET_DIR / 'art' / 'memories'


def _memory_id(value):
    value = int(value)
    if value not in (1, 2, 3):
        raise ValueError('memory_id must be 1, 2 or 3')
    return value


@lru_cache(maxsize=3)
def _original(memory_id):
    """기억별 원화만 읽으며 누락되거나 잘못된 원화를 다른 그림으로 대체하지 않는다."""
    memory_id=_memory_id(memory_id)
    path=MEMORY_ART_DIR / f'memory_{memory_id:02d}.png'
    if not path.is_file():
        raise FileNotFoundError(f'필수 기억 일러스트가 없습니다: {path}')
    try:
        image=pygame.image.load(str(path))
    except (OSError,pygame.error) as exc:
        raise RuntimeError(f'기억 일러스트를 읽을 수 없습니다: {path}') from exc
    width,height=image.get_size()
    # 16:9 내보내기의 한 픽셀 오차만 허용하고 원본 파일 대신 실행 중 표면만 맞춘다.
    if abs(width*9-height*16)>16:
        raise ValueError(f'기억 일러스트는 16:9 비율이어야 합니다: {path} ({width}×{height})')
    if pygame.mask.from_surface(image,threshold=254).count()!=width*height:
        raise ValueError(f'기억 일러스트는 완전히 불투명해야 합니다: {path}')
    if image.get_size()!=SIZE:
        image=pygame.transform.smoothscale(image,SIZE)
    # 영상 모드와 무관하게 원화 캐시를 고정하기 위해 불투명 RGB 표면에 담는다.
    original=pygame.Surface(SIZE);original.blit(image,(0,0))
    return original

def memory_original(memory_id):
    """호출자가 원화 캐시를 바꿀 수 없도록 사본을 반환한다."""
    return _original(_memory_id(memory_id)).copy()

@lru_cache(maxsize=3)
def _labels(memory_id):
    memory_id=_memory_id(memory_id);s=pygame.Surface(SIZE);s.fill((4,0,0));d=pygame.draw
    def poly(label,points):d.polygon(s,(label,0,0),points)
    if memory_id==1:
        # 수건을 먼저 공개하고 아기 고양이의 얼굴과 몸은 마지막 영역에 남긴다.
        poly(1,[(322,555),(413,532),(524,528),(696,526),(828,535),(915,579),
                (946,630),(900,659),(709,682),(492,669),(346,624),(305,584)])

        poly(2,[(0,0),(251,0),(258,321),(250,401),(294,479),(330,548),
                (313,579),(358,623),(423,655),(494,720),(0,720)])
        # 의자는 먼저 공개하되 할머니의 몸과 얼굴은 마지막까지 숨긴다.
        poly(3,[(1018,184),(1167,179),(1239,212),(1280,274),(1280,569),
                (1169,597),(1071,574),(1052,480),(1087,376),(1052,303)])
        poly(4,[(845,103),(1000,88),(1080,144),(1085,231),(1120,289),
                (1147,356),(1122,443),(1109,513),(1137,566),(1063,594),
                (948,598),(855,555),(852,497),(885,419),(868,371),
                (844,297),(856,229),(836,174)])

        poly(3,[(833,417),(858,416),(874,428),(863,455),(843,479),
                (822,491),(811,479),(818,452),(810,440)])
        poly(4,[(470,396),(711,396),(717,549),(691,578),(514,579),(470,535)])
        poly(3,[(606,554),(622,551),(636,559),(635,573),(606,574)])
        poly(3,[(650,551),(670,548),(680,560),(674,571),(650,569)])
    elif memory_id==2:
        # 보송·스카프 다음 이파리의 창가·씨앗 주머니를 공개한다.
        poly(1,[(0,0),(206,0),(218,178),(216,338),(290,385),(337,403),
                (364,460),(360,550),(348,593),(287,650),(0,681)])
        poly(2,[(216,0),(605,0),(605,164),(598,293),(571,374),(492,410),
                (378,409),(302,386),(239,348),(220,264)])
        # 도레·피아노·할머니를 먼저 분리하고 함께 쓰는 찻상은 마지막에 공개한다.
        poly(3,[(924,0),(1280,0),(1280,720),(937,720),(916,621),(910,501),
                (923,408),(930,293),(936,181)])
    else:
        poly(1,[(230,0),(1280,0),(1280,720),(1123,720),(1058,563),
                (1028,504),(996,455),(953,397),(896,355),(812,333),
                (720,327),(650,315),(600,330),(540,350),(435,375),
                (344,405),(235,371)])
        # 몸 아래쪽의 꼬리와 발은 유령의 정체가 드러나기 전에 공개한다.
        poly(2,[(110,576),(214,511),(330,504),(497,506),(640,510),
                (770,520),(910,520),(1041,526),(1176,601),(1244,637),
                (1280,720),(0,720),(0,632)])
        poly(3,[(592,0),(991,0),(986,131),(972,258),(975,372),
                (900,411),(741,388),(560,333),(555,267),(572,206),(598,195)])
        # 유령 윤곽과 떠다니는 온기는 마지막 영역이다. 방의 조명과 구분해 미리 노출하지 않는다.
        poly(4,[(270,482),(307,441),(357,401),(367,326),(417,291),
                (494,280),(575,296),(656,282),(722,292),(794,302),
                (853,315),(920,339),(967,384),(1025,439),(1060,476),
                (1062,510),(1101,535),(1060,559),(985,554),(927,544),
                (842,540),(767,529),(683,525),(589,519),(499,517),
                (432,514),(355,519),(304,506),(270,503)])
        # 얼굴 아래의 불투명한 발만 담요 영역에 돌려놓아 유령의 어깨 윤곽을 숨긴다.
        poly(2,[(451,492),(470,488),(488,491),(502,498),(499,507),
                (472,508),(453,502)])
        poly(2,[(548,487),(564,485),(580,489),(587,497),(580,505),
                (551,505),(544,499)])
        poly(2,[(601,493),(613,490),(626,495),(631,504),(626,510),
                (602,508),(597,503)])
        poly(2,[(677,504),(691,503),(713,509),(725,517),(719,524),
                (685,522),(677,515)])
        poly(2,[(755,513),(771,510),(787,516),(796,525),(789,532),
                (759,529),(752,523)])
        # 온기 구슬을 숨겨도 벽난로 안의 불씨는 보이게 한다.
        poly(3,[(762,274),(907,274),(912,302),(871,311),(806,307),(760,301)])
        # 문·손잡이·정원 입구도 마지막 단계의 단서다.
        poly(4,[(0,0),(229,0),(241,335),(223,452),(176,495),(0,498)])
    return s

def memory_region_mask(memory_id):
    """빨강 채널에 영역 번호 1~4를 저장한다. 각 픽셀은 한 영역에만 속한다."""
    return _labels(_memory_id(memory_id)).copy()

@lru_cache(maxsize=12)
def _region_mask(memory_id,region):
    # 일반 실행에 NumPy가 필요하지 않도록 Pygame의 기본 마스크 API만 사용한다.
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
        # 최근접 배율로 알파 마스크를 맞춰 목표 픽셀의 영역 번호가 섞이지 않게 한다.
        alpha.blit(pygame.transform.scale(_region_alpha(memory_id,region),size),(0,0))
    return alpha

@lru_cache(maxsize=1)
def _fog():
    s=pygame.Surface(SIZE);s.fill(FOG);d=pygame.draw
    # 안개 무늬는 숨겨진 원화와 독립적이어서 윤곽이나 축소본을 노출하지 않는다.
    for y in range(13,720,37):
        for x in range(17,1280,43):
            c=(71,66,82) if (x+y)%3 else (62,58,74)
            d.arc(s,c,(x,y,29,17),0,2.6,1)
    return s

def draw_memory(surface,memory_id,regions,rect,*,previous_regions=None,
                reveal_region=None,progress=1.,item_kind=None,elapsed=0.):
    """열린 영역만 공개하고 나머지는 불투명한 안개로 덮는다.

    regions와 previous_regions는 단계 순서의 불리언 네 개 또는 1부터 시작하는 영역 집합이다.
    reveal_region은 0부터 시작하며 새 영역에만 효과를 적용하고 이미 열린 원화는 유지한다.
    """
    def unlocked(value):
        value=list(value)
        if len(value)==4 and all(type(r) is bool for r in value):
            return {index+1 for index,opened in enumerate(value) if opened}
        return {int(r) for r in value if int(r) in (1,2,3,4)}
    memory_id=_memory_id(memory_id);regions=unlocked(regions)
    # 화면의 공개 영역 인덱스는 0부터, 마스크 번호는 1부터 시작한다.
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
        # 부드러운 축소의 경계색이 잠긴 픽셀로 번지지 않도록 마스크를 다시 적용한다.
        visible=old | (pending if progress>0. else set())
        clipped=pygame.Surface(target.size,pygame.SRCALPHA);clipped.blit(scaled,(0,0))
        clipped.blit(_scaled_visibility(memory_id,tuple(sorted(visible)),target.size),
                     (0,0),special_flags=pygame.BLEND_RGBA_MULT)
        scene=pygame.transform.smoothscale(_fog(),target.size)
        scene.blit(clipped,(0,0))
    surface.blit(scene,target)
