"""Capture real SDL renders and the shared character rig; uses temporary saves only.
Review tooling: python -m pip install -r requirements-review.txt
"""
import os,sys,json,tempfile,time
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pygame
from PIL import Image
from kkamang.app import GameApp
from kkamang.ui.illustration import cat,prop
from kkamang.ui.theme import text,CREAM,INK,panel

def main():
 out=ROOT/'evidence';out.mkdir(exist_ok=True)
 timings={}
 with tempfile.TemporaryDirectory() as tmp:
  app=GameApp(save_path=Path(tmp)/'preview.json',headless=True,no_audio=True)
  def snap(name):
   app.elapsed=2.3;app.draw();pygame.image.save(app.canvas,str(out/f'{name}.png'))
  snap('title')
  app.start_prologue();app.dialogue_time=99;snap('prologue')
  app.transition('home');snap('home_before')
  routes=json.loads((ROOT/'validation/reference_routes.json').read_text())['routes']
  for stage in range(1,13):
   app.stage=stage;app.prepare(stage);snap(f'briefing_{stage:02}')
   app.start_run();snap(f'room_{stage:02}')
   route=next(r for r in routes if r['level_id']==f'stage_{stage:02}' and r['target']==14)
   for action in route['actions']:result=app.core.step(action)
   assert result.outcome['status']=='COMPLETE'
   app.finish_run(result)
   if stage in (1,7,12):snap(f'result_{stage:02}')
   app.start_memory(stage);app.dialogue_time=99;snap(f'memory_{stage:02}')
  app.campaign.set_cosmetic('scarf','peach');app.campaign.set_cosmetic('rug','leaves');app.campaign.set_cosmetic('frame','stars')
  for scene in ('home','rooms','album','appearance','settings','tutorial'):
   app.transition(scene);snap(scene+'_after')
   app.draw();start=time.perf_counter()
   for _ in range(20):app.draw()
   timings[scene]=round((time.perf_counter()-start)*1000/20,2)
  app.start_ending();app.dialogue_time=99;snap('ending')
  # Side by side actual renders, not painted mockups.
  board=pygame.Surface((1280,1080));board.fill((35,32,47))
  scenes=[('title','시작 화면'),('home_after','복원된 쉼터'),('room_07','음악실 탐사'),('memory_07','같은 캐릭터의 대화'),('album_after','기억 앨범'),('appearance_after','목도리 · 러그 · 액자')]
  for i,(name,title) in enumerate(scenes):
   x=(i%2)*640;y=(i//2)*360
   im=pygame.image.load(str(out/f'{name}.png'));board.blit(pygame.transform.smoothscale(im,(640,360)),(x,y))
  pygame.image.save(board,str(out/'review_board.png'))
  frames=[]
  for frame in range(24):
   canvas=pygame.Surface((960,540));canvas.fill((47,42,61))
   text(canvas,app.assets,'같은 까망이 · 4방향 걷기와 표정',(30,20),28,CREAM,True)
   for i,(direction,label) in enumerate([('down','앞'),('left','왼쪽'),('right','오른쪽'),('up','뒤')]):
    panel(canvas,(20+i*235,76,215,253),fill=(75,62,79),border=(161,130,108))
    cat(canvas,(126+i*235,286),163,direction,time=frame/12,mood='walk')
    text(canvas,app.assets,label,(126+i*235,300),18,CREAM,anchor='center')
   for i,(mood,label) in enumerate([('idle','대기 · 깜빡임'),('happy','기쁨'),('collect','수집'),('surprised','놀람')]):
    cat(canvas,(126+i*235,492),113,time=frame/12,mood=mood,scarf='peach')
    text(canvas,app.assets,label,(126+i*235,515),17,CREAM,anchor='center')
   frames.append(Image.frombytes('RGB',canvas.get_size(),pygame.image.tobytes(canvas,'RGB')))
  frames[0].save(out/'cat_animation.gif',save_all=True,append_images=frames[1:],duration=83,loop=0)
  app.close()
 (ROOT/'validation/render_timings.json').write_text(json.dumps({'kind':'local dummy SDL CPU draw timings; not target-device FPS','mean_draw_ms':timings},indent=2))
 print('Captured actual screens and animation.',timings)
if __name__=='__main__':main()
