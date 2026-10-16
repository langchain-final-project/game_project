"""Run the desktop game. RL imports can use kkamang.env without this entrypoint."""
import argparse
import os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='까망이와 잠 못 드는 저택')
    parser.add_argument('--save-path',type=Path,help='별도 세이브 경로 (검증/시연용)')
    parser.add_argument('--headless',action='store_true',help='창 없는 렌더 검사')
    parser.add_argument('--no-audio',action='store_true')
    parser.add_argument('--frames',type=int,help='검사할 프레임 수')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--dev-stage',type=int,choices=range(1,13),help='개발용 방 바로 시작; 임시 세이브로 격리')
    parser.add_argument('--dev-screen',choices=['title','home','rooms','briefing','story','album','appearance','settings','likes'],help='개발용 화면 바로 보기')
    parser.add_argument('--dev-likes', choices=['overview','choice','happy_all','sad','ghost_a','ghost_b','ghost_c','normal'], help='호감도/엔딩 시연; 임시 점수와 세이브만 사용')
    args=parser.parse_args()
    import tempfile
    sandbox = tempfile.TemporaryDirectory(prefix='kkamang-preview-') if args.dev_stage or args.dev_screen or args.dev_likes else None
    if sandbox:
        args.save_path=Path(sandbox.name)/'preview.json'

    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
    if args.headless:
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
    from kkamang.app import GameApp
    app=GameApp(save_path=args.save_path,headless=args.headless,no_audio=args.no_audio,seed=args.seed)
    if sandbox:
        app.campaign.data['completed']=list(range(1,13))
        app.campaign.data['photos']=list(range(1,13))
        app.campaign.data['intro_seen']=True
        app.campaign.data['tutorial_seen']=True
        app.stage=args.dev_stage or 7
        if args.dev_likes:
            from kkamang.likes_preview import configure_preview
            configure_preview(app, args.dev_likes)
        elif args.dev_screen=='story':
            app.start_memory(app.stage,replay=True)
            app.dialogue_time=99.
        elif args.dev_screen=='briefing':app.prepare(app.stage)
        elif args.dev_screen:app.transition(args.dev_screen)
        else:app.start_run()
    app.run(max_frames=args.frames or (3 if args.headless else None),screenshot=args.screenshot)
    if sandbox:sandbox.cleanup()

if __name__=='__main__':
    main()
