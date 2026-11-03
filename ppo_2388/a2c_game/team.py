"""원본과 같은 환경에서 새 PPO 모델 준비·검사·학습·관람을 연결합니다."""
import argparse
import json
from pathlib import Path

from .paths import PROJECT_ROOT
from .runtime import resolve_run


SNAPSHOT = PROJECT_ROOT / 'shared_snapshot'


def resolve_ppo_run(value):
    run = resolve_run(value)
    state = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    if state.get('algorithm') != 'PPO':
        raise ValueError('PPO 실행 폴더를 지정하세요. shared_snapshot은 원본 A2C 기록입니다.')
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description='PPO 2,388 · 원본과 같은 전12맵 환경')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare', help='임의 가중치·빈 Adam·0스텝 PPO 모델 준비')
    prepare_parser.add_argument('--config', type=Path, default=SNAPSHOT / 'config.json')
    check_parser = sub.add_parser('check', help='새 PPO 모델·입력·전12맵 연결 검사 (학습0)')
    check_parser.add_argument('--run', default='runs/latest')
    watch = sub.add_parser('watch', help='선정 모델을 실제 게임에서 관람')
    watch.add_argument('--stage', type=int, default=1)
    watch.add_argument('--run', default='runs/latest')
    watch.add_argument('--model', default='selected_model.zip')
    watch.add_argument('--stochastic', action='store_true')
    watch.add_argument('--headless', action='store_true')
    watch.add_argument('--max-frames', type=int)
    watch.add_argument('--screenshot', type=Path)
    dashboard = sub.add_parser('dashboard', help='전체 학습 곡선·선정 모델 재생·최종 결과')
    dashboard.add_argument('--run', default='runs/latest')
    dashboard.add_argument('--port', type=int, default=8769)
    train = sub.add_parser('train', help='별도 새 모델을0스텝부터 학습하거나 새 실행 재개')
    train.add_argument('--hours', type=float, default=2.)
    train.add_argument('--new', action='store_true')
    train.add_argument('--run', default='runs/latest')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .fresh_training import prepare
            prepare(args.config)
        elif args.command == 'check':
            from .fresh_training import check
            print(json.dumps(check(resolve_ppo_run(args.run), baseline=False),
                             ensure_ascii=False, indent=2))
        elif args.command == 'watch':
            from .play import play
            play(resolve_ppo_run(args.run), stage=args.stage, model=args.model, stochastic=args.stochastic,
                 headless=args.headless, max_frames=args.max_frames, screenshot=args.screenshot)
        elif args.command == 'dashboard':
            from .dashboard import start_dashboard
            server = start_dashboard(resolve_ppo_run(args.run), port=args.port)
            print(f'PPO 학습 결과: {server.dashboard_url}', flush=True)
            try:
                server.dashboard_thread.join()
            finally:
                server.shutdown()
                server.server_close()
        else:
            from .fresh_training import prepare, train
            try:
                run = None if args.new else resolve_ppo_run(args.run)
            except FileNotFoundError:
                run = None
            if run is None:
                run = prepare(SNAPSHOT / 'config.json')
            train(run, max_hours=args.hours)
    except KeyboardInterrupt:
        print('저장·종료했습니다.')
    except ModuleNotFoundError as error:
        parser.exit(2, f'{error.name} 설치가 필요합니다. README의 설치 명령을 확인하세요.\n')
    except FileNotFoundError as error:
        parser.exit(2, f'파일 또는 실행 폴더를 찾지 못했습니다: {error}\n'
                      '설치 후 python run.py prepare로 새 PPO 모델을 준비하세요.\n')


if __name__ == '__main__':
    main()
