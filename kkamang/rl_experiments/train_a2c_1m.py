# 첫 번째 시도

import argparse
import torch
from torch import nn
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from kkamang.env import MansionEnv, OBSERVATION_VERSION
from time import perf_counter
from stable_baselines3.common.logger import configure
from kkamang.rl_experiments.progress import SimpleProgressCallback


def make_env():
    """호출할 때마다 새로운 게임 환경 하나를 만든다."""
    return MansionEnv(
        stage=1,
        mode="basic",
        randomize_phase=False,
        safety_coefficient=0.1,
        max_episode_steps=None,
        render_mode=None,
        expected_observation_version=OBSERVATION_VERSION,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    # 팀 공통 설정: PyTorch 계산 스레드 1개
    torch.set_num_threads(1)

    # 서로 독립적인 게임 환경 4개 생성
    vec_env = make_vec_env(
        make_env,
        n_envs=4,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
    )

    try:
        # 팀 공통 설정으로 A2C 모델 생성
        model = A2C(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_rms_prop=True,
            rms_prop_eps=1e-5,
            normalize_advantage=False,
            policy_kwargs={
                "net_arch": {
                    "pi": [64, 64],
                    "vf": [64, 64],
                },
                "activation_fn": nn.Tanh,
            },
            seed=args.seed,
            device="cpu",
            verbose=0,
        )

        # 본 실험은 위에서 새로 생성한 모델로 시작한다.
        run_dir = (
            Path("reports/local/stage1_a2c_v1_1m")
            / f"seed_{args.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # 화면 출력과 CSV 파일에 학습 기록 저장
        model.set_logger(
            configure(str(run_dir), ["csv"])
        )

        started = perf_counter()
        progress = SimpleProgressCallback(
            total_steps=1_000_000,
            print_every=10_000,
        )

        for target in (100_000, 300_000, 1_000_000):
            additional_steps = target - model.num_timesteps

            model.learn(
                total_timesteps=additional_steps,
                reset_num_timesteps=False,
                log_interval=100,
                callback=progress,
            )

            model_path = run_dir / f"a2c_{target}"
            model.save(model_path)

            print("체크포인트 저장:", model_path.with_suffix(".zip"))
            print("누적 학습 스텝:", model.num_timesteps)

        elapsed = perf_counter() - started
        print("본 실험 학습 완료")
        print("실제 소요 시간(초):", round(elapsed, 2))

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()