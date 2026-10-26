# 두 번째 시도
import argparse
import torch
from torch import nn
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from kkamang.env import MansionEnv, OBSERVATION_VERSION


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
            verbose=1,
        )

        # 네 환경을 초기화하고 모델이 행동을 선택하게 한다.
        obs = vec_env.reset()
        actions, _ = model.predict(obs, deterministic=False)

        # 각 환경에 해당 행동을 전달한다.
        next_obs, rewards, dones, infos = vec_env.step(actions)

        assert obs.shape == (4, 620)
        assert next_obs.shape == (4, 620)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)

        print("연결 확인 완료")
        print("학습 seed:", args.seed)
        print("환경 수:", vec_env.num_envs)
        print("관측 모양:", obs.shape)
        print("선택한 행동:", actions)
        print("받은 보상:", rewards)
        print("최적화 방식:", type(model.policy.optimizer).__name__)
        print("학습한 스텝:", model.num_timesteps)

        # 점검용 실행 결과를 별도 폴더에 저장
        run_dir = Path("reports/local/a2c_smoke") / f"seed_{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("1만 스텝 점검 학습 시작")

        model.learn(
            total_timesteps=10_000,
            log_interval=100,
        )

        # 학습한 모델 저장
        model_path = run_dir / "a2c_10000"
        model.save(model_path)

        print("학습 완료:", model.num_timesteps)
        print("저장 위치:", model_path.with_suffix(".zip"))

        # 저장한 모델을 다시 불러와 행동 선택 확인
        loaded_model = A2C.load(model_path, device="cpu")
        obs = vec_env.reset()
        actions, _ = loaded_model.predict(obs, deterministic=True)

        print("불러온 모델의 학습 스텝:", loaded_model.num_timesteps)
        print("불러온 모델이 선택한 행동:", actions)

    finally:
        vec_env.close()


if __name__ == "__main__":
    main()