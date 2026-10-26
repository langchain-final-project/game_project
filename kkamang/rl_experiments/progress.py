from collections import deque
from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback


class SimpleProgressCallback(BaseCallback):
    def __init__(self, total_steps=500_000, print_every=10_000):
        super().__init__(verbose=0)
        self.total_steps = total_steps
        self.print_every = print_every
        self.recent = deque(maxlen=100)
        self.started = None
        self.next_print = print_every

    def _on_training_start(self):
        # learn()을 여러 구간으로 나눠도 시작 시간은 유지한다.
        if self.started is None:
            self.started = perf_counter()
            self.next_print = (
                self.num_timesteps // self.print_every + 1
            ) * self.print_every

    def _on_step(self):
        # 네 환경 중 이번 행동으로 종료된 판의 결과를 모은다.
        for done, info in zip(
            self.locals["dones"], self.locals["infos"]
        ):
            if not done:
                continue

            outcome = info.get("outcome")
            if outcome is None:
                continue

            self.recent.append((
                outcome["quest_collected"],
                outcome["status"] == "COMPLETE",
                outcome["score"]["total"],
            ))

        if self.num_timesteps >= self.next_print:
            if self.recent:
                n = len(self.recent)
                quest_rate = sum(r[0] for r in self.recent) / n
                success_rate = sum(r[1] for r in self.recent) / n
                mean_score = sum(r[2] for r in self.recent) / n
                elapsed = perf_counter() - self.started

                # 새 지표도 CSV에 기록한다.
                self.logger.record("progress/quest_rate", quest_rate)
                self.logger.record("progress/success_rate", success_rate)
                self.logger.record("progress/mean_game_score", mean_score)
                self.logger.record("progress/episodes_in_window", n)

                print(
                    f"[{self.num_timesteps:,}/{self.total_steps:,}] "
                    f"최근{n}판 | 수건 {quest_rate:.1%} | "
                    f"성공 {success_rate:.1%} | "
                    f"점수 {mean_score:.1f} | {elapsed:.0f}초",
                    flush=True,
                )

            self.next_print = (
                self.num_timesteps // self.print_every + 1
            ) * self.print_every

        return True