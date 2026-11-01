from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 실제 정기 평가 로그의 귀환 성공 횟수입니다.
# 행: 학습 시드 11, 22, 33 / 열: 추가 정책 결정 수
# 각 평가: 12개 스테이지 × 50판 = 600판
decisions = [0, 5632, 11264, 16896]

counts = {
    "A2C": [
        [394, 395, 396, 395],
        [394, 395, 394, 396],
        [394, 395, 395, 395],
    ],
    "PPO": [
        [394, 395, 395, 398],
        [394, 395, 390, 391],
        [394, 389, 389, 391],
    ],
}

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "font.size": 13,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})

background = "#FFF9EF"
colors = {"A2C": "#8B50CE", "PPO": "#48998B"}

fig, ax = plt.subplots(figsize=(12, 6.75))
fig.patch.set_facecolor(background)
ax.set_facecolor(background)

for algorithm, success_counts in counts.items():
    # 성공 횟수를 백분율로 변환합니다.
    rates = np.array(success_counts) / 600 * 100
    average = rates.mean(axis=0)

    # 세 학습 시드 사이의 표본 표준편차입니다.
    deviation = rates.std(axis=0, ddof=1)
    color = colors[algorithm]

    for seed_curve in rates:
        ax.plot(
            decisions, seed_curve,
            color=color, alpha=0.3, linewidth=1,
            linestyle="--",
        )

    ax.fill_between(
        decisions,
        average - deviation,
        average + deviation,
        color=color, alpha=0.13,
    )

    ax.plot(
        decisions, average,
        color=color, linewidth=3, marker="o",
        label=f"{algorithm} · 마지막 평균 {average[-1]:.2f}%",
    )

ax.set_title("동일한 A2C 사전 학습 모델에서의 추가 학습 비교", pad=22)
ax.set_xlabel("추가 정책 결정 수", labelpad=12)
ax.set_ylabel("귀환 성공률 (%)", labelpad=12)
ax.set_xticks(decisions)
ax.set_xticklabels([f"{value:,}" for value in decisions])

# 작은 차이를 볼 수 있도록 확대한 범위입니다.
# 첨부 슬라이드처럼 전체 비율을 표시하려면 (0, 100)으로 바꾸세요.
ax.set_ylim(60, 70)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", frameon=False)

fig.text(
    0.5, 0.035,
    "전체 12개 스테이지 · 정기 평가 600판/시드/시점\n"
    "실선: 3개 시드 평균 | 점선: 개별 시드 | 음영: ±1 표준편차",
    ha="center", fontsize=11, color="#6F657C",
)

fig.tight_layout(rect=[0, 0.11, 1, 0.97])

output = Path(__file__).resolve().parent
fig.savefig(output / "A2C_PPO_귀환률.png", dpi=300)
fig.savefig(output / "A2C_PPO_귀환률.svg")
plt.show()