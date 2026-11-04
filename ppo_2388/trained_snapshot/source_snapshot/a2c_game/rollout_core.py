"""환경 내부 물리 프레임에서만 전체 상태 복사를 생략합니다.

원본 게임 코드는 그대로 상속하며 이동·충돌·시간·사건 처리는 변경하지 않습니다.
``step``/``snapshot``은 원래와 같이 독립된 복사본을 반환합니다. ``step_frame``의
중간 snapshot은 다음 물리 프레임까지 읽기 전용으로 빌리는 환경 전용 값입니다.
환경은 정책 결정 경계에서 일반 snapshot을 만들어 외부에 반환해야 합니다.
"""
from copy import deepcopy

from kkamang.core import StepResult
from kkamang.realtime import FRAMES_PER_TICK, RealtimeGameCore

ROLLOUT_CORE_VERSION = 'borrowed_physics_frame_detached_policy_snapshot_v1'


class RolloutRealtimeGameCore(RealtimeGameCore):
    """실제 물리는 원본 구현으로 실행하고 중간 결과의 복사만 줄입니다."""

    _borrowing_frame = False

    def step_frame(self, action):
        """동기 환경 소비 전용: 반환 snapshot의 중첩 값은 보관·수정하지 않습니다."""
        if self._borrowing_frame:
            raise RuntimeError('A borrowed physics frame cannot be stepped recursively.')
        self._borrowing_frame = True
        try:
            return super().step(action)
        finally:
            self._borrowing_frame = False

    def _frame_snapshot(self):
        # 프레임 보상은 이전 quest.collected와 비교하므로 이 작은 사전은 고정합니다.
        # 지도·경로·아이템 등은 즉시 소비하는 동안에만 빌립니다. 공개 snapshot은
        # 부모의 deepcopy 경로를 그대로 사용하므로 이 참조가 외부로 새지 않습니다.
        snapshot = dict(self._state)
        snapshot['quest'] = dict(self._state['quest'])
        if self.auto_interaction:
            snapshot['auto_interaction'] = True
        frame = self._frames
        phase = frame % FRAMES_PER_TICK / FRAMES_PER_TICK
        ghost_positions, chaser_position = self._ghost_motion(phase)
        snapshot['motion'] = {
            'position': list(self._motion_position), 'frame': frame, 'phase': phase,
            'moving': self._moving, 'distance': self._distance,
            'ghost_positions': ghost_positions, 'chaser_position': chaser_position,
            'chaser_stunned': self._chaser_stunned,
        }
        return snapshot

    def _result(self, events):
        if not self._borrowing_frame:
            return super()._result(events)
        # 사건은 정책 행동 사이에도 누적·저장되므로 별도 소유권을 유지합니다.
        return StepResult(self._frame_snapshot(), deepcopy(events),
                          self._state['terminal'], deepcopy(self._state['outcome']))
