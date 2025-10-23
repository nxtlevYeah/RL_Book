# This file is part of RL_Book Project.
#
# Copyright (C) 2025 SeongJin Yoon
#
# RL_Book is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL_Book is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from types import SimpleNamespace
from collections import defaultdict
from agents.base import Network
from datasets.buffer_schema import BufferSchema
from agents.actor import Actor
from envs import REGISTRY as env_REGISTRY

class EnvironmentLoop:

    """액터와 환경의 상호작용 루프를 실행."""

    def __init__(self,
                 config: SimpleNamespace,
                 network: Network,
                 buffer_schema: BufferSchema,
                 actor_class: Actor,
                 env_id: int):
        """
            1) 환경과 액터를 생성하고
            2) 환경 루프 카운터, 통계 정보, 에피소드를 초기화.
        Args:
            config: 설정
            network: 네트워크
            buffer_schema: 버퍼 스키마
            actor_class: 액터 클래스
            env_id: 환경 ID
        """

        # 1. 전달받은 인자 저장
        self.config = config
        self.env_id = env_id

        # 2. 환경/액터 생성
        self.make_environment()
        self.make_actor(network, buffer_schema, actor_class, env_id)

        # 3. 환경 루프 카운터 초기화
        self.n_timesteps_in_envloop = 0     # 환경 루프 타입 스텝 수
        self.n_episodes_in_envloop = 0      # 환경 루프 에피소드 수

        # 4. 통계 정보 초기화
        self.init_stats()

        # 5. 에피소드 초기화
        self.reset_episode()

        # 6. 렌더링 변수 설정
        self.b_render = self.config.render \
            if self.config.training_mode else True

    def make_environment(self):
        """환경 루프에서 사용할 환경을 생성."""

        self.env = env_REGISTRY[self.config.env_wrapper](
            self.config,
            self.env_id,
            **self.config.env_args)

    def make_actor(self, network, buffer_schema, actor_class, actor_id):
        """
            환경 루프에서 사용할 액터를 생성.
        Args:
            network: 네트워크
            buffer_schema: 버퍼 스키마
            actor_class: 액터 클래스
            actor_id: 액터 ID
        """

        # 1. 액터 생성
        self.actor = actor_class(
            config=self.config,
            env=self.env,
            buffer_schema=buffer_schema,
            network=network,
            actor_id=actor_id)

        # 2. 모델 GPU 로딩
        if self.config.use_cuda: self.actor.cuda()

    def run(self, max_n_timesteps: int = 0, max_n_episodes: int = 0):
        """
            지정된 타입 스텝 수 또는 에피소드 수만큼
            액터와 환경의 상호작용을 실행하고 경로 데이터와 통계 정보를 반환.
        Args:
            max_n_timesteps: 타입 스텝 수
            max_n_episodes: 에피소드 수

        Returns:

        """

        # 1. 실행 초기화
        if max_n_timesteps: max_n_episodes = 0
        if max_n_episodes: self.reset_episode()
        self.init_run()

        # 2. 환경 루프 실행
        while self.n_timesteps_in_run < max_n_timesteps \
                or self.n_episodes_in_run  < max_n_episodes:

            # 상호작용 이전 트랜지션 데이터
            pre_transition_data = self.pre_transition_data()

            # 3. 행동 선택
            action = self.select_action()

            # 4. 환경과의 상호작용
            next_state, reward, done, env_info  = self.env.step(action)

            # 💡 렌더링 코드
            if self.config.render:
                self.env.render()

            # 5. 트랜지션 데이터 관측
            # 상호작용 이후 트랜지션 데이터
            post_transition_data = \
                self.post_transition_data(action, reward, next_state, done)
            # 트랜지션 데이터 생성
            transition_data = {**pre_transition_data, **post_transition_data}
            # 액터의 롤아웃버퍼에 트랜지션 데이터 저장
            self.actor.observe(transition_data)

            # 6. 다음 스텝으로 이동
            self.state = next_state             # 다음 상태를 현재 상태로 변경

            # 타입 스텝 카운터 증가
            self.n_timesteps_in_envloop += 1    # 환경 루프 타입 스텝 수
            self.n_timesteps_in_run += 1        # 런 메서드 타입 스텝 수
            self.n_timesteps_in_episode += 1    # 에피소드 타입 스텝 수
            self.return_in_episode += reward    # 에피소드 리턴 계산

            # 7. 에피소드 종료 처리
            if done:
                # 에피소드 카운터 증가
                self.n_episodes_in_envloop += 1  # 환경 루프 에피소드 수
                self.n_episodes_in_run  += 1     # 런 메서드 에피소드 수

                # 통계 정보 업데이트
                self.update_stats()

                # 에피소드 리셋
                self.reset_episode()

        # 8. 환경 루프 실행 결과 반환
        return self.final_result()

    def init_run(self):
        """
            run() 메서드를 실행하기 위해
            1) 런 메서드 카운터와 2) 액터의 롤아웃 버퍼를 초기화.
        """

        # 1. 런 메서드 카운터를 초기화
        self.n_timesteps_in_run = 0     # 런 메서드 타입 스텝 수
        self.n_episodes_in_run  = 0     # 런 메서드 에피소드 수

        # 2. 액터 롤아웃버퍼 초기화
        self.actor.clear_rollouts()

    def final_result(self):
        """
            환경 루프 실행 결과 딕셔너리를 생성.
            {경로 데이터, 환경 루프 타임 스텝 수, 환경 루프 에피소드 수,
            런 메서드 타임 스텝 수, 런 메서드 에피소드 수, 통계 정보}

        Returns:
            환경 루프 실행 결과 딕셔너리
        """

        # 실행 결과 딕셔너리 생성
        result = {
            'rollouts': self.actor.rollouts(),                      # 액터의 롤아웃버퍼
            'n_timesteps_in_envloop': self.n_timesteps_in_envloop,  # 환경 루프 타임 스텝 수
            'n_episodes_in_envloop': self.n_episodes_in_envloop,    # 환경 루프 에피소드 수
            'n_timesteps_in_run': self.n_timesteps_in_run,          # 런 메서드 타임 스텝 수
            'n_episodes_in_run': self.n_episodes_in_run,            # 런 메서드 에피소드 수
            'stats': self.stats                                     # 환경 루프의 통계 정보
        }
        return result

    def reset_episode(self):
        """
            새로운 에피소드를 실행하기 위해 1) 환경 2) 에피소드 카운터를 초기화.
        """

        # 1. 환경 초기화
        self.state = self.env.reset()

        # 2. 에피소드 카운터 초기화
        self.return_in_episode = 0          # 에피소드의 리턴
        self.n_timesteps_in_episode = 0     # 에피소드의 타입 스텝 수

    def select_action(self):
        """
            에이전트가 환경과 상호작용을 하기 위해 행동을 선택
            (단, 워밍업 상태이면 환경에서 제공하는 랜덤한 행동을 선택)
        Returns:
            선택한 행동
        """

        # 1. 워밍업 시 환경의 랜덤 행동 선택
        if self.n_timesteps_in_envloop < self.config.warmup_step:
            return self.env.select_action()

        # 2. 액터의 행동을 선택
        action = self.actor.select_action(self.state,
                                          self.n_timesteps_in_envloop)

        return action

    def update_policy(self, state_dict):
        """
            에이전트의 네트워크 상태(파라미터와 버퍼)를 액터의 네트워크에 로딩
        Args:
            state_dict: 네트워크 상태(파라미터와 버퍼) 딕셔너리
        """

        self.actor.update(state_dict)

    def pre_transition_data(self):
        """
            에이전트와 환경의 상호작용 이전 데이터 {s_t}를 생성
        Returns:
            상호작용 이전 데이터 딕셔너리
        """
        # 1. 상호작용 이전 데이터 생성
        pre_transition_data = {
            "state": self.state,    # 상태 데이터
        }
        # 2. 상호작용 이전 데이터 반환
        return pre_transition_data

    def post_transition_data(self, action, reward, next_state, done):
        """
            에이전트와 환경의 상호작용 이후 데이터 {a_t, r_t, s_(t+1), e_t}를 생성
        Args:
            action: 행동
            reward: 보상
            next_state: 다음 상태
            done: 에피소드 완료 여부

        Returns:
            상호작용 이후 데이터 딕셔너리
        """
        
        # 1. 상호작용 이후 데이터 생성
        post_transition_data = {
            "action": action,           # 행동
            "reward": reward,           # 보상
            "next_state": next_state,   # 다음 상태
            "done": done                # 에피소드 종료 여부
        }

        # 1. 상호작용 이후 데이터 반환
        return post_transition_data

    def init_stats(self):
        """ 환경 루프의 실행 통계 정보 딕셔너리를 생성. """

        # 1. 통계 정보 딕셔너리 생성
        self.stats = defaultdict(float)

        # 2. 실행 결과 초기화
        self.result = None

    def reset_stats(self):
        """ 통계 정보 딕셔너리를 초기화. """

        self.stats.clear()

    def update_stats(self):
        """ 통계 정보 딕셔너리 값을 업데이트. """

        # 1. 에피소드 실행 횟수 증가
        self.stats['n_episodes'] += 1

        # 2. 에피소드 리턴와 에피소드 길이 누적 합산
        self.stats['returns'] += self.return_in_episode
        self.stats['len_episodes'] += self.n_timesteps_in_episode