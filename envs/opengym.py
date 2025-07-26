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
from envs.environment import Environment, EnvironmentSpec
import numpy as np
from utils.util import scale_bias
import gym
import pybullet_envs


class OpenGym(Environment):
    """
        OpenGym 패키지에서 제공하는 강화학습 환경을
        강화학습 프레임워크에 표준화된 형태로 제공하기 위한 래퍼 클래스.
    """

    def __init__(self, config: SimpleNamespace, env_id, **kwargs):
        """
            환경을 생성하고 연속 행동인 경우 정규화 하기 위해 편향과 스케일을 계산
        Args:
            config: 설정
            env_id: 환경 ID
            **kwargs: 환경를 생성할 때 사용할 인자
        """
        # 1. 전달 받은 인자 저장
        self.config = config
        self.env_id = env_id
        env_name = self.config.env_name
        random_seed = self.config.random_seed

        # 2. 환경 생성
        self.env = gym.make(env_name, **kwargs)

        # 3. 환경의 난수 발생기 초기화
        self.env.seed(random_seed+env_id)
        self.env.action_space.seed(random_seed+env_id)
        self.env.observation_space.seed(random_seed+env_id)

        # 4. 연속 행동 여부 설정
        self.b_continuous_action = True
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.b_continuous_action = False

        # 5. 연속 행동의 크기와 편향 계산
        if self.b_continuous_action:
            self.action_scale, self.action_bias = \
                scale_bias(self.env.action_space.high,
                           self.env.action_space.low)

    def render(self):
        """ 강화학습 환경을 화면에 렌더링. """

        return self.env.render()

    def reset(self):
        """ 환경 리셋. """

        return self.env.reset()

    def step(self, action):
        """
            환경에 맞게 행동의 스케일과 데이터 타입을 맞추고 환경에 대해 행동을 취함.
        Args:
            action: 행동

        Returns:
            (다음 상태, 보상, 에피소드 종료 여부, 환경 정보)
        """
        # 1. 행동을 원래의 크기로 복구
        action = self.original_scale(action)

        # 2. 연속 행동인 경우 원래 모양 복구
        if self.b_continuous_action:
            action = np.reshape(action, self.env.action_space.shape)
        else:
            # 3. 이산 행동인 경우 스칼라로 복구
            if isinstance(action, np.ndarray):
                action = action.item()

        # 4. 환경에 대해 행동을 실행
        return self.env.step(action)

    def close(self):
        """환경 종료."""

        self.env.close()

    def original_scale(self, action):
        """
            정규화된 연속 행동의 크기를 원래 스케일로 복구.
        Args:
            action: 정규화된 크기의 행동

        Returns:
            원래 크기의 행동
        """

        # 1. 연속 행동인 경우 원래 크기로 복구
        if self.b_continuous_action:
            return self.action_scale*action + self.action_bias

        # 2. 이산 행동을 그대로 반환
        return action

    def normed_scale(self, value, bias, scale):
        """
            연속 행동의 크기를 [-1,1] 범위로 정규화
        Args:
            value: 행동
            bias: 편향
            scale: 스케일

        Returns:
            정규화된 크기의 행동
        """
        # 1. 연속 행동인 경우 정규화
        if self.b_continuous_action:
            return (value - bias)/scale

        # 2. 이산 행동을 그대로 반환
        return value

    def environment_spec(self):
        """
            환경의 MDP 정보를 표준화된 형태로 제공하기 위해,
            OpenGym에서 제공하는 환경 정보를
            EnvironmentSpec으로 변환해서 제공.

        Returns:
            환경 정보
        """
        # 1. 연속 행동에 대한 정보 추출
        if self.b_continuous_action:
            action_shape = self.env.action_space.shape          # 행동의 모양
            action_size = self.env.action_space.shape[0]        # 행동의 크기
            action_high = self.env.action_space.high            # 행동은 상한과 하한
            action_low = self.env.action_space.low
        else:
            # 2. 이산 행동에 대한 정보 추출
            action_shape = self.env.action_space.shape or [1]   # 행동의 모양 계산
            action_size = self.env.action_space.n               # 행동의 크기
            action_high = [action_size - 1]                     # 행동은 상한과 하한
            action_low = [0]

        # 3. 환경 정보 객체 생성 및 반환
        environment_spec = EnvironmentSpec(
            action_shape=action_shape,
            action_dtype=self.env.action_space.dtype,
            action_high=action_high,
            action_low=action_low,
            action_size=action_size,
            b_continuous_action=self.b_continuous_action,
            state_shape=self.env.observation_space.shape,
            state_dtype=self.env.observation_space.dtype)

        return environment_spec

    def select_action(self):
        """
            환경에서 제공하는 행동 공간에서 임의의 행동을 샘플링해서 반환.
            (연속 행동인 경우 값을 [-1,1] 범위로 정규화해서 반환)
        Returns:
            선택된 행동
        """

        # 1. 환경에서 랜덤 행동 샘플링
        action = self.env.action_space.sample()

        # 2. 연속 행동인 경우 정규화
        if self.b_continuous_action:
            action = self.normed_scale(action,
                                       self.action_bias,
                                       self.action_scale)
        # 3. 행동 반환
        return action

    def max_episode_limit(self):
        """ 환경의 최대 에피소드 길이를 반환. """

        return self.env._max_episode_steps