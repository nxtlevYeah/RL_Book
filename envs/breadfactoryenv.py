# This file is part of RL_Book Project.
#
# Copyright (C) 2025 nxtlevYeah
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import random
# RL_Book 프레임워크의 Environment 인터페이스와 Spec 클래스를 임포트
from envs.environment import Environment, EnvironmentSpec
# Spec에 필요한 BoundedArray 클래스를 임포트
from utils.array_types import BoundedArray 

# --- 상태(State) 정의 ---
# 1:머핀, 2:계란빵, 3:개미빵
ITEM_MUFFIN = 1
ITEM_EGG = 2
ITEM_ANT = 3

# --- 행동(Action) 정의 ---
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_DROP = 2

class BreadFactoryEnv(Environment):
    """
    '빵공장 아르바이트' 게임 로직의 "턴제" 학습용 환경.
    '신속함'은 배제하고 '정확도'만 학습하는 것을 목표로 함.
    """

    def __init__(self, config, env_id, **kwargs):
        """환경 초기화"""
        self.config = config
        self.env_id = env_id
        
        # 한 판(에피소드)은 1000번의 행동 기회로 제한
        self.max_episode_steps = 1000 
        self.current_episode_step = 0
        
        # main.py의 random_seed를 따라가도록 random 모듈 시드 설정
        if hasattr(config, 'random_seed'):
            random.seed(self.config.random_seed + self.env_id)
        
        self.current_item = ITEM_MUFFIN
        self.reset()

    def _spawn_item(self):
        """새로운 아이템을 무작위로 생성합니다."""
        self.current_item = random.randint(ITEM_MUFFIN, ITEM_ANT)
        return self.current_item

    def step(self, action):
        """
        에이전트의 행동(action)을 받아 1 스텝(1턴)을 진행시킴.
        '정확도'에 대해서만 보상(+1 / -1)을 줌.
        """
        if not isinstance(action, int):
            action = action.item() # 텐서나 numpy일 경우 int로 변환

        reward = -1.0 # 기본 페널티 (틀리면 -1)

        # 정답 판정
        if (self.current_item == ITEM_MUFFIN and action == ACTION_UP) or \
           (self.current_item == ITEM_EGG and action == ACTION_DOWN) or \
           (self.current_item == ITEM_ANT and action == ACTION_DROP):
            reward = 1.0 # 정답!

        # 정답/오답 여부와 관계없이 다음 빵으로 넘어감
        self._spawn_item()
        
        # 에피소드 종료 판정 (1000번 행동했는지)
        self.current_episode_step += 1
        done = self.current_episode_step >= self.max_episode_steps
        
        # 다음 상태(다음 빵)를 float 배열로 반환
        next_state = np.array([self.current_item], dtype=np.float32)
        
        return next_state, reward, done, {}

    def reset(self):
        """환경 초기화"""
        self.current_episode_step = 0
        self._spawn_item() # 첫 번째 빵 생성
        return np.array([self.current_item], dtype=np.float32)

    def environment_spec(self):
        """프레임워크에 이 환경의 스펙을 알려줌"""
        return EnvironmentSpec(
            action_shape=(1,),
            action_dtype=np.int32,
            action_high=[ACTION_DROP], # 행동 최대값 (2)
            action_low=[ACTION_UP],    # 행동 최소값 (0)
            action_size=3,             # 총 행동 개수 (0, 1, 2)
            b_continuous_action=False, # 이산 행동
            state_shape=(1,),          # 상태 (아이템 종류 1개)
            state_dtype=np.float32
        )
    
    def close(self): pass
    def render(self): pass
    def max_episode_limit(self): return self.max_episode_steps
    def select_action(self): return np.array(random.randint(ACTION_UP, ACTION_DROP))