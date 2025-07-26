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
"""run_gpy.py: OpenGym 환경이 정상적으로 실행되는지 점검하기 위한 프로그램."""
import argparse
import gym


def run_gym(env_name, n_steps=100):
    """OpenGym 환경 실행.

    @param env_name: 환경 이름 @param n_steps: 환경과의 상호작용 횟수
    """

    # 1. 환경 생성
    env = gym.make(env_name)

    # 2. 행동 공간 및 환경 초기화
    env.action_space.seed(42)
    observation = env.reset()

    # 3. 환경과의 상호작용
    for _ in range(n_steps):
        # 행동 선택
        action = env.action_space.sample()
        # 행동 실행 및 환경 정보 반환
        next_state, reward, done, env_info  = env.step(action)
        env.render()
        
        # 4. 환경 리셋
        if done:
            observation = env.reset()

    # 환경 종료
    env.close()

if __name__ == '__main__':

    # 1. 명령어 인자 파서 생성
    desc = 'OpenGym Test'
    parser = argparse.ArgumentParser(description=desc)
    
    # 2. 환경 이름 인자 추가
    parser.add_argument('-e',
                        '--env',
                        help='run type {CartPole-v1, LunarLanderContinuous-v2}',
                        type=str,
                        default='LunarLanderContinuous-v2')

    # 3. 환경과의 상호작용 횟수 인자 추가
    parser.add_argument('-s',
                        '--steps',
                        help='Number of environment step executions',
                        type=int,
                        default=1000)

    # 4. 명령어 인자 파싱
    args = parser.parse_args()
    print(args)

    # 5. OpenGym 환경 실행
    run_gym(args.env, n_steps=args.steps)