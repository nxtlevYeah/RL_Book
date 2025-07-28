# 파이토치로 완성하는 실전 강화학습
**본 저장소는 [파이토치로 완성하는 실전 강화학습] 책에서 설명하는 강화학습 프레임워크와 강화학습 알고리즘의 소스 코드를 제공하고 있습니다.**

<img src="img/chapters.png" alt="cover" width="500"/>

## 1. 디렉토리 구성
|디렉토리              |설명                        |
|:--        |:--                          |
| agents 	| REINFORCE, A2C, DQN, DDQN, PPO 에이전트 관련 클래스 정의 |
| config	| 에이전트 실행을 위한 설정 파일 |
| datasets	| 데이터셋 클래스 정의 |
| envs	    | 환경 클래스 정의 |
| models	| 정책과 가치 함수 모델 정의 |
| runner	| 에이전트 실행을 위한 러너와 환경 루프 클래스 정의 |
| utils	    | 다양한 유틸리티 함수 정의 |

##  2. 개발 환경 설치
#### 2-1. 가상 환경 구성
Python 3.9 버전의 가상 환경을 만든다. 
예를 들어, 다음과 같은 conda 명령어로 'RL_Book' 가상환경을 만들어 보자.
```bash
conda create -n RL_Book python=3.9
```

#### 2-2. PyTorch 설치
[PyTorch 홈페이지](https://pytorch.org/get-started/locally/)에 가면 
로컬 환경에 맞게 PyTorch 설치 명령어를 생성해 주는 기능이 제공되고 있으니 
이를 활용하여 PyTorch를 설치해보자.
##### CPU 버전
```bash
pip3 install torch torchvision torchaudio
```
##### GPU 버전
다음 명령어는 CUDA 11.8 버전 상에서 PyTorch GPU 버전을 설치하는 예시이다.
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
   * GPU 버전을 설치할 때는 CUDA 툴킷과 cuDNN이 설치되어 있어야 하므로 [CUDA](https://developer.nvidia.com/cuda-downloads)와 [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)에서 download 받아서 설치하라.

#### 2-3. 파이썬 패키지 설치
requirement.txt을 이용해서 파이썬 패키지를 일괄로 설치한다.

```bash
pip3 install -r requirements.txt
```
#### 2-4. 개발 환경 설치 점검
강화학습 프레임워크를 개발하기 위한 환경이 정상적으로 구성됐는지 확인해 보자! 

##### OpenGym 설치 확인
```bash
python run_gym.py --env LunarLanderContinuous-v2 --steps 1000
```
   * --env: OpenGym 환경 이름 
   * --steps: 환경과의 상호작용 횟수

##### 강화학습 프레임워크 설치 확인
```bash
python main.py
```

## 3. 강화학습 알고리즘 실행 방법
 
```bash
python main.py --agent ppo --env CartPole-v1
```
  * --agent: 에이전트 이름 {reinforce, reinforce_b, a2c, dqn, ddqn, ppo}
  * --env: 환경 이름 {CartPole-v1, LunarLanderContinuous-v2}

## 4. 설정 항목 
### 공통 설정 항목
##### GPU
  * **use_cuda**: True
    * GPU 사용 여부
    * 로컬 환경에 GPU가 없는데 use_cuda를 True로 설정하면 Runner에서 이를 체크해서 False로 만든다.
  * **device_num**: 0
    * 딥러닝 모델 학습 및 추론 시 사용할 GPU 번호(기본 값은 0)

##### Ray
  * **n_cpus**: 4
    * Ray에서 분산 처리를 할 때 사용할 CPU 개수
  * **n_gpus**: 1
    * Ray에서 분산 처리를 할 때 사용할 GPU 개수

##### 기타
  * **epsilon**: 0.0000001
    * 산술 연산을 할 때 수치적 안정성을 위해 사용하는 작은 상수
    * 예를 들어 나누기를 할 때 분모가 0이 되지 않게 분모에 더해준다.

##### 옵티마이저
  * **optim_betas** : [0.9, 0.999]
    * Adam에서 사용하는 (beta1, beta2) 계수
    * beta1: 그레이디언트의 이동 평균을 계산할 때 사용하는 계수
    * beta2: 그레이디언트 제곱의 이동 평균을 계산할 때 사용하는 계수
  * **optim_eps**: 0.00001
    * Adam에서 사용하는 상수 값
  * **torch_deterministic**: True
    * PyTorch에서 난수를 고정했을 때 동일한 학습 결과가 나오게 하는 옵션이다. 값을 True로 하면 학습 성능이 조금 느려질 수 있다.

##### 환경
  * **env_wrapper**: 'opengym'
    * 강화학습 환경을 제공하는 패키지 이름
    * 단, 현재는 'opengym'만 제공하고 있다.
  * **env_name**: 'CartPole-v1'
    * 환경 이름(환경 이름은 패키지에서 제공하는 이름을 준수해야 한다)
    * 단, main.py에서는 인자로 받은 환경 이름을 사용하므로 설정 파일의 환경 이름은 실행에 영향을 주지 않는다.
  * **n_envs**: 1
    * 학습 시 사용할 환경의 개수
  * **distributed_processing_type**: "sync"
    * 분산 처리 방식 {"sync", "async"}
    * "sync": 동기적 분산 처리 방식
    * "async": 비동기적 분산 처리 방식
  * **render**: False
    * 강화학습 환경을 화면으로 보여줄지 여부
    * 학습할 때는 환경을 화면으로 보여주면 학습 속도가 매우 늦어지므로 False로 두는 것이 좋다.

##### 로깅
  * **log_interval**: 2000
    * 로깅 주기(타입 스텝 단위)
    * Runner에서 학습 성능을 주기적으로 콘솔로 출력할 때 사용한다.
  * **use_tensorboard**: True
    * 로거에 통계 정보를 로깅할 때 텐서보드에도 같이 로깅 할지를 지정
    * 전체적인 학습 곡선의 모양을 확인할 때 사용

##### 체크포인트
  * **save_model**: True
    * 학습 시 모델과 옵티마이저의 체크포인트를 저장할지 여부
    * 학습 과정이 길어질 경우 주기적으로 체크포인트를 저장해서 장애가 발생했을 때 복구해서 사용한다.
    * 학습이 완료된 체크포인트는 모델 선정 과정을 거쳐 추론 모델로 사용한다.
  * **save_model_interval**: 20000
    * 체크포인트를 저장하는 주기(타입 스텝 단위)
    * 체크포인트를 저장 시점별로 구분하기 위해 타임 스텝 디렉토리에 저장한다.
  * **checkpoint_path**: ""
    * 복구해야 할 체크포인트가 있는 경로
  * **load_step**: 0
    * 복구할 체크포인트의 타입 스텝
    * 지정한 타입 스텝의 체크포인트가 없다면 가장 가까운 타임 스텝의 체크포인트로 복구한다.
  * **local_results_path**: "results" 
    * 학습 과정의 산출물을 저장하기 위한 디렉토리의 이름
    * 텐서보드 로그이나 체크포인트를 저장하기 위한 용도로 강화학습 프레임워크의 실행 디렉토리 하위에 생성된다.

### 학습과 관련된 설정 항목 (성능 튜닝의 대상)
##### 실행 모드
  * **training_mode** : True
    * 학습 모드인 경우 True로 추론 모드인 경우 False로 설정
  * **trained_model_path**: ""
    * 추론 모델의 경로
  * **test_mode_max_episodes**: 100
    * 추론 모드에서 실행할 최대 에피소드 수
    * 현재 추론 모드에서는 지정된 에피소드 수만큼 환경을 실행하고 종료하는 방식으로 구현돼 있다.

##### 훈련 스텝 (Training Steps)
  * **max_environment_steps**: 100000
    * 학습 모드에서 실행할 최대 환경 타임 스텝 수
  * **n_steps**: 1000
    * 환경 루프에서 실행할 타입 스텝 수
  * **n_episodes**: 0
    * 환경 루프에서 실행할 에피소드 수
    * n_steps과 n_episodes가 같이 설정돼 있으면 n_steps가 우선순위가 높음
  * **n_epochs**: 1
    * Learner가 실행할 에포크 수
    * REINFORCE, A2C, PPO와 같은 온라인 정책 알고리즘에서 롤아웃 버퍼에 저장된 데이터셋을 몇 번 학습할지를 나타내는 횟수
  * **gradient_steps**: 64
    * Learner가 실행할 그레이디언트 스텝 수
    * DQN, DDQN과 같은 오프라인 정책 알고리즘에서 리플레이 버퍼에서 몇 번 배치를 샘플링해서 학습할지를 나타내는 횟수
  * **batch_size**: 32
    * 배치 크기

##### 할인 계수
  * **gamma**: 0.99
    * 리턴이나 이득, 가치를 계산할 때 사용하는 할인 계수 (discount factor)

##### 학습률
  * **lr_policy**: 0.005
    * 정책 모델의 학습률
  * **lr_ciritic**: 0.005
    * 가치 함수 모델의 학습률

##### 학습률 스케쥴링
  * **lr_annealing**: True
    * 학습률 감소를 처리할지 여부
    * 현재 학습률 감소 방식은 코사인 어낼링(cosine annealing)으로 고정돼 있다.

##### 리플레이 버퍼 워밍업
  * **warmup_step**: 0
    * 학습 초반에 리플레이 버퍼를 채우기 위해 대기하는 타임 스텝 수

##### 입실론 그리디
  * **epsilon_greedy**: False
    * 입실론-그리디 사용 여부
    * 입실론-그리디는 $𝜀$의 확률로 무작위 행동을 선택하고 $1−𝜀$의 확률로 가장 가치가 높은 최적 행동을 선택한다
  * **epsilon_start**: 0.1
    * 입실론-그리디에서 입실론을 감쇄할 때 입실론의 시작 값
  * **epsilon_finish**: 0.01
    * 입실론-그리디에서 입실론을 감쇄할 때 입실론의 종료 값
  * **epsilon_anneal_time**: 70000
    * 입실론을 줄여 나가는 기간(타임 스텝 기준)
 
##### 그레이디언트 클립핑
  * **grad_norm_clip**: 0.3
    * 그레이디언트 클리핑에 사용하는 임계치 값
    * 그레이디언트 클리핑은 그레이디언트 폭발을 막기 위해 사용한다.
    * 단, 임계치가 너무 작으면 정상적인 그레이디언트 값도 작아져서 학습이 원활하지 않을 수 있으니 임계치를 주의해서 지정해야 한다.

##### 리턴과 이득
  * **advantage_type**: 'gae'
    * 리턴(return) 또는 이득(advantage)의 유형으로 {"gae", "n_step", "mc"} 중에 선택한다.
    * "gae": GAE(Generalized Advantage Estimate)
    * "n_step": n 스텝 리턴
    * "mc": 몬테카를로 리턴
    * 정책을 학습하는 REINFORCE, A2C, PPO 알고리즘에서 사용한다.
  * **n_steps_of_return**: 10
    * n 스텝 리턴을 계산하기 위한 스텝 수
  * **return_standardization**: True
    * (n_step이나 mc인 경우)리턴을 표준화할지를 지정한다.
  * **gae_standardization**: False
    * GAE 계산 시 이득을 표준화할지를 지정한다.
  * **gae_lambda**: 0.98
    * GAE 계산 시 분산-편향 조절 할인 계수

##### PPO 클립핑
  * **ppo_clipping_epsilon**: 0.2 
    * PPO에서 이전 정책과 현재 정책의 로그 가능도 비율의 클리핑 임계치  
  * **clip_schedule**: True
    * 클리핑 임계치인 입실론을 감쇄할지 여부를 지정한다.

##### 손실 함수 계수
  * **vloss_coef**: 0.2 
    * 가치 함수의 손실 계수(PPO, A2C에서 사용)
  * **eloss_coef**: True
    * 엔트로피 보너스 계수(PPO, A2C에서 사용)


##### 네트워크
  * **actor_hidden_dims**: [64, 64, 64]
    * 정책 모델의 은닉 계층별 뉴런 수를 나타내는 리스트
  * **critic_hidden_dims**: [64, 64, 64]
    * 가치 모델의 은닉 계층별 뉴런 수를 나타내는 리스트

## 5. 라이선스
이 저장소의 소스 코드는 [GPL 3.0 라이선스](LICENSE)를 따릅니다.
상업용과 비상업용으로 자유롭게 이용하으며, 사용 시 출처를 밝히고 소스를 공개해야 할 의무가 있다는 점을 말씀드립니다.

## 6. 자주 묻는 질문

**어떤 파이썬 버전을 사용해야 하나요?**

최종 테스트 버전은 Python 3.9입니다.</br>
단, Python 3.7이나 3.8에서도 실행은 될 수 있으나 전체적인 테스트는 수행되지 않았다는 점을 참고해 주세요. 

**어떤 pytorch 버전을 사용해야 하나요?**

최신 버전을 사용하시면 됩니다. 설치 이슈가 있으면 최종 테스트 버전은 아래와 같으니 참고하세요.
  * torch 2.7.1
  * torchvision 0.22.1
  * torchaudio 2.7.1