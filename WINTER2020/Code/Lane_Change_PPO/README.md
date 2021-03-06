# flow-autonomous-driving

### Requirement(Installment)
- Ubuntu 18.04 is recommended. (Window OS is not supported.)
- anaconda : https://anaconda.com/
- flow-project : https://github.com/flow-project/flow
- ray-project(rllib) : https://github.com/ray-project/ray (need at least 0.8.6 is needed)
- pytorch : https://pytorch.org/

### How to Download Requirement
#### Anaconda(Python3) installation:
- Prerequisites
```shell script
    sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
- Installation(for x86 Systems)
In your browser, download the Anaconda installer for Linux (from https://anaconda.com/ ), and unzip the file. 
``` shell script
bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh
```
We recomment you to running conda init 'yes'.<br/>
After installation is done, close and open your terminal again.<br/>


#### Flow installation
Download Flow github repository.
```shell script
    git clone https://github.com/flow-project/flow.git
    cd flow
``` 
We create a conda environment and installing Flow and its dependencies within the enivronment.
```shell script
    conda env create -f environment.yml
    conda activate flow
    python setup.py develop
```
For install flow within the environment
```shell script
pip install -e .
```
For Ubuntu 18.04: This command will install the SUMO for simulation.<br/>
```shell script
bash scripts/setup_sumo_ubuntu1804.sh
```
For checking the SUMO installation,
```shell script
    which sumo
    sumo --version
    sumo-gui
```
(if SUMO is installed, pop-up window of simulation is opened)
- Testing your SUMO and Flow installation
```shell script
    conda activate flow
    python simulate.py ring
```


#### Torch installation (Pytorch)
You should install at least 1.6.0 version of torch.(torchvision: 0.7.0)
```shell script
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```


#### Ray RLlib installation
You should install at least 0.8.6 version of Ray.(Recommend 0.8.7)<br/>
```shell script
pip install -U ray==0.8.7
```
- Testing RLlib installation
```shell script
    conda activate flow
    python train_rllib.py singleagent_ring
```
If RLlib is installed, turn off the terminal after confirming that "1"  appears in the part where iter is written in the terminal.


#### Visualizing with Tensorboard
To visualize the training progress:<br/>
```shell script
tensorboard --logdir=~/ray_results singleagent_ring
```

If tensorboard is not installed, you can install with pip, by following command `pip install tensorboardx`.

### Downloads Flow-autonomous-driving repository 
Download related file for training and visualizing:<br/>
```shell script
cd 
git clone https://github.com/bmil-ssu/flow-autonomous-driving.git
```


## How to Use

## RL examples
### RLlib (for multiagent and single agent)

for PPO(Proximal Policy Optimization), DDPG(Deep Deterministic Policy Gradient), and TD3(Twin Delayed DDPG) algorithm
```shell script
python train_rllib.py EXP_CONFIG --algorithm [DDPG or PPO or TD3]
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in directory`exp_configs/rl/singleagent`.<br/>
In '[DDPG or PPO or TD3]', You can choose 'DDPG' or 'PPO or TD3' Algorithm.(Default: PPO)

### Visualizing Training Results
If you want to visualizing after training by rllib(ray), 
- First, ```conda activate flow``` to activate flow environment.
- Second,
```shell script
    python ~/flow-autonomous-driving/visualizer_rllib.py 
    ~/home/user/ray_results/EXP_CONFIG/experiment_name/ number_of_checkpoints
```
```experiment_name``` : Name of created folder when learning started.<br/>
```number_of_checkpoints``` : 
It means the name of the checkpoint folder created in the experiment_name folder. Enter the checkpoint (number) you want to visualize.
### Results for training Ring Network and Figure-Eight Network
#### PPO (Proximal Policy Optimization)
- Ring Network (ring length 220-270 for training)
![image](https://user-images.githubusercontent.com/59332148/91409511-78e5b780-e880-11ea-8d57-6f1d3008694a.png) <br/>
Mean velocity in 22 Non-AVs system: 4.22m/s (ring length: 260)<br/>
Mean velocity in 1 AV, 21 Non-AVs system: 4.67m/s, Mean cumulative reward: 2350 (ring length: 260) <br/>
 Use Stochastic Sampling Exploration method<br/>
 Reward seems to converge in 2300, this result is regarded as success experiment.
- Figure-eight Network
![image](https://user-images.githubusercontent.com/59332148/91409219-1ab8d480-e880-11ea-8331-7eabc58afef2.png) <br/>
Mean velocity in 14 Non-AVs system: 4.019m/s (total length: 402)<br/>
Mean velocity in 1 AV, 13 Non-AVs system: 6.67m/s (total length: 402)<br/>
 Use Gaussian Noise Exploration method<br/>
 Reward seems to converge in 19,000, this result is regarded as success experiment.<br/>
 Graph that is represented going back and forward penomenon is normal graph due to its failures.<br/>
 Having failure gives penalty to autonomous vehicle.<br/>
#### DDPG (Deep Deterministic Policy Gradient)
- Ring Network(ring length 220-270 for training)
![image](https://user-images.githubusercontent.com/59332148/91408962-b0079900-e87f-11ea-95b3-020a5809e746.png) <br/>
 Mean velocity in 22 Non-AVs system: 4.22m/s (ring length: 260)<br/>
 Mean velocity in 1 AV, 21 Non-AVs system: 4.78m/s, Mean cumulative reward: 2410 (ring length: 260) <br/>
 Use Ornstein Uhlenbeck Noise Exploration method<br/>
 
- Figure-eight Network
will be added

## non-RL examples

```shell script
python simulate.py EXP_CONFIG
```

where `EXP_CONFIG` is the name of the experiment configuration file, as located in `exp_configs/non_rl.`

If you want to run with options, use
```shell script
python simulate.py EXP_CONFIG --num_runs n --no_render --gen_emission
```
![Figure_Eight Ring](https://user-images.githubusercontent.com/59332148/91126855-f1f9d900-e6df-11ea-96ec-b3a5ee49b917.png)
    Ring Network, Figure-Eight Network(left, right)
## OSM - Output (Open Street Map)
![OSM_Combined](https://user-images.githubusercontent.com/59332148/91114406-ccaaa200-e6c2-11ea-932b-cfc2f18a6669.png)

[OpenStreetMap]https://www.openstreetmap.org/ 

If you want to use osm file for making network, Download from .osm files. After that _map.osm_ file should **replace** the same name of file in 'Network' directory.
You want to see their results, run this code.

```shell script
python simulate.py osm_test
```

After that, If you want to see those output file(XML), you could find in `~/flow/flow/core/kernel/debug/cfg/.net.cfg`

### 설치 및 코드 실행 방법 (한글설명)
(안내하는 모든 명령어는 컴퓨터를 초기화후, Ubuntu 18.04 환경을 설치한 뒤 Terminal에서 실행한다.)

#### Anaconda3 설치방법
1. Anaconda3 설치 전 다음 명령어를 실행한다.
```
   sudo apt-get update
   sudo apt-get upgrade   
   sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
2. Anaconda 사이트 또는 아래 링크에서 Anaconda installer(Linux version)를 다운로드한 후, 다음 명령어를 실행한다. (https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)
```
    sudo apt-get install python3 python python3-pip python-pip git
```

3. 압축을 해제한 설치파일(.sh)을 실행한다.
 ```
 bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
 ```
(실행 파일을 다운받은 시기와 32bit, 64bit 여부에 따라 파일명이 달라질 수 있다. 설치 파일 실행 중 나오는 메시지는 ‘yes’, ‘enter’로 이용약관의 동의 및 설치 주소 확인 후, conda init 여부에는 ‘yes’를 순차적으로 입력한다.)

4. 가상환경을 활성화하기 위해 anaconda 설치가 완료되면 terminal을 종료 후 다시 연다. 

#### FLOW, SUMO 설치방법
1. Flow github repository를 다운로드한다. 
   ```
    conda
    git clone https://github.com/flow-project/flow.git
    cd flow
    ```

2. Anaconda를 이용해서 가상환경을 만든다. 
    ```
    conda update –n base –c defaluts conda
    conda env create –f environment.yml
    source ~/.bashrc
    conda activate flow
    python setup.py develop
    ```
3. 가상환경 내에 FLOW 관련 파일을 설치한다.
   ```shell script
   pip install -e .
   ```
4. Ubuntu 18.04에서 simulation을 위해서 SUMO를 설치한다. 
    ```
    bash scripts/setup_sumo_ubuntu1804.sh
    ```
    
5. SUMO 설치를 확인한다. (SUMO가 설치된 경우, simulation 팝업 창이 열린다.)
    ```
    which sumo
    sumo --version
    sumo-gui
    ```
6. FLOW 설치를 확인한다. 
    ```
    conda activate flow
    python examples/simulate.py ring
    ```

#### Pytorch 설치
Pytorch 1.6.0 이상의 version을 설치한다. 
```
   conda deactivate
   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

#### RLlib 설치 
1. Ray 0.8.7 version을 설치한다.(해당 버전이 아닌 경우 오류가 발생할 수 있다.)
    ```
    conda activate flow
    pip install –U ray==0.8.7
    pip install dm-tree
    ```
2. RLlib 설치를 확인한다.
    ```
    conda activate flow
    python examples/train.py singleagent_ring
    ```
(RLlib이 설치된 경우, terminal에서 iter라고 쓰여진 부분에 1이 나타나는 것을 확인한 후, terminal을 끈다. )

3. Training 과정은 iter라고 쓰여진 부분이 2 이후로 진행된 후부터 경향성을 확인할 수 있고, 이는 tensorboard를 이용하여 확인한다. 
    ```
    cd
    cd ray_results
    tensorboard --logdir singleagent_ring
    ```
   만약 tensorboard가 설치가 되어 있지 않다면 다음 명령어를 입력하여 설치한다. 
   ```
   pip install tensorboardx
   ```
   
#### Flow-autonomous-drivng repository 다운로드
1. 아래에서 관련 파일을 다운로드한다.  
```
    cd 
    git clone https://github.com/bmil-ssu/flow-autonomous-driving.git
```

#### Lane Change Training Setup
Lane_Change_PPO/Requirement 안의 파일들을 flow 내부로 옮긴다.
```angular2html
cd flow-autonomous-driving/WINTER2020/Code/Lane_Change_PPO
bash ./setup_requirements.sh
```
#### Lane Change Training Execution
1. Lane Change를 학습시키려면 Lane_Change_PPO 디렉토리 안에서 다음의 명령을 실행한다.
```angular2html
python train_rllib.py EXP_CONFIG
```
2. 특정한 satefy를 설정하려면 `--safety [0~2]`옵션을 준다.

#### Lane Change training result visualization
학습 결과를 보려면 Lane_Change_PPO 디렉토리 안에서 다음의 명령을 실행한다.
```angular2html
python visualizer_rllib.py {result_dir}
```
[result_dir] : 학습 결과가 저장되는 경로

#### Lane Change reward params tune
reward 함수의 파라미터들은 `InitialConfig.reward_params`객체로부터 조정할 수 있다.
```angular2html
InitialConfig(reward_params:dict = {...})
```
ray rllib에서 제공하는 PPO 알고리즘은 -2000 이하나 +2000 이상의 매우 큰 리워드를 지원하지 않는다. 따라서 reward 함수의 파라미터들을 적절히 조정할 필요가 있다.

또한 Environment 마다 적절한 파라미터로 최적화할 필요가 있다. 그렇지 않으면 학습에 실패할 수 있다.

#### Applying new environment for Lane Change Training
`Lane_Change_PPO/requirements/my_lane_change_accel.py`에서 4가지의 Lane Change를 위한 Environment를 찾아 볼 수 있다.
1. `MyLaneChangeAccelEnv`
2. `MyLaneChangeAccelPOEnv`
3. `TestLCEnv`
4. `TestLCPOEnv`

#Appendix

부록에서는 바로 Lane change를 위한 설정들을 익히고 커스터마이징 할 수 있도록 한다.

##1. Experiment Configuration
이 절은 학습을 진행하기 위해 필요한 파라미터 설정을 설명한다.
###Start up

Exp_config 파일은 학습 시 필요한 설정 파라미터를 Ray에 넘겨주는 역할을 한다. Ray는 이 파일로부터 flow_params 객체를 받아 학습 환경을 설정한다.  
Exp_config 파일 내부에는 다음 설정 파라미터가 위치해 있다.

1. 학습 환경(Environment)에 대한 정보 : env_name, env
2. 학습 환경에 배치할 차량에 대한 정보 : veh
3. 학습 환경 내부의 도로망(Network) 정보 : net
4. 학습 시뮬레이션에 대한 정보 : sim
5. 기타 추가적인 정보 : initial

Exp_config 파일은 `Lane_Change_PPO/exp_configs/` 에서 찾아볼 수 있다.

### Vehicle setup
학습 설정 파일에서 차량에 관련한 세팅은 VehicleParams 객체를 통해서 이루어진다.
Lane change 를 위해서 다음의 파라미터를 설정할 수 있다.
* veh_id : 차량 그룹의 id. (그룹 내 차량들은 "{veh_id}_[int]" 형식을 가진다.)
* acceleration_controller : 차량의 accel 을 컨트롤하는 객체.
* lane_change_controller : 차량의 lane change 를 컨트롤 하는 객체
* routing controller : 차량의 주행 경로를 컨트롤하는 객체
* num_vehicles : 그룹의 차량 수
* initial_speed : 시뮬레이션 시작 시 차량의 속

### EnvParams setup
학습 설정 파일에서 학습 환경에 관련한 세팅은 EnvParams 객체를 통해서 이루어진다. 
여기서는 Horizon, Warmup steps, Evaluate 여부 등 환경 설정 중 시뮬레이션과 관련된 설정이 주로 이루어지며,
새로운 학습 환경을 만들거나 학습 환경 자체를 커스터마이징을 하고자 하면, `env_name`에 넘겨준 gym.Env 객체에서 커스터마이징 해야 한다.

### NetParams setup
학습 설정 파일에서 도로망에 관련한 세팅은 NetParams 객체를 통해서 이루어진다. 

### InitialCongif setup
학습 설정 파일에서 시뮬레이션 초기화와 관련한 세팅은 InitialConfig 객체를 통해서 이루어진다.
이 객체를 통해 차량의 배치(spacing), 차량의 배치 밀도(bunching) 등을 설정할 수 있다.

## 2. New Environment
새로운 학습 환경을 만들고자 한다면, flow.envs.base.Env 객체를 상속해야 한다.
여기서 <b>action_space</b>, <b>observation_space</b> 를 정의할 수 있으며 (gym.space) action의 적용과 리워드 처리 등을 직접 커스터마이징 할 수 있다.

새롭게 커스터마이징 하였던 환경을 `flow/envs/ring/my_lance_change_accel.py`에서 찾을 수 있다.

## 3. New Reward
새로운 환경을 만들 때 새로운 보상 함수 체계를 정의하려면 `flow.core.rewards.py`에 정의된 보상함수들을 고치거나 새롭게 정의해야 한다.

새롭게 커스터마이징 하였던 리워드를 `flow.core.lane_change_rewards.py` 에서 찾을 수 있다.

## 4. New Network
새로운 도로망을 만들고자 한다면, flow.networks.base.Network 객체를 상속해야 한다.
여기서 도로망의 edge를 정의할 수 있으며 도로의 모양을 설정할 수 있다.
또한 도로 내의 차량 배치도 정의할 수 있다.

새롭게 커스터마이징 히였던 도로망을 `flow/networks/lane_change_ring.py` 에서 찾을 수 있다.


###


### Encrypted 2020 SUMMER Documentation for Flow 
-English Ver: [DocumentPDF]https://drive.google.com/file/d/1NQRoCFgfIh34IJh4p0GqqOWagZh543X2/view?usp=sharing

-Korean Ver: [DocumentPDF]https://drive.google.com/file/d/1BUStOlq8LRypEmwXfRLD-_Xd04wnmCwL/view?usp=sharing


## Contributors
_BMIL at Soongsil Univ._
Prof. Kwon (Minhae Kwon), 
Minsoo Kang, 
Gihong Lee, 
Hyeonju Lim,
Dongsu Lee,
Sunwoong Kim.
