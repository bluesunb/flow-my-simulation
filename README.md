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
1. 아래에서관련 파일을 다운로드한다.  
```
    cd 
    git clone https://github.com/bmil-ssu/flow-autonomous-driving.git
```

#### DDPG-TD3 Neural network Output layer Activation func. 
   만약 DDPG-TD3 알고리즘에서 actor network의 Output layer에 tanh를 적용하고 싶다면 다음을 따른다.
   (2020 겨울학기 이후로 TD3와 DDPG의 연구 결과는 위와 같은 network 구조를 사용했다.)
```
   cd ~/flow-autonomous-driving/Additional_File
   cp -f ddpg_torch_model.py ~/anaconda3/envs/flow/lib/python3.7/site-packages/ray/rllib/agents/ddpg/
```
   수정한 부분은 주석처리 하여 다음과 같이 표시했다.
   만약 default 값을 사용하고 싶다면 SlimFC 내부 activation_fn을 None으로 바꾸어 입력하면 된다.
```shell script
        #bmil edit for apply activation function - 'tanh' - to output layer in actor network
        output_policy_fc = get_activation_fn("tanh", framework="torch")
        self.policy_model.add_module(
            "action_out",
            SlimFC(
                ins,
                self.action_dim,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn=output_policy_fc)) #defalut: activation_fn=None
```

#### Training: DDPG +Ring 
1. 원형 도로에서 DDPG 알고리즘 기반으로 RL agent를 Learning 시키기 위해 터미널에 다음과 같은 명령어를 입력한다.
```
   cd flow-autonomous-driving
   python train_rllib.py singleagent_ring --algorithm ddpg
```

2. Training이 끝난 후 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다. (visualizing을 하기 위해서는 40번 이상의 iter를 돌려야 볼 수 있다. )
```
   cd
   cd flow-autonomous-driving
   python visualizer_rllib.py ./ray_results/singleagents_ring/experiment_name  number_of_checkpoints
```
 -  experiment_name: Learning을 시작할 때, 생성된 폴더의 이름
 - number_of_checkpoints: experiment_name 폴더 안에 생성된         checkpoint 폴더의 이름을 의미한다. visualizing을 하고자 하는    
   checkpoint(숫자)를 입력한다. 
   
3. Training이 끝난 후, 그래프를 확인을 하고자 하면 다음과 같은 명령어를 입력한다. (명령어를 친 후 위의 experiment_name을 찾아서 본다.)
```
  cd
  cd ray_results
  tensorboard --logdir singleagent_ring
```

#### High performance Example Code
1. High performance Example Code의 training 결과를 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
    cd
    cd flow-autonomous-driving/2020SUMMER/Code/Ring_Network-DDPG
    python visualizer_rllib.py ./results/DDPG_WaveAttenuationPOEnv-v0_0_2020-08-25_14-03-01r8h_t432 330
```

2. High performance Example Code의 Training 결과를 tensorboard를 이용하여 확인하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
    tensorboard —logdir results
```



#### Training: PPO+figure8 
1. 교차로 원형 혼합 도로 구조에서 PPO 알고리즘 기반으로 RL agent를 Learning 시키기 위해 터미널에 다음과 같은 명령어를 입력한다. 
```
  cd flow-autonomous-driving  
  python train_rllib.py singleagent_figure_eight
```

2. Training이 끝난 후 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다.
    ```
   cd
   cd flow-autonomous-driving
   python visualizer_rllib.py ./ray_results/singleagent_figure_eight/experiment_name  number_of_checkpoints
   ```
 -  experiment_name: Learning을 시작할 때, 생성된 폴더의 이름
 - number_of_checkpoints: experiment_name 폴더 안에 생성된         checkpoint 폴더의 이름을 의미한다. visualizing을 하고자 하는    
   checkpoint(숫자)를 입력한다. 
   
3. Training이 끝난 후, 그래프를 확인을 하고자 하면 다음과 같은 명령어를 입력한다. (명령어를 친 후 위의 experiment_name을 찾아서 본다.)
```
  cd
  cd ray_results
  tensorboard --logdir singleagent_figure_eight
```


#### High performance Example Code 
1. High performance Example Code의 training 결과를 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다.
```
    cd
    cd flow-autonomous-driving/2020SUMMER/Code/Figure_Eight_Network-PPO
    python visualizer_rllib.py ./results/PPO_AccelEnv-v0_0_2020-09-04_13-47-12x8hecjmf 1500
```

2. High performance Example Code의 Training 결과를 tensorboard를 이용하여 확인하려면, terminal에 다음과 같은 명령어를 입력한다.  
 ```
    tensorboard —logdir results
 ```
 
 
 
#### Training : PPO+ring
1. 원형 도로에서 PPO 알고리즘 기반으로 RL agent를 Learning 시키기 위해 터미널에 다음과 같은 명령어를 입력한다. 
```
   cd flow-autonomous-driving
   python train_rllib.py singleagent_ring 
```   

2. Training이 끝난 후 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다.
```
   cd
   cd flow-autonomous-driving
   python visualizer_rllib.py ./ray_results/singleagent_ring/experiment_name  number_of_checkpoints
```   
 -  experiment_name: Learning을 시작할 때, 생성된 폴더의 이름
 - number_of_checkpoints: experiment_name 폴더 안에 생성된         checkpoint 폴더의 이름을 의미한다. visualizing을 하고자 하는    
   checkpoint(숫자)를 입력한다. 
3. Training이 끝난 후, 그래프를 확인을 하고자 하면 다음과 같은 명령어를 입력한다. (명령어를 친 후 위의 experiment_name을 찾아서 본다.)
```
  cd
  cd ray_results
  tensorboard --logdir singleagent_ring
```

#### High performance Example Code 
1. High performance Example Code의 training 결과를 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
   cd 
   cd flow-autonomous-driving/2020SUMMER/Code/Ring_Network-PPO
   python visualizer_rllib.py ./results/PPO_WaveAttenuationPOEnv-v0_0_2020-08-22_21-45-12bc33z2g0 1500
```   

2. High performance Example Code의 Training 결과를 tensorboard를 이용하여 확인하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
  tensorboard —logdir results
```

#### Training : TD3+ring
1. 원형 도로에서 TD3 알고리즘 기반으로 RL agent를 Learning 시키기 위해 터미널에 다음과 같은 명령어를 입력한다. 
```
   cd flow-autonomous-driving
   python train_rllib.py singleagent_ring --algorithm ddpg
```   

2. Training이 끝난 후 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다.
```
   cd
   cd flow-autonomous-driving
   python visualizer_rllib.py ./ray_results/singleagent_ring/experiment_name/  number_of_checkpoints
```   
 -  experiment_name: Learning을 시작할 때, 생성된 폴더의 이름
 - number_of_checkpoints: experiment_name 폴더 안에 생성된         checkpoint 폴더의 이름을 의미한다. visualizing을 하고자 하는    
   checkpoint(숫자)를 입력한다. 
3. Training이 끝난 후, 그래프를 확인을 하고자 하면 다음과 같은 명령어를 입력한다. (명령어를 친 후 위의 experiment_name을 찾아서 본다.)
```
  cd
  cd ray_results
  tensorboard --logdir singleagent_ring
```

#### High performance Example Visualizing Code 
1. High performance Example Code의 training 결과를 visualizing 하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
   cd ~/flow-autonomous-driving
   python visualizer_rllib.py ./Results/best_td3_ring/uniform_motion_TD3/TD3_WaveAttenuationPOEnv-v0_0_2021-02-04_17-40-56i6kxtghu 800
```   

2. High performance Example Code의 Training 결과를 tensorboard를 이용하여 확인하려면, terminal에 다음과 같은 명령어를 입력한다. 
```
  tensorboard —logdir results
```

#### High performance Example Regenerating Code
1. High performance Example Code를 이용해 Agent를 학습시키고 싶다면 다음과 같은 명령어를 입력한다.  
```
   cd flow-autonomous-driving/Code/Ring_Network-TD3
   python train_rllib.py my_singleagent_ring 
```   

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
