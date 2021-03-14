"""
이 파일은 exp_configs/rl/singleagent/exps/exps.txt 에 reward_params를 빠르게 세팅하기 위해 만든 파일입니다.
exps.txt에 exp_config name, reward_params 를 \t 로 구분하여 적으면 그 reward_params와 exp_config_name으로 세팅된 새로운 파일을 만들어줍니다.

console을 통해 현재 위치에서 python exp_initial.py 를 실행해보세요.
"""

import os
import sys
HOMEPATH = os.path.expanduser('~')

path = '/home/bmil10/flow-autonomous-driving/WINTER2020/Code/Lane_Change_PPO/exp_configs/rl/singleagent/'


def main(flag):

    # exp_name = flag[1]

    with open(path + 'exps/exps.txt', 'r') as e:
        total = e.read()
        lines = total.split('\n')
        exps = [ein.split('\t') for ein in lines]
        e.close()

    with open(path+'exps/singleagent_basering.txt') as f:
        baselines = f.read()
        f.close()

    for i,trial in enumerate(exps):
        new_file_name, rl_mean_speed, simple_lc_penalty, rl_action_penalty, unsafe_penalty, dc3_penalty = trial
        # only_rl = (['FALSE', 'TRUE', '0'].index(only_rl))
        # only_rl = 0 if only_rl==2 else bool(only_rl)

        code_str = "        reward_params={\n" \
                   + f"            \'rl_mean_speed\': {rl_mean_speed},\n" \
                     f"            \'simple_lc_penalty\': {simple_lc_penalty},\n" \
                     f"            \'rl_action_penalty\': {rl_action_penalty},\n" \
                     f"            \'unsafe_penalty\': {unsafe_penalty},\n" \
                     f"            \'dc3_penalty\': {dc3_penalty},\n" \
                   + "\t\t},),)\n"

        # new_file_name = exp_name + f'_ring{i+1}.py'
        new_file_name += '.py'
        with open(path + new_file_name, 'w') as n:
            n.write(baselines[:])
            n.write(code_str)
            n.close()

if __name__ == "__main__":
    main(sys.argv)
