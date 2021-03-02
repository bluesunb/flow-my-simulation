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
