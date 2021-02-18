# time horizon of a single rollout
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
from flow.envs import WaveAttenuationEnv, LaneChangeAccelEnv, MyLaneChangeAccelEnv, LaneChangeAccelPOEnv
from flow.networks import RingNetwork

import os
current_file_name_py = os.path.abspath(__file__).split('/')[-1]
current_file_name = current_file_name_py[:-3]

HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

# We place one autonomous vehicle and 22 human-driven vehicles in the network
vehicles = VehicleParams()

vehicles.add(
    veh_id='outline',
    acceleration_controller=(IDMController, {'v0': 2}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=2,
    num_vehicles=8,
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0
    )
)


vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=2,
    num_vehicles=1,
)

flow_params = dict(
    seed=1004,
    exp_tag=current_file_name,
    env_name=LaneChangeAccelPOEnv,
    network=RingNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False,
        seed=1004
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        clip_actions=False,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "ring_length": [220, 270],
            "lane_change_duration": 10,
            "target_velocity": 5,
            'sort_vehicles': False
        },
    ),
    net=NetParams(
        additional_params={
            "length": 260,
            "lanes": 2,
            "speed_limit": 30,
            "resolution": 40,
        },
    ),

    veh=vehicles,
    initial=InitialConfig(
        spacing='my',
        lanes_distribution=1,
        additional_params={
            'inline_veh_nums': sum(['inline' in vid for vid in vehicles.ids]),
            'outline_veh_nums': sum(['outline' in vid for vid in vehicles.ids]),
        },
        reward_params={
            'only_rl': False,
            'simple_lc_penalty': 0,
            'unnecessary_lc_penalty': (-1.0, 3.0),
            'max_lc_headway': 10,
            'dc2_penalty': 0,
            'overtake_reward': 0,
            'unsafe_penalty': 0,
		},),)