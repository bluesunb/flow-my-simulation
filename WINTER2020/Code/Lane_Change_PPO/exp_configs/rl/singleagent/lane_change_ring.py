from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
from flow.envs.ring.my_lane_change_accel import *
from flow.networks.lane_change_ring import RingNetwork
import math

import os

# name of this file
current_file_name_py = os.path.abspath(__file__).split('/')[-1]
# remove file extension
current_file_name = current_file_name_py[:-3]

HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 2

vehicles = VehicleParams()

vehicles.add(
    veh_id='outline',
    acceleration_controller=(IDMController, {'v0': 2}),  # To make controlled vehicles to drive along the speed 2(m/s)
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=8,
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0  # To train safe driving, set 0
    )
)

vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    initial_speed=5,
    num_vehicles=1,
)

flow_params = dict(
    exp_tag=current_file_name,
    seed=1014,
    env_name=TestLCEnv,
    network=RingNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=False,
        seed=1014,
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=150,
        clip_actions=False,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "ring_length": [270, 300],
            "lane_change_duration": 0,
            "target_velocity": 10,
            'sort_vehicles': False
        },
    ),
    net=NetParams(
        additional_params={
            "length": 300,
            "lanes": 2,
            "speed_limit": 30,
            "resolution": 40,
        },
    ),

    veh=vehicles,
    initial=InitialConfig(
        spacing='custom',
        reward_params={
            'rl_mean_speed': 0.18,
            'simple_lc_penalty': 0.3,
            'rl_action_penalty': 0.5,
            'unsafe_penalty': 0.6,
            'dc3_penalty': 1,
        },
    ),
)
