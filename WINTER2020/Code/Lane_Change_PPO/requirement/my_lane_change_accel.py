from flow.envs.ring.lane_change_accel import LaneChangeAccelEnv
from flow.core import myrewards as rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

from collections import defaultdict
from pprint import pprint

class MyLaneChangeAccelEnv(LaneChangeAccelEnv):
    """
    LC 학습을 위해 discrete한 action과 새로운 reward를 추가한 환경.
    """
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.accumulated_reward = None
        self.last_lane = None
        self.log = None

    def _to_lc_action(self, rl_action):
        """Make direction components of rl_action to discrete"""
        if rl_action is None:
            return rl_action
        for i in range(1, len(rl_action), 2):
            if rl_action[i] < -1 + 2 / 3:
                rl_action[i] = -1
            elif rl_action[i] >= -1 + 4 / 3:
                rl_action[i] = 1
            else:
                rl_action[i] = 0
        return rl_action

    def _apply_rl_actions(self, actions):
        actions = self._to_lc_action(actions)

        acceleration = actions[::2]
        direction = actions[1::2]

        self.last_lane = self.k.vehicle.get_lane(self.k.vehicle.get_rl_ids())

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]

        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def compute_reward(self, rl_actions, **kwargs):
        lc_action = self._to_lc_action(rl_actions)
        reward, rwds = rewards.full_reward(self, lc_action)
        # self.evaluate_rewards(lc_action, self.initial_config.reward_params.keys())

        # if self.accumulated_reward is None:
        #     self.accumulated_reward = defaultdict(int)
        # else:
        #     for k in rwds.keys():
        #         self.accumulated_reward[k] += rwds[k]
        #
        # if self.time_counter == self.env_params.horizon \
        #         + self.env_params.warmup_steps - 1:
        #     print('=== now reward ===')
        #     pprint(dict(rwds))
        #     print('=== accumulated reward ===')
        #     pprint(dict(self.accumulated_reward))

        return reward

    def evaluate_rewards(self, lc_action, args):

        vehicle = self.k.vehicle

        if len(args) == 0:
            return
        if self.log is None:
            self.log = defaultdict(list)

        rls = vehicle.get_rl_ids()
        if 'rl_action_penalty' in args:
            if lc_action is not None and self.last_lane == vehicle.get_lane(rls) \
                and any(lc_action[1::2]):
                self.log['rl_action_penalty'].append(sum(lc_action))
                # print(f'[ACT]: {lc_action}, {self.last_lane, vehicle.get_lane(rls)}')
        if 'unsafe_penalty' in args:
            lc_taken = [rl for rl in rls if self.time_counter == vehicle.get_last_lc(rl)]
            if any(lc_taken):
                self.log['unsafe_penalty'].extend(vehicle.get_tailway(lc_taken))
                # print(f'[USF]: {vehicle.get_tailway(lc_taken)}')
        if 'dc3_penalty' in args:
            # follower_accel = vehicle.get_accel(vehicle.get_follower(vehicle.get_rl_ids()[0]))
            # if follower_accel is not None and follower_accel < 0:
            #     print(f'[dd]: {follower_accel}')
            # lc_taken = [rl for rl in rls if self.time_counter == vehicle.get_last_lc(rl)]
            # if any(lc_taken):
            #     self.log['dc3_penalty'].extend([vehicle.get_accel(follower) for follower in vehicle.get_follower(lc_taken)])
            accels = [vehicle.get_accel(vid) or 0 for vid in vehicle.get_human_ids()]
            accels = np.array(accels).clip(max=0)
            self.log['dc3_penalty'].extend(accels.tolist())

        if self.time_counter == self.env_params.horizon \
                + self.env_params.warmup_steps - 1:
            print(f'[rlp]: {sum(self.log["rl_action_penalty"])}')
            print(f'[unsf]: {sum(self.log["unsafe_penalty"]) / (len(self.log["unsafe_penalty"]) or float("inf"))}')
            print(f'[unsf]: {sum(np.array(self.log["unsafe_penalty"]) < 5)}')
            print(f'[dc3]: {sum(self.log["dc3_penalty"]) / (len(self.log["dc3_penalty"]) or float("inf"))}')
            print(f'[dc3]: {sum(np.array(self.log["dc3_penalty"]) < -0.2)}')


class TestLCEnv(MyLaneChangeAccelEnv):

    @property
    def action_space(self):
        """
        return: Tuple(Box, MultiDiscrete)
        0: direction -1
        1: direction 0
        2: direction 1
        """
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel)] * self.initial_vehicles.num_rl_vehicles  # lower bound of acceleration
        ub = [max_accel] * self.initial_vehicles.num_rl_vehicles  # upper bound of acceleration

        lc_b = [3] * self.initial_vehicles.num_rl_vehicles  # boundary of lane change direction

        return Tuple((Box(np.array(lb), np.array(ub), dtype=np.float32), MultiDiscrete(np.array(lc_b))))

    def compute_reward(self, rl_actions, **kwargs):
        reward, rwds = rewards.full_reward(self, rl_actions)

        # if self.accumulated_reward is None:
        #     self.accumulated_reward = defaultdict(int)
        # else:
        #     for k in rwds.keys():
        #         self.accumulated_reward[k] += rwds[k]
        #
        # if self.time_counter == self.env_params.horizon \
        #             + self.env_params.warmup_steps - 1:
        #     print('=== now reward ===')
        #     pprint(dict(rwds))
        #     print('=== accumulated reward ===')
        #     pprint(dict(self.accumulated_reward))

        return reward

    def _apply_rl_actions(self, actions):
        acceleration, raw_direction = actions
        direction = raw_direction - 1

        self.last_lane = self.k.vehicle.get_lane(self.k.vehicle.get_rl_ids())


        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.k.vehicle.get_last_lc(veh_id)
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

class MyLaneChangeAccelPOEnv(MyLaneChangeAccelEnv):

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(6*self.net_params.additional_params["lanes"] + 3* self.k.vehicle.num_rl_vehicles,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # NOTE: this works at only single agent environmnet
        rl = self.k.vehicle.get_rl_ids()[0]
        lane_followers = self.k.vehicle.get_lane_followers(rl)
        lane_leaders = self.k.vehicle.get_lane_leaders(rl)

        lane_followers_speed = self.k.vehicle.get_lane_followers_speed(rl)
        lane_leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl)
        lane_followers_pos = [self.k.vehicle.get_x_by_id(follower) for follower in lane_followers]
        lane_leaders_pos = [self.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]
        follower_lanes = [self.k.vehicle.get_lane(follower) for follower in lane_followers]
        leader_lanes = [self.k.vehicle.get_lane(leader) for leader in lane_leaders]

        speeds = [speed/max_speed
                  for speed in lane_followers_speed+lane_leaders_speed+[self.k.vehicle.get_speed(rl)]]
        positions = [pos/length
                     for pos in lane_followers_pos+lane_leaders_pos+[self.k.vehicle.get_x_by_id(rl)]]
        lanes = [lane/max_lanes
                 for lane in follower_lanes+leader_lanes+[self.k.vehicle.get_lane(rl)]]

        return np.array(speeds + positions + lanes)

class TestLCPOEnv(TestLCEnv):
    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(6 * self.net_params.additional_params["lanes"] + 3 * self.k.vehicle.num_rl_vehicles,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        # normalizers
        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        # NOTE: this works at only single agent environmnet
        rl = self.k.vehicle.get_rl_ids()[0]
        lane_followers = self.k.vehicle.get_lane_followers(rl)
        lane_leaders = self.k.vehicle.get_lane_leaders(rl)

        lane_followers_speed = self.k.vehicle.get_lane_followers_speed(rl)
        lane_leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl)
        lane_followers_pos = [self.k.vehicle.get_x_by_id(follower) for follower in lane_followers]
        lane_leaders_pos = [self.k.vehicle.get_x_by_id(leader) for leader in lane_leaders]
        follower_lanes = [self.k.vehicle.get_lane(follower) for follower in lane_followers]
        leader_lanes = [self.k.vehicle.get_lane(leader) for leader in lane_leaders]

        speeds = [speed / max_speed
                  for speed in lane_followers_speed + lane_leaders_speed + [self.k.vehicle.get_speed(rl)]]
        positions = [pos / length
                     for pos in lane_followers_pos + lane_leaders_pos + [self.k.vehicle.get_x_by_id(rl)]]
        lanes = [lane / max_lanes
                 for lane in follower_lanes + leader_lanes + [self.k.vehicle.get_lane(rl)]]


        return np.array(speeds + positions + lanes)

class MYLC(MyLaneChangeAccelEnv):
    def compute_reward(self, rl_actions, **kwargs):
        lc_action = self._to_lc_action(rl_actions)
        reward = rewards.total_lc_reward(self, lc_action)

        if self.accumulated_reward is None:
            self.accumulated_reward = defaultdict(int)
        else:
            for k in reward.keys():
                self.accumulated_reward[k] += reward[k]

        if self.time_counter == self.env_params.horizon \
            + self.env_params.warmup_steps -1:
            print('=== now reward ===')
            pprint(dict(reward))
            print('=== accu reward ===')
            pprint(dict(self.accumulated_reward))

        return sum(reward.values())