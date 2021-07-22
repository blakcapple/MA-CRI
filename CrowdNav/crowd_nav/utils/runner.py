from numpy.lib.function_base import average 
import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionXY
import numpy as np

class Explorer(object):
    def __init__(self,env,device):
        self.env = env 
        self.device = device 
        self.policy = env.robots_policy

    def tuple_actions(self,actions):
        actionxy=[]
        for action in actions:
            action_tuple = ActionXY(action[0],action[1])
            actionxy.append(action_tuple)
        return actionxy

    def run(self, phase, k, episode=None):
        robot_num = self.env.robot_num
        total_rewards = []
        collision = np.zeros(robot_num)
        timeout = np.zeros(robot_num)
        success = np.zeros(robot_num)
        collision_time = []
        timeout_time = []
        success_time = []
        for i in range(robot_num):
            collision_time.append([])
            timeout_time.append([])
            success_time.append([])

        for i in range(k):
            episode_rewards = np.zeros(robot_num)
            step = 0
            obs = self.env.reset(phase)
            dones = [False]*robot_num
            while not all(dones) :
                states, self_states = self.policy.states_transform(obs, self.env.robots)
                actions = self.policy.choose_actions(self_states)
                obs2, reward, dones, infos = self.env.step(self.tuple_actions(actions))
                states2, _ = self.policy.states_transform(obs2, self.env.robots)
                self.policy.memory.push(states, states2, actions, reward, dones)
                self.policy.learn()
                obs = obs2
                for i,r in enumerate(reward):
                    episode_rewards[i] += pow(self.policy.models[i].gamma, 
                    step*self.env.time_step*self.env.robots[i].v_pref)*r
                step +=1

                for i,info in enumerate(infos):
                    if isinstance(info, Collision):
                        collision[i] +=1
                        collision_time[i].append(self.env.robot_times[i])
                    elif isinstance(info, Timeout):
                        timeout[i] +=1
                        timeout_time[i].append(self.env.robot_times[i])
                    elif isinstance(info, ReachGoal):
                        success[i] +=1
                        success_time[i].append(self.env.time_limit)
            total_rewards.append(episode_rewards)

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k 
        avg_nav_time = [0,0,0]
        avg_reward = [0,0,0]
        for i in range(robot_num):
            avg_nav_time[i] = average(success_time[i])
            avg_reward[i] = average(np.array(total_rewards)[:,i])

        extra_info = '' if episode is None else 'in episode {}'.format(episode)

        for i in range(robot_num):
            robot_info = extra_info + f' Robot {i} '
            logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), robot_info, success_rate[i], collision_rate[i], 
                            timeout_rate[i], avg_nav_time[i], avg_reward[i]))