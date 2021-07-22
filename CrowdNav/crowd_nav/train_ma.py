import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import git
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.multi_robot_rl import MRLPolicy
from crowd_nav.utils.runner import Explorer


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')


    args = parser.parse_args()
    log_file = os.path.join(args.output_dir, 'output.log')
    il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

    #configure logging 
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
    format = '%(asctime)s, %(levelname)s: %(message)s',  datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure env
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    env.robots = []
    for i in range(env.robot_num):
        env.robots.append(Robot(env.config, 'robots')) # init robots

    # configure agents 
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    mrl_policy = MRLPolicy(policy_config, env, device)
    env.set_policy(mrl_policy) # set policy

    # configure training parameters 
    rl_train_config = configparser.RawConfigParser()
    rl_train_config.read(args.policy_config)
    rl_train_episodes = rl_train_config.getint('rl','trainning_episodes')
    explorer = Explorer(env,device)
    for index in range(env.robot_num):
        critic_weight_file = os.path.join(args.output_dir, f'Agent{index}_critic_model.pth')
        actor_weight_file = os.path.join(args.output_dir, f'Agent{index}_actor_model.pth')
    for i in range(rl_train_episodes):
        explorer.run('train',1, i)

        if i % 100 ==0:
            for model in mrl_policy.models:
                model.save_weight(critic_path=critic_weight_file, 
                                  actor_path=actor_weight_file )
            
    
if __name__ == '__main__':
    main()
