import logging
import numpy as np
import torch as T 
from torch.autograd import Variable
import torch.nn.functional as F 
from crowd_nav.policy.network import MLPActorCritic
from crowd_nav.policy.multi_agent import MultiRobot
from crowd_nav.utils.replaybuffer import MutiAgentReplyBuffer
from copy import deepcopy



class MRLPolicy(MultiRobot):

    def __init__(self, config, env,device=None):
        super().__init__(config)
        self.models = []
        self.n_robot = config.getint('train','robot_num')
        self.mem_size = config.getint('rl','mem_size')
        self.batch_size = config.getint('rl','batch_size')
        self.tau = config.getfloat('rl','tau')
        self.action_shape = config.getint('ob','action_dim')
        self.n_human = config.getint('train','human_num')
        self.time_step = env.time_step
        self.v_pref = env.robots[0].v_pref
        self.set_device(device)
        for i in range(self.n_robot):
            self.models.append(MLPActorCritic(config,self.device))
        self.set_memory([self.action_shape],[self.n_robot+self.n_human-1,self.models[0].input_dims])
        self.env = env
    def set_device(self, device):
        self.device = device

    def get_models(self):
        return self.models

    def set_memory(self,action_shape,state_shape):
        self.memory = MutiAgentReplyBuffer(self.mem_size, self.batch_size, self.n_robot, 
                                           action_shape, state_shape, self.device)
        
    def learn(self):
        if not self.memory.ready():
            return 

        data = self.memory.sample_batch()
        states, new_states, actions, rewards,\
        dones = data['state'],data['state2'],data['act'],data['rew'],data['done']
        old_actions = actions.view(actions.shape[0],-1)
        self_state = states[:,:,0,:self.models[0].self_state_dim]
        new_self_state = states[:,:,0,:self.models[0].self_state_dim]
        next_actions = []
        new_actions = []
        for i in range(self.n_robot):
            next_action = self.models[i].target_actor(new_self_state[:,i,:])
            new_action = self.models[i].actor(self_state[:,i,:])
            new_action = Variable(new_action)
            next_actions.append(next_action)
            new_actions.append(new_action)
        next_actions = T.cat([act for act in next_actions],dim=1)
        new_actions = T.cat([act for act in new_actions], dim=1)
        for i, model in enumerate(self.models):
            critic_value_ = model.target_critic(new_states[:,i], next_actions).flatten()
            critic_value = model.critic(states[:,i], old_actions).flatten()
            target = rewards[:,i] + (1-dones[:,i])*pow(model.gamma, \
                     self.time_step*self.v_pref)*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            model.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            model.critic.optimizer.step()

            actor_loss = model.critic(states[:,i], new_actions).flatten()
            actor_loss = -T.mean(actor_loss)
            model.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            model.actor.optimizer.step()

            model.update_network_parameters(tau=self.tau)
    
    def states_transform(self,ob, robots):
        humans_ob , robots_ob = ob['human'], ob['robot']
        states = np.zeros((3, 7, 13))
        self_states = np.zeros((3,6))
        for i, robot in enumerate(robots):
            obs = deepcopy(humans_ob)
            for robot_ob in robots_ob:
                if robot.get_full_state().list != robot_ob.list:
                    obs.append(robot.get_observable_state())
            state = np.concatenate([np.array([robot.get_full_state() + ob])
                                    for ob in obs], axis = 0)
            state = T.from_numpy(state)
            state = self.rotate(state)
            state = state.numpy()
            states[i] = state
            self_states[i] = robots_ob[i].list[:6]
        return states, self_states
       
    def choose_actions(self,self_states):
        actions = []
        for i, model in enumerate(self.models):
            if self.env.robots[i].finish_flag == True:
                action = [0,0]
            else:
                action  = model.choose_action(self_states[i])
            actions.append(action)
        return actions

    def reached_destination(self,state):
        if np.linalg.norm((state.py - state.gy, state.px - state.gx)) < state.radius:
            return True
        else:
            return False




    
        






        
    

            






