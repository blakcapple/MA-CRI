import torch
import torch.nn as nn
import logging
from crowd_nav.policy.cadrl import mlp
import torch.optim as optim
from copy import deepcopy

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, attention_dims,
                action_dim, robot_num, mlp3_dims, with_global_state,beta):
        super().__init__()
        self.action_dim = action_dim
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_states = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1]*2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim + self.action_dim*robot_num
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = None
    
    def forward(self,state,action):
        """
        First transform the world coordinates to self-centric coordinates and 
        then do forward computation
        param state: tensor of shape (batch_size, num of agents-1, length of a rotated state)
        """

        size = state.shape

        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.contiguous().view(-1, size[2]))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_states:
            # 计算注意力权重
            global_state = torch.mean(mlp1_output.view(size[0],size[1],-1),1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        scores_exp = torch.exp(scores)*(scores!=0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0,:,0].data.cpu().numpy()
        
        feature = mlp2_output.view(size[0], size[1],-1)
        weighted_feature = torch.sum(torch.mul(weights, feature), dim=1)
        mlp3_input = torch.cat([self_state, weighted_feature, action], dim=1)
        value = self.mlp3(mlp3_input)

        return value

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class ActorNetwork(nn.Module):
    
    def __init__(self, self_state_dim, actor_dims,beta):
        super().__init__()
        self.actor_network=mlp(self_state_dim, actor_dims)
        self.optimizer = optim.Adam(self.parameters(),lr=beta)
    
    def forward(self, self_state):

        action = self.actor_network(self_state)
        action = torch.tanh(action)
        return action
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class MLPActorCritic:
    def __init__(self, config,device):
        super().__init__()
        self.name = 'MA-SARL'
        self.set_parameter(config)
        self.configure(config)
        self.device = device 
        self.set_device(device)
    
    def set_parameter(self, config):

        self.lr_rate = config.getfloat('rl','lr_rate')
        self.action_dim = config.getint('actor', 'action_dim')
        self.robot_num = config.getint('train', 'robot_num')
        self.self_state_dim = config.getint('ob', 'self_state_dim')
        self.joint_state_dim = config.getint('ob', 'joint_state_dim')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')
        self.with_om = config.getboolean('om','with_om')
        self.max_action = config.getint('actor', 'max_action')

    def configure(self, config):
        mlp1_dims = [int(x) for x in config.get('critic','mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('critic','mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('critic','mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('critic','attention_dims').split(', ')]
        actor_dims = [int(x) for x in config.get('actor','actor_dims').split(', ')]
        self.input_dims = self.joint_state_dim + \
                (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)
        with_global_state = config.getboolean('train','with_global_state')
        self.gamma = config.getfloat('rl','gamma')
        self.critic = CriticNetwork(self.input_dims, self.self_state_dim, mlp1_dims, mlp2_dims, 
                        attention_dims,self.action_dim, self.robot_num, mlp3_dims, 
                        with_global_state, self.lr_rate)
        self.actor = ActorNetwork(self.self_state_dim,actor_dims,self.lr_rate)
        self.target_critic = deepcopy(self.critic)
        self.target_actor = deepcopy(self.actor)
        logging.info('Policy: {} {} global state'.format(self.name, 'with' if with_global_state else 'without'))

    def get_attention_weights(self):
        return self.critic.attention_weights
    
    def save_weight(self, critic_path, actor_path):
        self.critic.save_model(critic_path)
        self.actor.save_model(actor_path)
    
    def load_weight(self, critic_path, actor_path):
        self.critic.load_model(critic_path)
        self.actor.load_model(actor_path)
    
    def choose_action(self,ob):
        self_state = torch.tensor(ob, dtype=torch.float).to(self.device)
        action = self.actor(self_state)
        noise = torch.rand(self.action_dim).to(self.device)
        action = action + noise
        action = torch.clamp(action, -self.max_action, self.max_action)

        return action.detach().cpu().numpy()

    def update_network_parameters(self,tau):
        with torch.no_grad():
            for q, q_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                q_targ.data.mul_(tau)
                q_targ.data.add_((1-tau)*q.data)
            
            for p, p_targ in zip(self.actor.parameters(), self.target_actor.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1-tau)*p.data)
    
    def set_device(self,device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)
        self.target_actor.to(device)





            

