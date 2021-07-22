import numpy as np
import torch as T

class MutiAgentReplyBuffer:
    def __init__(self, capacity, batch_size, n_robot, action_shape, state_shape, device):
        # init 
        self.capacity = capacity
        self.n_robot = n_robot
        self.batch_size = batch_size
        self.state_buf = np.empty((capacity, n_robot, *state_shape), dtype=np.float32)
        self.state2_buf = np.empty((capacity, n_robot, *state_shape), dtype=np.float32)
        self.act_buf = np.empty((capacity, n_robot, *action_shape), dtype=np.float32)
        self.rew_buf = np.empty((capacity, n_robot), dtype=np.float32)
        self.done_buf = np.empty((capacity, n_robot), dtype=np.float32)
        self.mem_cntr = 0
        self.device = device


    def push(self, state, state2, action,reward, done):
        #store
        idx = self.mem_cntr % self.capacity
        self.state_buf[idx] = state
        self.state2_buf[idx] = state2
        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.done_buf[idx] = done
        self.mem_cntr +=1
    
    def sample_batch(self):
        size = min(self.mem_cntr, self.capacity)
        idxs = np.random.randint(0, size, size=self.batch_size)
        batch = dict(state=self.state_buf[idxs],
                     state2=self.state2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: T.as_tensor(v, dtype=T.float32, device=self.device) 
                                                    for k,v in batch.items()}

    def ready(self):
        if self.mem_cntr>=self.batch_size:
            return True



           



         