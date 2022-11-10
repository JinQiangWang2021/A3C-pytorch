import random
import torch
import torch.optim as optim
import multiprocessing as mp
import torch.multiprocessing as mp
from multiprocessing import Process
from utils.untils import ValueNetwork,ActorDiscrete,ActorContinous
from algorithm.worker import Worker

GAMMA = 0.99
LR = 1e-4
GLOBAL_MAX_EPISODE = 5000

class A3C():
    def __init__(self,env,continuous,state_size,action_size):
        self.max_episode=GLOBAL_MAX_EPISODE
        self.global_episode = mp.Value('i', 0)  # 进程之间共享的变量
        self.global_epi_rew = mp.Value('d',0)
        self.rew_queue = mp.Queue()
        self.worker_num = mp.cpu_count()
        # self.worker_num = 2

        # define the global networks
        self.global_valueNet= ValueNetwork(state_size,1)
        # global 的网络参数放入 shared memory，以便复制给各个进程中的 worker网络
        self.global_valueNet.share_memory()

        if continuous:
            self.global_policyNet = ActorContinous(state_size, action_size)
        else:
            self.global_policyNet = ActorDiscrete(state_size, action_size)
        self.global_policyNet.share_memory()

        # global optimizer
        self.global_optimizer_policy = optim.Adam(self.global_policyNet.parameters(), lr=LR)
        self.global_optimizer_value = optim.Adam(self.global_valueNet.parameters(),lr=LR)

        # define the workers
        self.workers=[Worker(env,continuous,state_size,action_size,i,
                             self.global_valueNet,self.global_optimizer_value,
                             self.global_policyNet,self.global_optimizer_policy,
                             self.global_episode,self.global_epi_rew,self.rew_queue,
                             self.max_episode,GAMMA) 
                      for i in range(self.worker_num)]

    def train_worker(self):
        scores=[]
        [w.start() for w in self.workers]
        while True:
            r = self.rew_queue.get()
            if r is not None:
                scores.append(r)
            else:
                break
        [w.join() for w in self.workers]
        return scores

    def save_model(self):
        torch.save(self.global_valueNet.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_policyNet.state_dict(), "a3c_policy_model.pth")




