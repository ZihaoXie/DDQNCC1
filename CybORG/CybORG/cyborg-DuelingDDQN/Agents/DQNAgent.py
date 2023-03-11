# Checkout https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN
# No changes are made except ensuring it fits with the CybORG BaseAgent

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
import torch as T
import numpy as np
from DQN.DeepQNetwork import DeepQNetwork
from DQN.ReplayBuffer import ReplayBuffer

class DuelingDDQNAgent(BaseAgent):
    def __init__(self, gamma=0.9, epsilon=0, lr=0.1, n_actions=41, input_dims=(52,),
                 mem_size=1000, batch_size=32, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo='DuelingDDQN', env_name='Scenario1b', chkpt_dir='chkpt', load=False):  #是否加载已有模型参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon   # 初始贪婪度 0 逐步递减
        self.lr = lr  # 学习率 0.1
        self.n_actions = n_actions  # 动作空间 41
        self.input_dims = input_dims  # 输入网络的维度
        self.batch_size = batch_size  # batch 大小
        self.eps_min = eps_min  # 最小贪婪度
        self.eps_dec = eps_dec  # 贪婪度衰减度
        self.replace_target_cnt = replace  # 目标网络更新频率
        self.algo = algo  # 算法 可以是DQN也可以是DDQN
        self.env_name = env_name    # 环境的名字
        self.chkpt_dir = chkpt_dir  # 模型保存的路径
        self.action_space = [i for i in range(n_actions)]  # 动作空间
        self.learn_step_counter = 0  # 学习步数计数器

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)  #  记忆大小 ，输入维度大小， 动作空间大小

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions, # Q网络
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_eval',
                                        chkpt_dir=self.chkpt_dir)
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,  # Q target
                                        input_dims=self.input_dims,
                                        name=self.env_name+'_'+self.algo+'_q_next',
                                        chkpt_dir=self.chkpt_dir)

    # if epsilon=0 it will just use the model
    # 这段代码实现了epsilon-greedy策略 贪心 ，用于选择代理(agent)在当前状态(observation)下执行的行动(action)。
    def get_action(self, observation, action_space=None):
        if np.random.random() > self.epsilon:  # 如果随机数大于 epslion 则将当前的观察作为输入，计算Q网络中的Q值，并选择最大的动作输出
            state = np.array([observation], copy=False, dtype=np.float32)
            state_tensor = T.tensor(state).to(self.q_eval.device)
            _, advantages = self.q_eval.forward(state_tensor)  # 与我修改的版本的差距就在选择动作的时候
            action = T.argmax(advantages).item()  # 他是根据 advantage的最大值选动作

            # DDQN 选动作 forward 网络直接返回 actions
            # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # actions = self.q_eval.forward(state)
            # action = T.argmax(actions).item()   # 返回Q值最大的动作
        else:
            action = np.random.choice(self.action_space)  # 否则回随机选择一个动作

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)  # 存储经验

    def sample_memory(self):  # 随机采样一批memory 并返回成pytorch张量
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:  # 每1000步更新一下 目标网络
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min # 如果epsilon 大于 最小值
                                                                            # 则减去一个 dec，如果小于最小值，则赋值为最小值

    def train(self):
        if self.memory.mem_cntr < self.batch_size:   # 如果经验池没有存储够足够多的经验 则返回
            return
        self.q_eval.optimizer.zero_grad()  # 优化器梯度置0
        self.replace_target_network()  # 调用replace函数来更新目标网络
        states, actions, rewards, states_, dones = self.sample_memory()  # 从经验池中随机采样一批经验
        indices = np.arange(self.batch_size)  # 生成一个[0,1,2....batch_size -1]大小的数组，用来索引
                                              #  一个批次的样本

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)
        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))) # 其实跟我的一模一样

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))
        # q_pred = self.q_eval.forward(states)[indices, actions]  # 使用当前网络预测当前状态下所有动作的Q值
        # q_next = self.q_next.forward(states_)  # 使用目标网络q_next预测下个状态采取所有动作的Q值
        #q_eval = self.q_eval.forward(states_)  # 使用当前网络q_eval 预测下个状态采取所有动作的Q值 ，这里是为了计算double dqn的 Q值估计
        max_actions = T.argmax(q_eval, dim=1)  # 找到当前网络中最大Q值的动作
        q_next[dones] = 0.0  #如果下个状态是done的话 下个状态的Q值为O
        q_target = rewards + self.gamma*q_next[indices, max_actions] # 根据 Q target和 Q网络计算出的最大动作
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device) #计算损失
        loss.backward()  # 反向传播
        self.q_eval.optimizer.step()  # 更新当前网络中的参数 更新模型
        self.learn_step_counter += 1   # 计数器加一 表示完成了学习
        self.decrement_epsilon()  # 降低探索率

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()