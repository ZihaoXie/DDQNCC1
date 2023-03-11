import torch
import numpy as np
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from CybORG.Agents.Wrappers import *
from Agents.DQNAgent import DQNAgent
from utils import plot_learning_curve
import inspect
import os

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario1b.yaml'


# check if cuda is available
def cuda():
    print("CUDA: " + str(torch.cuda.is_available()))

#  训练episode2200 每一轮的步数100 1000步更新一次目标网络 经验池大小是5000  模型的学习率，用于控制模型参数的更新速度。默认为0.0001
#  ε-greedy策略中的epsilon衰减率。默认为0.000005
def train_DuelingDDQN(red_agent=RedMeanderAgent, num_eps=2200, len_eps=100, replace=1000, mem_size=5000,
lr=0.0001, eps_dec=0.000005, eps_min=0.05, gamma=0.99, batch_size=32, epsilon=1, chkpt_dir="model_meander"):

    CYBORG = CybORG(PATH, 'sim', agents={
        'Red': red_agent
    })
    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")
    # 保存机器学习模型的目录
    model_dir = os.path.join(os.getcwd(), "Models", chkpt_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) # 如果目录不存在 ，创建一个目录

    agent = DuelingDDQNAgent(gamma=gamma, epsilon=epsilon, lr=lr,
                     input_dims=(env.observation_space.shape), # 各种属性，观测空间 等等
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                     batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                     chkpt_dir=model_dir, algo='DuelingDDQNAgent',
                     env_name='Scenario1b')
    # 最佳值，设的初始值为 负无穷大，比任何值都小，目的是 这样做可以第一次更新的时候 best _score 会被正确的更新
    best_score = -np.inf

    # not using a checkpoint (i.e new model) 新模型
    load_checkpoint = False  # 不加载之前保存的模型参数，重新训练一个新的模型
    if load_checkpoint:
        agent.load_models()

    n_steps = 0  # 步数的计数器，每次更新的模型的时候，他的值会加上当前步数的值
    scores, eps_history, steps_array = [], [], []
    # score 变量是一个空列表，用于存储每个回合（episode）的得分。
    # 在训练过程中，每完成一个回合，就会将这个回合的得分添加到 scores 列表中。

    # eps_history 变量也是一个空列表，用于存储每个回合的探索率（epsilon）。
    # 在训练过程中，每个回合的探索率都可能不同，因此需要将每个回合的探索率记录下来，以便后续的分析和可视化。

    # steps_array 变量同样是一个空列表，用于存储每个回合中所用的步数
    # 在训练过程中，每个回合的步数都可能不同，因此需要将每个回合的步数记录下来，以便后续的分析和可视化。





    for i in range(num_eps): # 循环的次数为num_eps 2200  每个episode的长度为len_eps  100步
        score = 0 #  重置得分
        # need to reset the environment at the end of the episode (this could also be done by using end_episode() of the red agent)
        # 需要在episode结束时重置环境（这也可以通过使用红色代理的 end_episode() 来完成）
        observation = env.reset()  #  每个episode结束（开始）的时候重置 obs
        for j in range(len_eps):
            action = agent.get_action(observation)  #agent 根据当前观察值 并采取行动
            observation_, reward, done, info = env.step(action=action) # 行动之后获得下一时刻的观测，奖励 完成 info
            score += reward
            if not load_checkpoint: # 没有加载训练好的模型  ， load_checkpoint == false ，则会存储经验
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.train()  # 训练
            observation = observation_  # 下一时刻的观测， 赋值
            n_steps += 1
        scores.append(score)  # score 列表增加一列
        steps_array.append(n_steps)  # 训练步数

        avg_score = np.mean(scores[-100:])  # 计算平均得分情况
        print('episode: ', i,'score: ', score,  #  输出这是第几个episode reward的情况，平均得分，最佳得分情况，epsilon值 步数
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        # keep track of best score to see if we are converging
        # 跟踪最佳分数以查看我们是否收敛
        if avg_score > best_score:  # 如果平均的reward 比最好的reward 好的话，则平均的reward 就赋值给 best
            best_score = avg_score

        eps_history.append(agent.epsilon)  # agent 历史的epsilon

    # plot learning curves (moving average over last 20 episodes and shows epsilon decreasing)
    # will generate png to the models directory
    # learning curves are misleading since epsilon = 0.05 means 5% chance of random action, it should be evaluated every 200 episodes without epsilon
    plot_learning_curve(steps_array, scores, eps_history, os.path.join(model_dir, "plot.png"))
    # 绘制 step score  and eps_history
    # save model so we can use it after training complete
    # it will be stored in model_dir (both models [i.e the target one also so we can train again if needed])
    agent.save_models()


# this is used to see how well the model performs (without epsilon)
# and to generate test cases to see what the model is doing
def test_DDQN(eps_len=100, num_eps=10, chkpt_dir="model_b_line", red_agent=B_lineAgent):
    CYBORG = CybORG(PATH, 'sim', agents={
        'Red': red_agent
    })
    env = ChallengeWrapper(env=CYBORG, agent_name="Blue")

    model_dir = os.path.join(os.getcwd(), "Models", chkpt_dir)
    # the default epsilon is 0. we also don't need to define most hyperparamters since all we will do is agent.get_action()
    agent = DuelingDDQNAgent(chkpt_dir=model_dir, algo='DuelingDDQNAgent',
                     env_name='Scenario1b')
    # gets the checkpoint from model_dir
    agent.load_models()

    scores = []
    
    for i in range(num_eps):
        s = []
        a = []
        observation = env.reset()
        for j in range(eps_len):
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action=action)
            s.append(reward)
            a.append((str(env.get_last_action('Blue')), str(env.get_last_action('Red'))))
        total_score = np.sum(s)
        scores.append(total_score)
        print('score: ', total_score)
        print('actions: ', a)
    avg_score = np.mean(scores)
    print('average score: ', avg_score)


if __name__ == '__main__':
    cuda()
    # we should tune hyperparameters here (with random search)
    train_DuelingDDQN(red_agent=RedMeanderAgent, num_eps=2000, len_eps=100, replace=5000, mem_size=5000, lr=0.0001, eps_dec=0.000005, eps_min=0.05, gamma=0.99, batch_size=32, epsilon=1, chkpt_dir="model_meander3")
    #test_DDQN(chkpt_dir="model_b_line", red_agent=B_lineAgent)
