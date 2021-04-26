import numpy as np
from .Brain import BrainInfo, Platform
import cvxpy as cvx
from collections import deque
import torch
import math
from itertools import chain


epsilon = 1e-2

#wi更新时k值的大小
max_len=30
my_round=0.1
my_C=1.5
class Environment(object):
    """
    params: n_agent: the number of mobile users;
    params: all_agent_params: mobile users' params consisting of cost ci, whether to participate mi, reward ri;
    params: all_environment_params: environment's params consisting of n_agents, n_tasks, weights Wij,
            strategy xij, task_budget bj, time_budget ti, reward r_p, price_bound;
    """
    def __init__(self, agent_config=None, platform_config=None, train_config=None, result_inf=None):
        self.platform = Platform(config=platform_config)
        self._n_agent = self.platform.n_agent
        self._n_task = self.platform.n_task
        self.n_stacked_observation = self.platform.n_stacked_observation
        self.brain_info = BrainInfo(agents=np.array([i for i in range(self._n_agent)]))
        self.price_bound = self.platform.price_bound
        self.action_size = agent_config['action_size']
        self.obs_size = agent_config['obs_size']
        self.len_window = train_config['len_window']
        self.penalty_factor = platform_config['penalty_factor']
        self.alloc_thr = platform_config['allocation_threshold']
        self.welfare = dict(result_inf)
        self.welfare_episode = dict(result_inf)
        self.welfare_avg = dict(result_inf)
        self.welfare_episode_window = dict(result_inf)
        self.time_epslion = 0
        #记录以往的交易价格队列，长度为max_len
        self.wi_price=deque(maxlen=max_len)
        #记录以往的交易时间队列，长度为max_len
        self.wi_time=deque(maxlen=max_len)
        #记录以往的分配时间队列，长度为max_len
        self.wi_allocation=deque(maxlen=max_len)
        
        for i in range(max_len):
            tk=i
            t=max_len
            self.wi_time.append(math.exp(-my_round*(t-tk)))

        for key in result_inf:
            self.welfare_episode_window[key] = deque(maxlen=self.len_window)

    def get_num_agents(self):
        return self._n_agent

    def get_action_size(self):
        return self.action_size

    def reset(self, train_mode=True) -> BrainInfo:

        """
        Sends a signal to reset the environment.
        :return: AllBrainInfo: A Data structure corresponded to the initial set state of the environment.
        """
        # TODO train_mode
        self.platform.reset()
        self.brain_info.rewards = np.zeros(self._n_agent)
        self.brain_info.feedback = np.zeros(self._n_agent)
        self.brain_info.vector_observations = np.zeros((self._n_agent, self.n_stacked_observation*self.obs_size))
        self.brain_info.local_done = [False for i in range(self._n_agent)]
        self.brain_info.previous_earning = np.ones((1, self._n_agent)) * 0
        for key in self.welfare:
            self.welfare[key] = []
        actions = np.random.random((self._n_agent, self.action_size))
        self.step(actions)
        return self.brain_info

    def step(self, vector_action=None) -> BrainInfo:
        """
        Provides the environment with an action, moves the environment dynamics forward forward accordingly,
        and returns observation, state, reward information to the agent.
        :param vector_action: Agent's vector action to send to environment. Can be a scalar or vector of int/floats.
        :return: AllBrainInfo: A Data structure corresponding to the new state of the environment.
        """
        platform = self.platform
        s = self.process(vector_action=vector_action, platform=platform)
        return s

    def process(self, vector_action, platform):
        """
        input actions(n_agent * action_size),the system return the observation, rewards and local dones,
        :param vector_action:array of shape == (agent_count * action_size) =2(MU) * 1(the price of goods)
        :param platform:
        :return:
        """
        #mu=create_sij.get_mu()
        mu=self.get_mu()
        #print(mu)
        bound = platform.price_bound
        time_budget = platform.time_budget
        task_budget = platform.task_budget
        weights = platform.weights
        #print("weights",weights)
        price = np.maximum(np.array(vector_action), 0) * bound
        wi=self.update_wi()
        #print("wi",wi)
        omiga=wi.sum()/6-0.1
        ex_reward=(wi-omiga)/wi
        #print("ex_reward",ex_reward)
        #print("price",price)
        task_allocation, log_welfare = self.platform_opti(price=price, time_budget=time_budget,
                                                          task_budget=task_budget, weights=weights,wi=wi)
        #print("task_allocation",task_allocation)

        allocation = np.sum(task_allocation, axis=1)
        ti_earnings = np.sum(weights*task_allocation-price*task_allocation, axis=0)
        temp005=weights*task_allocation
        temp005[0][0]=temp005[0][0]*wi[0]
        temp005[0][1]=temp005[0][1]*wi[0]
        temp005[1][0]=temp005[1][0]*wi[1]
        temp005[1][1]=temp005[1][1]*wi[1]
        temp005[2][0]=temp005[2][0]*wi[2]
        temp005[2][1]=temp005[2][1]*wi[2]
        temp005[3][0]=temp005[3][0]*wi[3]
        temp005[3][1]=temp005[3][1]*wi[3]
        temp005[4][0]=temp005[4][0]*wi[4]
        temp005[4][1]=temp005[4][1]*wi[4]
        temp005[5][0]=temp005[5][0]*wi[5]
        temp005[5][1]=temp005[5][1]*wi[5]
        #print("temp005",temp005)
        welfare = np.sum(temp005)
        
        self.wi_price.append(price)
        self.wi_allocation.append(task_allocation)

        
        #print("wi",wi)
        self.brain_info.vector_observations[:, :(self.n_stacked_observation-1)*self.obs_size] = \
            self.brain_info.vector_observations[:, self.obs_size:]
        self.brain_info.vector_observations[:, -2] = price[:, 0]
        #print("allocation",allocation)
        self.brain_info.vector_observations[:, -1] = allocation[:]

        #向量空间加入点在这加入wi
        self.brain_info.vector_observations[:, -3] = wi[:]
        #print("self.brain_info.vector_observations",self.brain_info.vector_observations)
        allocation_proportion = allocation/np.sum(time_budget)
        mu_earning = np.squeeze(price)*allocation
        mu_earning_sum = np.sum(mu_earning)
        #print("mu_earning_sum",mu_earning_sum)
        #print("ti_earnings",ti_earnings)
        #print("welfare",welfare)

        # rewards = np.log(np.maximum(mu_earning+epsilon+1-mu*allocation, epsilon))/10
        ex_earning=mu_earning*(1+ex_reward)
        #print("mu_earning",mu_earning)
        #print("ex_earning",ex_earning)
        #这是要优化的奖励函数，需要增加信誉度
        rewards = np.log(np.maximum(ex_earning+1-np.multiply(mu,allocation), epsilon)) / 10
        #rewards = np.log(np.maximum(mu_earning+1-np.multiply(mu,allocation), epsilon)) / 10
        #奖励函数1，α和β分别代表利润和信誉度的占比
        #rewards = np.log(np.maximum(1+my_alpha*(mu_earning-np.multiply(mu,allocation))+my_beta*0.1*wi, epsilon)) / 10
        #rewards = np.log(np.maximum(1+(mu_earning-np.multiply(mu,allocation))*0.1*wi, epsilon)) / 10


        #print("mu_earning+1-np.multiply(mu,allocation)",mu_earning-np.multiply(mu,allocation))
        self.brain_info.rewards = rewards
        self.brain_info.previous_earning = np.array(mu_earning)
        self.brain_info.previous_action = np.array(vector_action)
        self.welfare["prices"].append(price)
        self.welfare["allocation"].append(task_allocation)
        self.welfare["MU_rewards"].append(rewards)
        self.welfare["MU_earnings"].append(mu_earning)
        self.welfare["MU_earnings_sum"].append(mu_earning_sum)
        self.welfare["resource_utilization"].append(np.sum(allocation_proportion))
        self.welfare["TI_earnings"].append(ti_earnings)
        self.welfare["welfare"].append(welfare)
        self.welfare["log_welfare"].append(log_welfare)
        return self.brain_info

    


    def platform_opti(self, price, time_budget, task_budget, weights,wi):
        """
        compute tasks allocation for agents, platform_rewards,
        :param price: agents price for tasks: shape(n_agents,)
        :param time_budget: time budget of each mobile user: shape(n_agent,)
        :param task_budget: budget of task initiator: shape(n_task)
        :param weights: value weights of MUs contribute to tasks
        :return:allocation_agent: shape(n_agent,) and rewards(scalar,)
        """
        #print("plat_weights",weights)

        #print("time_epslion",self.time_epslion)
        x = cvx.Variable((self._n_agent, self._n_task))
        w = np.ones((self._n_task, 1))
        #print("price",price)
        #新的权重
        Qij=weights/(price+0.1)
        #wi=np.array([[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]])
        #wi=np.array([[1.0],[1.0],[1.0]])
        wi=wi.reshape((6,1))
        #print("wi",wi)
        #objective = cvx.Maximize(cvx.sum(cvx.log(1+cvx.sum(cvx.multiply(x, Qij), axis=0))))
        objective = cvx.Maximize(cvx.sum(cvx.log(1+cvx.sum(cvx.multiply(cvx.multiply(x, weights),wi), axis=0))))
        constraints = [x >= 0, cvx.matmul(x, w) <= time_budget, cvx.matmul(x.T, price) <= task_budget]
        prob = cvx.Problem(objective, constraints)

        rewards = prob.solve(solver=cvx.SCS)
        return x.value, rewards

    def close(self):
        self.reset()
    
    def get_mu(self):
        Ci= [[0.01,0.02,0.02],[0.02,0.02,0.01],[0.01,0.03,0.01],[0.04,0.005,0.005],[0.005,0.015,0.03],[0.025,0.01,0.015]]
        Ci=np.array(Ci)
        MUci=Ci.sum(axis=1)
        #print("MUci",MUci)
        return MUci

    def get_sij(self):
        Ci= [[0.01,0.02,0.02],[0.02,0.02,0.01],[0.01,0.03,0.01],[0.04,0.005,0.005],[0.005,0.015,0.03],[0.025,0.01,0.015]]
        Ci=np.array(Ci)
        MUci=Ci.sum(axis=1)
        #print("MUci",MUci)
        #质量值系数，代表任务发起者对ni，oi，ki的重视程度分别为aj,vj,yj 
        Yj= [[1,7],[8,2],[1,3]]
        Yj=np.array(Yj)
        #质量函数，Qij=aij*ni+bij*oi+yij*ki+v
        Qij=np.dot(Ci,Yj)
        #print("Qij",Qij)
        Qi=Qij.sum(axis=0)
        #print("Qi",Qi)
        Sij=Qij
        Sij[0][0]=Qij[0][0]/Qi[0]
        Sij[1][0]=Qij[1][0]/Qi[0]
        Sij[2][0]=Qij[2][0]/Qi[0]
        Sij[3][0]=Qij[3][0]/Qi[0]
        Sij[4][0]=Qij[4][0]/Qi[0]
        Sij[5][0]=Qij[5][0]/Qi[0]

        Sij[0][1]=Qij[0][1]/Qi[1]
        Sij[1][1]=Qij[1][1]/Qi[1]
        Sij[2][1]=Qij[2][1]/Qi[1]
        Sij[3][1]=Qij[3][1]/Qi[1]
        Sij[4][1]=Qij[4][1]/Qi[1]
        Sij[5][1]=Qij[5][1]/Qi[1]

        #print("Sij",Sij)
        return Sij


    def update_wi(self):          
        get_tk=len(self.wi_price)
        if(get_tk==0):
            wi=[1.0,1.0,1.0,1.0,1.0,1.0]
            wi=np.array(wi)
            return wi


        get_tk=len(self.wi_price)
        Sij=self.get_sij()
        temp=[]
        for i in range(get_tk):
            ooo=self.wi_time[i]/(self.wi_price[i]+0.01)*my_C
            temp.append(ooo)

        my_temp=np.array(temp)
        my_sum=np.sum(my_temp)
        temp1=[]
        for i in range(get_tk):
            xij_sum=np.sum(self.wi_allocation[i],axis=1)
            Sij_xij=np.sum((Sij*self.wi_allocation[i]),axis=1)
            temp003=np.array(self.wi_price[i])
            temp003=list(chain.from_iterable(self.wi_price[i]))
            temp003=np.array(temp003)
            
            temp01=self.wi_time[i]/(temp003+0.01)*my_C*Sij_xij/xij_sum
            temp1.append(temp01)
        my_temp1=np.array(temp1)
        my_sum1=np.sum(my_temp1,axis=0)
        wi=my_sum1/my_sum
        return wi

    