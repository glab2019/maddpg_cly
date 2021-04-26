import numpy as np


class BrainInfo:
    def __init__(self, vector_observations=None,
                 reward=None, agents=None,
                 feedback=None, local_done=False, memory=None, vector_action=None):
        """
        Describes experience at current step of all agents linked to a brain.  描述与大脑相关的所有代理当前步骤的经验。
        """
        self.vector_observations = vector_observations
        self.memories = memory
        self.rewards = reward
        self.agents = agents
        self.local_done = local_done
        self.feedback = feedback
        self.previous_earning = np.ones((1, self.agents.size)) * 0
        self.previous_action = np.zeros((1, self.agents.size))


class Platform(object):
    def __init__(self, config):
        self.n_agent = config['n_agent']
        self.n_task = config['n_task']
        self.n_stacked_observation = config["n_stacked_observation"]
        self.price_bound = config['price_bound']
        self.weights_bound = config['weights_bound']
        self.task_bound = config['task_budget_bound']
        self.time_bound = config['time_budget_bound']
        self.expected_earning_bound = config['expected_earning_bound']
        self.penalty_factor = config['penalty_factor']
        if config['weights'] is None:
            self.weights = np.random.randint(self.weights_bound[0], self.weights_bound[1], (self.n_agent, self.n_task))
        else:
            #self.weights = np.array(config['weights'])
            self.weights=self.get_sij()*7
            #self.weights = np.array(config['weights'])*10
            #self.weights = np.array(config['weights'])/10+1
        self.strategy = np.zeros((self.n_agent, self.n_task))
        self.task_budget = np.random.randint(self.task_bound[0], self.task_bound[1], (self.n_task, 1))
        self.time_budget = np.random.randint(self.time_bound[0], self.time_bound[1], (self.n_agent, 1))
        self.expected_earning = \
            np.random.randint(self.expected_earning_bound[0], self.expected_earning_bound[1], (self.n_agent, 1))
        self.rewards = None

    def reset(self):
        # self.weights = np.random.randint(self.weights_bound[0], self.weights_bound[1], (self.n_agent, self.n_task))
        self.strategy = np.zeros((self.n_agent, self.n_task))
        # self.time_budget = np.random.randint(self.time_bound[0], self.time_bound[1], (self.n_agent, 1))
        # self.task_budget = np.random.randint(self.task_bound[0], self.task_bound[1], (self.n_task, 1))

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