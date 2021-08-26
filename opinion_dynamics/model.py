import numpy as np
import pandas as pd
import colorsys
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from .agent import odAgent, ncHouseholdInsured, ncStudentInsured, ncLevel1Insurance, ncLevel4Insurance
import statistics
from sklearn.preprocessing import MinMaxScaler
from numpy import random
import math


class odModel(Model):
    def __init__(self, N, min_eps, max_eps, alpha, beta, cr, agg, ee_rate, ext_range, ext_type, org, mbbn, mpe, mdata, width, height, max_iters):       
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.alpha = alpha
        self.beta = beta
        self.num_agents = N
        self.communication_regime = cr
        self.aggregation_in_HK = agg
        self.entry_exit_rate = ee_rate
        self.exteremisim_range = ext_range
        self.exteremisim_type = ext_type
        self.original = org
        self.space = ContinuousSpace(width, height, True, 0, 0)
        self.schedule = RandomActivation(self)
        self.data = mdata
        self.bbn = mbbn
        self.pe = mpe
        self.base_max_iters=max_iters
        self.max_iters = max_iters
        self.iteration = 0  
        self.promotion_signal = 'pulse'
        self.new_agent_breed = ''
        self.running = True
        self.added_new_users = []
        model_reporters = {
            'Opinion mean': lambda m: self.rpt_opinion_mean (m),
            'Opinion median': lambda m: self.rpt_opinion_median (m)
   
        }        
        agent_reporters = {
            "x": lambda a: a.pos[0],
            "y": lambda a: a.pos[1],
            "opinion": lambda a: a.opinion,
            'eps': lambda a: a.eps
        }
        self.dc = DataCollector(model_reporters=model_reporters,
                                agent_reporters=agent_reporters)  
        # Create agents
        for i in range(self.num_agents):
            x = 0
            y = 0
            pos = np.array((x, y))
            oda = odAgent(i, self)
            oda.pos = pos
            oda.breed = 'base'
            self.space.place_agent(oda, oda.pos)
            self.schedule.add(oda)
            
    def sigmoid(self, x):
        return 1 / (1 + 100*np.exp(-x))
       
    def step(self):
        if self.promotion_signal != 'pulse':
            for j in range(self.added_new_users[self.iteration - self.base_max_iters]):
                x = 0
                y = 0
                pos = np.array((x, y))
                ncA = None
                if self.new_agen_breed == 'household':
                    ncA = ncHouseholdInsured (j + self.base_max_iters, self)
                elif self.new_agen_breed == 'student':
                    ncA = ncStudentInsured (j + self.base_max_iters, self)
                elif self.new_agen_breed == 'level1':
                    ncA = ncLevel1Insurance (j + self.base_max_iters, self)
                elif self.new_agen_breed == 'level4':
                    ncA = ncLevel1Insurance (j + self.base_max_iters, self)
                ncA.pos = pos
                ncA.breed = self.new_agen_breed
                self.space.place_agent(ncA, ncA.pos)
                self.schedule.add(ncA)            
        self.dc.collect(self) 
        self.schedule.step()
        if self.communication_regime == "DW" and self.original:
            for agent in range (len(self.schedule.agents)):
                other = agent.random.choice(self.schedule.agents)
                other.update_opinion()
        else:
            for agent in self.schedule.agents:
                agent.update_opinion()
        for agent in self.schedule.agents:             
            loc = len(agent.opinion_list) - 1
            agent.opinion_list[loc] = agent.opinion
        for agent in self.schedule.agents: 
            if len (agent.opinion_list) == self.space.x_max + 1:
                agent.opinion_list.pop(0)
        for agent in self.schedule.agents:                     
            agent.entry_exit()  
        self.iteration += 1           
        if self.iteration >= self.max_iters:
            self.running = False  
            self.promotion_signal = 'pulse'
            
    def promote_users (self, promoted_users_type, new_customers_rate, added_iters, signal_function):
        num_new_customers = int(self.num_agents * new_customers_rate)
        self.max_iters += added_iters
        self.running = True
        self.new_agen_breed = promoted_users_type
        if signal_function == 'pulse':
            for j in range(self.num_agents , num_new_customers + self.num_agents):
                x = 0
                y = 0
                pos = np.array((x, y))
                ncA = None
                if promoted_users_type == 'household':
                    ncA = ncHouseholdInsured (j + self.base_max_iters, self)
                elif promoted_users_type == 'student':
                    ncA = ncStudentInsured (j + self.base_max_iters, self)
                elif promoted_users_type == 'level1':
                    ncA = ncLevel1Insured (j + self.base_max_iters, self)
                elif promoted_users_type == 'level4':
                    ncA = ncLevel4Insured (j + self.base_max_iters, self)
                ncA.pos = pos
                ncA.breed = promoted_users_type
                self.space.place_agent(ncA, ncA.pos)
                self.schedule.add(ncA)
        else:
            x=range(self.base_max_iters, self.max_iters)
            dfx = pd.DataFrame(x, columns = ['X'])    
            scaler = MinMaxScaler(feature_range=(0, 10))
            Xs = scaler.fit_transform(dfx)  
            Xs = Xs.ravel()   
            new_user = []
            previous_user = 0
            for x in Xs:
                y = self.sigmoid(x) 
                user = int(round(40 * y))
                if user > previous_user:
                    new_user.append(user - previous_user)
                else:
                    new_user.append(0)
                previous_user = user
            self.added_new_users = new_user
            self.promotion_signal = signal_function
                    
    @staticmethod        
    def rpt_opinion_mean (model):
        opinions = []
        for agent in model.schedule.agents:
            opinions.append(agent.opinion)
        opinion_mean = statistics.mean(opinions)  
        return opinion_mean

    @staticmethod        
    def rpt_opinion_median (model):
        opinions = []
        for agent in model.schedule.agents:
            opinions.append(agent.opinion)
        opinion_median = statistics.median(opinions)  
        return opinion_median
