#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import heapq
from numpy import linalg as LA


# 

# In[3]:


tolerance = 0.001

# Please use this setup for assignment 1
np.random.seed(5467)
# env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)     #define the environment.
# env.seed(48304)
# Create an environment with the TimeLimit wrapper
env = gym.make("FrozenLake-v1", is_slippery=False)
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

# This will raise an AttributeError: 'TimeLimit' object has no attribute 'P'
# print(env.P)

# Correct way to access the P attribute
print(env.unwrapped.P)

# env.render()


# In[4]:


# We are not supposed to peek into the model
# But for this assignment we will
P  = env.unwrapped.P
nA = 4
nS = 16
# Demo of how to peek and use P# You can run this and inspect the printout to understand the structure
# of entries in P
s=np.random.randint(nS)
a=np.random.randint(nA)

print(env.action_space)

# print("at state ",s," using action ",a)
# print("Transition to: ",P[s][a])
# print(type(P))
# print(P)


# In[ ]:


# This is the agent that performs Value Iteration

class VIagent:
    def __init__(self,env):
        self.P=env.unwrapped.P
        self.nA = 4
        self.nS = 16
        self.max_steps=env._max_episode_steps
        self.values = np.random.random(env.observation_space.n)
        self.policy = np.zeros(env.observation_space.n)  # Doesn't matter if initial policy is random or not
        self.tolerance = 0.001
        self.val1=[]
        self.val2=[]
        self.tick=0
        self.tock=0
        
    def value_iteration(self):
        no_of_backups=0
        while True:
            delta=0
            previous_values = np.copy(self.values)
            
            for state in range(nS):
                Q=[]
                
                for action in range(nA):
                    expected_future_rewards=[]
                    
                    for next_state_prob in P[state][action]:
                        prob, nextstate, reward, is_terminal = next_state_prob
                        
                        #Here we perform the core calculation of the equation given in the algorithm
                        expected_future_rewards.append((prob * (reward + self.values[nextstate])))
                
                    Q.append(sum(expected_future_rewards))  #Summation

                
                self.values[state] = max(Q)  #choosing the Max reward
                no_of_backups +=1
                delta = max(delta,abs(self.values[state]-previous_values[state]))
            
            if (delta < self.tolerance):
                break
            self.tick=time.time()    
            self.val1.append(self.mean_rewards_per_500())
            self.val2.append(no_of_backups)
            self.tock=time.time()    
                
        return self.values
    
    def policy_extract(self):
        
        for state in range(self.nS):
            
            q_stateaction=[]
            for action in range(self.nA):
                
                expected_future_rewards=[]
                for next_state_prob in P[state][action]:
                    
                    prob, nextstate, reward, is_terminal = next_state_prob
                    expected_future_rewards.append(prob*(reward + self.values[nextstate]))
                
                q_stateaction.append(sum(expected_future_rewards))
            
            self.policy[state]=np.argmax(q_stateaction)
            
        
        return self.policy
        
    def choose_action(self,observation):

        return self.policy[observation]

    def mean_rewards_per_500(self):
        self.policy_extract()
        total_reward = 0
        for episodes in range(500):
            observation, _ = env.reset()
            for _ in range(1000):

                action = self.choose_action(observation)
                observation, reward, done, _, info = env.step(action)
                total_reward += reward
                if done:
                    observation, _ = env.reset()
                    break
        return (total_reward/500)


viagent = VIagent(env)
viagent.value_iteration()
print("Value Iteration is performed \n\n")
print(viagent.values,"\n\n")
viagent.policy_extract()
print("Policy is now extracted\n\n")
print(viagent.policy)

#print(viagent.val1,viagent.val2)
print("\n The total time to perform 500 episodes in seconds is ",(viagent.tock-viagent.tick))


# Prioritized Sweeping Converges to
# the Optimal Value Function
# by **Lihong Li** and **Michael L. Littman**


# This is the agent that performs Prioritized Value Iteration

class PSVIagent:
    def __init__(self, env, no_of_dequeue):
        self.P  = env.unwrapped.P
        self.nA = 4
        self.nS = 16
        self.dequeues=no_of_dequeue
        self.max_steps=env._max_episode_steps
        self.values = np.random.random(env.observation_space.n)
        self.policy = np.zeros(env.observation_space.n)  # Doesn't matter if initial policy is random or not
        self.tolerance = 0.0009
        self.pr = np.ones(env.observation_space.n)
        self.val1=[]
        self.val2=[]
        self.tick=0
        self.tock=0
        self.dependent_states=[]

        for _ in range(nS):
            self.dependent_states.append([])
        

        for s in range(nS):
            for a in range(nA):
                for temp in P[s][a]:
                    prob, nextstate, reward, is_terminal = temp
                    self.dependent_states[nextstate].append(s)


    def value_iteration(self):
        no_of_backups=0
        flo = True
        count=0
        while flo==True:
            delta=0
            previous_values = np.copy(self.values)
            
            for _ in range(self.dequeues):
                Q=[]
                
                #print("========")
                state = np.argmax(self.pr)
                #print(state)
                for action in range(nA):
                    expected_future_rewards=[]
                    
                    for next_state_prob in P[state][action]:
                        prob, nextstate, reward, is_terminal = next_state_prob
                        
                        #Here we perform the core calculation of the equation given in the algorithm
                        expected_future_rewards.append((prob * (reward + self.values[nextstate])))
                        
                    Q.append(sum(expected_future_rewards))  #Summation

                
                self.values[state] = max(Q)  #choosing the Max reward
                no_of_backups +=1
                delta = max(delta,abs(self.values[state]-previous_values[state]))
                # delta=np.amax(np.absolute(self.values-previous_values))
                # # delta = np.linalg.norm(self.values-previous_values,1)
                self.pr[state]=abs(self.values[state]-previous_values[state])

                for e in self.dependent_states[state]:
                    self.pr[e]=abs(self.values[state]-previous_values[state])
            #print(delta)               
            count+=1
            # if (delta < self.tolerance):
            #     break
            self.tick=time.time()    
            self.val1.append(self.mean_rewards_per_500())
            self.val2.append(no_of_backups)
            self.tock=time.time()  

            if(count >= 1500):
                break  
                
        return self.values
    
    def policy_extract(self):
        
        for state in range(self.nS):
            
            q_stateaction=[]
            for action in range(self.nA):
                
                expected_future_rewards=[]
                for next_state_prob in P[state][action]:
                    
                    prob, nextstate, reward, is_terminal = next_state_prob
                    expected_future_rewards.append(prob*(reward + self.values[nextstate]))
                
                q_stateaction.append(sum(expected_future_rewards))
            
            self.policy[state]=np.argmax(q_stateaction)
            
        return self.policy
        
    def choose_action(self,observation):

        return self.policy[observation]
                        
    def mean_rewards_per_500(self):
        self.policy_extract()
        total_reward = 0
        for episodes in range(500):
            observation, _ = env.reset()
            for _ in range(1000):

                action = self.choose_action(observation)
                observation, reward, done, _, info = env.step(action)
                total_reward += reward
                if done:
                    observation, _ = env.reset()
                    break
        return (total_reward/500)

# In[ ]:


psagent10 = PSVIagent(env,10)
psagent10.value_iteration()

psagent18 = PSVIagent(env,25)
psagent18.value_iteration()


# In[15]:


psagent25 = PSVIagent(env,18)
psagent25.value_iteration()


# In[23]:



plt.plot(psagent10.val1,label="sweep t=10")
plt.plot(psagent25.val1,label="sweep t=18")
plt.plot(psagent18.val1,label="sweep t=25")
plt.plot(viagent.val1,label="Value Iteration")


plt.xlabel("Iterations")
plt.ylabel("Average reward")
plt.legend()


plt.show()


# In[22]:


plt.plot(psagent10.val2[::15],psagent10.val1[::15],label="sweep t=10")
plt.plot(psagent25.val2[::15],psagent25.val1[::15],label="sweep t=18")
plt.plot(psagent18.val2[::15],psagent18.val1[::15],label="sweep t=25")

plt.plot(viagent.val2[::15],viagent.val1[::15],label="Value Iteration")


plt.xlabel("No of Bellman backups")
plt.ylabel("Average reward")
plt.legend()


plt.show()


# In[ ]:




