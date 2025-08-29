
"""## Configuration for Colab"""

import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !apt install python-opengl
    !apt install ffmpeg
    !apt install xvfb
    !pip install PyVirtualDisplay==3.0
    !pip install gymnasium==1.2.0
    from pyvirtualdisplay import Display

    # Start virtual display
    dis = Display(visible=0, size=(400, 400))
    dis.start()

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):

    """ TODO: implement the n-step sarsa algorithm """
    
    total_episode_length = 0
    for ep in tqdm(range(num_ep)):      
        state, _  = env.reset()
        obs, _    = env.reset()
        n_states  = env.observation_space.n
        n_actions = env.action_space.n
        Q = np.random.rand(n_states,n_actions)   # Initializing random Q values for each (s,a)
        Q[-1,:] = 0                              # Setting final state values to 0
        policy = np.argmax(Q,axis=1)             # Greedy policy based on the random Q(s,a)     

        done = False
        t = 0
        tau = 0
        G=0
        T = np.inf

        states = [0]
        actions = [policy[0]]
        rewards = [0]

        while not done:
            if t < T:
                A = actions[t]   # take action A_t
                S , R , done , _, _ =  env.step(A)
                states.append(S)
                actions.append(A)
                rewards.append(R)

                if done:
                    T = t+1
                else:
                    actions.append(policy[S])  # action corresponding to current state S

            tau = t - n + 1
            if tau >= 0:
                for i in range(tau+1,min((tau+n,T))):
                    R_i = rewards[i]
                    G += ((gamma ** (i-tau-1)) * R_i)

                if (tau+n) < T:
                    G = G + (gamma**n) * Q[states[tau+n], actions[tau+n]]

                Q[states[tau],actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
                e = np.random.rand()
                if e > epsilon:
                    policy = np.argmax(Q,axis=1) # Update Policy (eps-Greedy)
                else:
                    policy = np.random.randint(n_actions, size=n_states)

            t += 1

      
        total_episode_length += t
        
    avg_ep_length = (total_episode_length/num_ep) 
    print("N =",n," ||  α = {:.2f}".format(alpha)," || Avg. Episode_length =",avg_ep_length)
    return avg_ep_length


# In[ ]:

env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)     #define the environment.

# TODO: run multiple times, evaluate the performance for different n and alpha
alphas = np.linspace(0.1,1,10)
lst = []
for n in range(10):
    for alpha in alphas:
        avg_ep_length = nstep_sarsa(env, n=n, alpha=alpha)
        lst.append((n,alpha,avg_ep_length))


# In[ ]:

L = []
lst1 = [(alpha,ep) for _,alpha,ep in lst]
for i in range(10):
    L.append(lst1[i*10:(i+1)*10])


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,8))
n=1
for l in L:
    x = []
    y = []
    for alpha,ep in l:
        x.append(alpha)
        y.append(ep)

    plt.plot(x,y,'.-',label="n={}".format(n))
    n +=1

plt.title("Performance evaluation of n-step Sarsa")
plt.xlabel("alpha")
plt.ylabel("Average episode length")
plt.legend()
plt.show()
  
# In[ ]:

n_states  = env.observation_space.n
n_actions = env.action_space.n
Q = np.random.rand(n_states,n_actions)


# In[ ]:


Q[-1,:]=0


# In[12]:


print(Q)






