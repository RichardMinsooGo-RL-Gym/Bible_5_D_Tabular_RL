
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
import time, pickle, os
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])

    def learn(self, state, next_state, reward, action):
        # predict = Q[state, action]
        # Q[state, action] = Q[state, action] + lr_rate * (target - predict)
        target = reward + gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] = (1 - lr_rate) * self.Q[state, action] + lr_rate * target


def train():
    total_rewards = []
    for episode in range(total_episodes):
        state, _ = agent.env.reset()
        t = 0
        ep_rewards = 0
        
        while t < max_steps:
            # env.render()

            action = agent.choose_action(state)  
            next_state, reward, done, info, _ = agent.env.step(action)  

            agent.learn(state, next_state, reward, action)
            state = next_state

            ep_rewards+= reward
            t += 1
            if done:
                break
            
        total_rewards.append(ep_rewards)
        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
        # time.sleep(0.1)

        # if done:
        #     break
    return total_rewards

# test
video_folder="videos/per"

def test(video_folder):
    total_rewards = []
    for episode in range(100):
        ep_rewards = 0
        
        # for recording a video
        naive_env = agent.env
        agent.env = gym.wrappers.RecordVideo(agent.env, video_folder=video_folder,  fps=2)

        state, _ = agent.env.reset()
        
        for t in range(max_steps):
            #agent.env.render()
            #time.sleep(0.5)
            act = np.argmax(agent.Q[state,:])
            next_state, reward, done, info, _ = agent.env.step(act)
            if done:
                ep_rewards += reward
                break
            else:
                state = next_state
        total_rewards.append(ep_rewards)
    return total_rewards 
    
def train_details(total_rewards):
    print("Q table:\n", agent.Q)

    # Perfect actions: [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0]
    print("Total Rewards in training: {0} in {1} episodes".format(sum(total_rewards), total_episodes))

    # Policy
    def act_names(x):
        dt = {0:'left', 1:'down', 2:'right', 3:'up'}
        return dt[x]
    act_np = np.vectorize(act_names)
    act = np.argmax(agent.Q, axis=1)
    act = act_np(act)
    act = act.reshape(4,4)
    print("Policy:\n", act)

def test_details(total_rewards):
    print("\n\nTotal Rewards in testing: {0} in {1} episodes".format(sum(total_rewards), total_episodes))

    # Policy
    def act_names(x):
        dt = {0:'left', 1:'down', 2:'right', 3:'up'}
        return dt[x]
    act_np = np.vectorize(act_names)
    act = np.argmax(agent.Q, axis=1)
    act = act_np(act)
    act = act.reshape(4,4)
    print("Policy:\n", act)

# Setup
env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)     #define the environment.

epsilon = 0.9
lr_rate = 0.1
gamma = 0.95

total_episodes = 10000
max_steps = 100

agent = Agent(env)

print("Total States:", env.observation_space.n)
print("Total Actions:", env.action_space.n)

print("####### Training ########")
total_rewards = train()
train_details(total_rewards)

print("####### Testing ########")
total_test_rewards = test(video_folder)
test_details(total_test_rewards)

# with open("frozenLake_qTable.pkl", 'wb') as f:
#     pickle.dump(Q, f)



"""## Render"""

import base64
import glob
import io
import os

from IPython.display import HTML, display


def ipython_show_video(path: str) -> None:
    """Show a video at `path` within IPython Notebook."""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    display(
        HTML(
            data="""
        <video width="320" height="240" alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
        </video>
        """.format(
                encoded.decode("ascii")
            )
        )
    )


def show_latest_video(video_folder: str) -> str:
    """Show the most recently recorded video from video folder."""
    list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    latest_file = max(list_of_files, key=os.path.getctime)
    ipython_show_video(latest_file)
    return latest_file


latest_file = show_latest_video(video_folder=video_folder)
print("Played:", latest_file)

