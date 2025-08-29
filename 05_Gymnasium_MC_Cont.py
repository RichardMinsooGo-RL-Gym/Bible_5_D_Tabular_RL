
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
# from utils import JupyterRender

get_ipython().run_line_magic('matplotlib', 'inline')

# In[ ]:


class MC_control:
    def __init__(self, env, gamma=0.9, eps=0.1, max_episode=100, render=False):
        self.state_dim = env.observation_space.n # self.nrow * self.ncol
        self.action_dim = env.action_space.n
        
        self.env = env
        
        # self.nrow = env.nrow
        # self.ncol = env.ncol 
        
        self.nrow = 4
        self.ncol = 4 
        
        self.gamma = gamma
        self.eps = eps
        self.max_episode = max_episode
        
        self.render = render
        
        self.pi = np.ones([self.state_dim, self.action_dim]) / self.action_dim # probability of each action
        self.q = np.zeros([self.state_dim, self.action_dim])
        
        self.seed = 777
        
    def action(self, s):
        #epsilon-soft policy
        assert all(list(map(lambda x: x >= (self.eps/self.action_dim), self.pi[s])))
        
        action = np.random.choice(list(range(self.action_dim)), p=self.pi[s])
        
        return action
    
    def run(self):
        self.success = 0

        #trajectory generation
        for episode in range(self.max_episode):
            observation, _ = self.env.reset()

            done = False
            
            episode_reward = 0
            local_step = 0
            
            trajectory = []
            returns = [[[] for _ in range(self.action_dim)] for _ in range(self.state_dim)]
            
            while not done:
                action = self.action(observation)
                next_observation, reward, done, _, _ = self.env.step(action)
                
                # if self.render:
                #     self.env.render(title=f"Episode {episode} / step {local_step}", q=self.q)
                
                # give penalty for staying in ground
                if reward == 0:
                    reward = -0.001
                    
                # give penalty for falling into the hole
                if done and next_observation != 15:
                    reward = -1

                if local_step == 100:
                    done = True #prevent infinite episode
                    reward = -1

                if observation == next_observation: # prevent meaningless actions
                    reward = -1
                
                trajectory.append({'s': observation, 'a': action, 'r': reward})
                
                observation = next_observation
                
                episode_reward += reward
                local_step += 1

            trajectory.reverse()

            G = 0
            
            traj_sa_list = list(map(lambda x: [x['s'], x['a']], trajectory))
            
            for i in range(len(trajectory)):
                G = self.gamma * G + trajectory[i]['r'] 
                cur_s = trajectory[i]['s']
                cur_a = trajectory[i]['a']
                
                if [cur_s, cur_a] not in traj_sa_list[i+1:]:
                    returns[cur_s][cur_a].append(G)
                    
                    self.q[cur_s][cur_a] = sum(returns[cur_s][cur_a])/len(returns[cur_s][cur_a])

                    optimal_action = np.argmax(self.q, axis=1)

                    for i in range(self.state_dim):
                        for j in range(self.action_dim):
                            if optimal_action[i] == j:
                                self.pi[i][j] = 1 - self.eps + self.eps/self.action_dim
                            else:
                                self.pi[i][j] = self.eps/self.action_dim    
                                
            #print("Episode: {} -> Step: {}, Episode_reward: {}".format(episode, local_step, episode_reward))
            
            if observation == 15:
                self.success += 1
                
        #print("Success rate: ", success / self.max_episode)

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder,  fps=2)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.action(state)
            next_observation, reward, done, _, _ = self.env.step(action)
            
            state = next_observation
            score += reward
            
        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env


# In[ ]:


env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)#define the environment.
# env = JupyterRender(env)


# In[ ]:


mc_config = {
    'env': env,
    'gamma': 0.9,
    'eps': 0.1,
    'render': True,
    'max_episode': 1000
}

mc = MC_control(**mc_config)
mc.run()


# In[ ]:


"""## Test"""

# test
video_folder="videos/per"
mc.test(video_folder=video_folder)

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

