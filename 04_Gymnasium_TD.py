
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


class TD_learning:
    def __init__(self, env, pi, alpha=0.1, gamma=0.9, max_episode=100, render=False):
        self.state_dim = env.observation_space.n # self.nrow * self.ncol
        self.action_dim = env.action_space.n
        
        self.env = env
        
        # self.nrow = env.nrow
        # self.ncol = env.ncol 
        
        self.nrow = 4
        self.ncol = 4 
        
        self.alpha = alpha
        self.gamma = gamma
        self.max_episode = max_episode
        
        self.render = render
        
        self.pi = pi
        self.v = np.zeros([self.state_dim])
        
        self.seed = 777
        
        #check policy validity
        assert len(self.pi) == self.state_dim
        
        for i in range(self.state_dim):
            assert self.pi[i] >= 0 and self.pi[i] < self.action_dim
        
    def run(self):

        #trajectory generation
        for episode in range(self.max_episode):
            observation, _ = self.env.reset()

            done = False
            
            local_step = 0
            
            while not done:
                action = self.pi[observation]
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
                
                self.v[observation] += self.alpha * (reward + self.gamma * self.v[next_observation] - self.v[observation])
                
                observation = next_observation
                local_step += 1

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
            action = self.pi[state]
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

policy = np.array([1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0], dtype=int) #optimal policy

# In[ ]:


td_config = {
    'env': env,
    'pi': policy,
    'gamma': 0.9,
    'alpha': 0.5,
    'render': True,
    'max_episode': 10
}

td = TD_learning(**td_config)
td.run()


# In[ ]:


"""## Test"""

# test
video_folder="videos/per"
td.test(video_folder=video_folder)

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

