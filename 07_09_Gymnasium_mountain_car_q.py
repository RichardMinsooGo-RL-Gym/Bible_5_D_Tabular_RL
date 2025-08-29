import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # hyperparameters
    learning_rate_a = 0.1           # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.
    
    epsilon = 1                     # 1 = 100% random actions
    epsilon_decay_rate = 0.001      # epsilon decay rate. 1/0.001 = 1,000
    rng = np.random.default_rng()   # random number generator
    pos_divisions = 20           # used to convert continuous state space to discrete space
    vel_divisions = 20           # used to convert continuous state space to discrete space
    
    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()


    rewards_per_episode = []     # list to store rewards for each episode

    i = 0                        # episode counter

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal
        
        rewards=0
		
        ep_count = 0

        # Episode
        while(not terminated and ep_count < 500):
            ep_count+= 1
            
            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )

            # Update state
            state   = new_state
            state_p = new_state_p
            state_v = new_state_v

            # Collect rewards
            rewards+=reward

            if not is_training and rewards%100==0:
                print(f'Episode: {i+1}  Rewards: {rewards}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])

        if is_training and i!=0 and (i+1)%100==0:
            print(f'Episode: {i+1} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')


        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001
        
        # Increment episode counter
        i+=1

    env.close()

    # Save Q table to file
    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.savefig(f'mountain_car.png')

    if is_training:
        f = open('mountain_car.pkl','wb')
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    # run(15000)

    run(episodes = 3000, is_training=True, render=True)
    