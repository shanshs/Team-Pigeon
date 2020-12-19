import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint
import pandas as pd 
from pandas.plotting import table

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from malmo import MalmoPython
except:
    import MalmoPython


# Hyperparameters
OBSTACLE_DENSITY = 0.2
OBS_SIZE = 3
DEPTH = 40
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 1000000
REPLAY_BUFFER_SIZE = 10000
EPSILON_DECAY = .999
MIN_EPSILON = .1
BATCH_SIZE = 128
GAMMA = .9
TARGET_UPDATE = 100
LEARNING_RATE = 1e-4
START_TRAINING = 500
LEARN_FREQUENCY = 1
ACTION_DICT = {
    0: 'move 1',  # Move one block forward
    1: 'strafe -1',  # Moves left
    2: 'strafe 1',  # Moves right
    3: 'move -1'  # Moves back
}


# Q-Value Network
class QNetwork(nn.Module):
    #------------------------------------
    #
    #   TODO: Modify network architecture
    #
    #-------------------------------------

    def __init__(self, obs_size, action_size, hidden_size=200):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, 100),
                                 nn.Softplus(),
                                 nn.Linear(100, 4))

    def forward(self, obs):
        """
        Estimate q-values given obs

        Args:
            obs (tensor): current obs, size (batch x obs_size)

        Returns:
            q-values (tensor): estimated q-values, size (batch x action_size)
        """
        batch_size = obs.shape[0]
        obs_flat = obs.view(batch_size, -1)
        return self.net(obs_flat)

def genString(x,y,z, blocktype):
    """
    Returns string to Draw block of a blocktype at given coordinates
    """
    return '<DrawBlock x="' + str(x) + '" y="' + str(y) + '" z="' + str(z) + '" type="' + blocktype + '"/>'

def genMap():
    """
    Generates string of map for the XML

    x x x x
    x     x  
    x     x
    x x x x

    250 blocks high
    """
    mapStr = ""
    fixed = False

    for elemX in range(5):
        for elemZ in range(5):
            for elemY in range(250):
                if not (elemX == 1 and elemZ == 1 or 
                        elemX == 1 and elemZ ==2 or 
                        elemX == 1 and elemZ ==3 or 
                        elemX == 2 and elemZ ==1 or 
                        elemX == 2 and elemZ==2 or 
                        elemX == 2 and elemZ==3 or 
                        elemX == 3 and elemZ==1 or 
                        elemX == 3 and elemZ==2 or 
                        elemX == 3 and elemZ==3): 
                    mapStr += genString(elemX, elemY, elemZ, 'wool')

    if fixed:
        # x o x
        # o x o
        # x o x   water = o, obsidian = x, for the first floor
        pos = [(1,2,2), (2,2,1), (2,2,3), (3,2,2), (1,2,1), (1,2,3), (2,2,2), (3,2,3), (3,2,1)]
        for p in pos:
            mapStr += genString(p[0], p[1], p[2], 'water')

        pos = [(1,1,2), (2,1,1),(2,1,3), (3,1,2), (1,1,1), (1,1,3), (2,1,2), (3,1,3), (3,1,1)]
        for p in pos:
            mapStr += genString(p[0], p[1], p[2], 'glass')
            
        # Some random obsidian block obstacles
        pos = [(2,100,2), (2,100,1), (2,150,3), (1,150,3), (3,200,3), (2,60,2), (2,80,2), (1,80,1), (3,80,3), (2,80,3)]
        for p in pos:
            mapStr += genString(p[0], p[1], p[2], 'obsidian')  

        """mapStr += gen_air()"""

    if not fixed: 
        
        for X in range(1,4):
            for Z in range(1,4):
                mapStr += genString(X,2,Z, 'water')
                mapStr += genString(X,3,Z, 'water')
                mapStr += genString(X,1,Z, 'wool')
        
        
        for X in range(1,4):
            for Z in range(1,4):
                for Y in range(5, 240, 35):
                    p = np.random.random() 
                    if p <= OBSTACLE_DENSITY:
                        w = genString(X,Y,Z, 'obsidian')
                        mapStr += w

    return mapStr

def GetMissionXML():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                <About>
                    <Summary>Dropper</Summary>
                </About>

                <ServerSection>
                    <ServerInitialConditions>
                        <Time>
                            <StartTime>12000</StartTime>
                            <AllowPassageOfTime>false</AllowPassageOfTime>
                        </Time>
                        <Weather>clear</Weather>
                    </ServerInitialConditions>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;1*minecraft:grass;1;"/>
                        <DrawingDecorator>''' + \
                              "<DrawCuboid x1='-6' x2='6' y1='-1' y2='250' z1='-6' z2='6' type='air'/>" +\
                                              genMap()+ \
                            '''
                        </DrawingDecorator>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                </ServerSection>

                <AgentSection mode="Survival">
                    <Name>CS175Dropper</Name>
                    <AgentStart>
                        <Placement x="2.5" y="250" z="2.5" pitch="90" yaw="0"/>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ObservationFromFullStats/>
                        <RewardForTouchingBlockType>
                            <Block type = "water" reward = '100' behaviour = 'onceOnly'/>
                            <Block type = "obsidian" reward = '-1' behaviour = 'onceOnly'/>
                        </RewardForTouchingBlockType>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                    <min x="-''' + str(int(OBS_SIZE/2)) + '''" y="-''' + str(DEPTH - 1) + '''" z="-''' + str(int(OBS_SIZE/2)) + '''"/>
                                    <max x="''' + str(int(OBS_SIZE/2)) + '''" y="0" z="''' + str(int(OBS_SIZE/2)) + '''"/>
                            </Grid>
                        </ObservationFromGrid>
                        <AgentQuitFromTouchingBlockType>
                        <Block type = "water"/>
                        </AgentQuitFromTouchingBlockType>
                    </AgentHandlers>
                </AgentSection>
            </Mission>'''


def get_action(obs, q_network, epsilon):
    with torch.no_grad():
        # Calculate Q-values fot each action
        obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
        action_values = q_network(obs_torch)

        action_prob = [epsilon/4.0, epsilon/4.0, epsilon/4.0, epsilon/4.0]
        actions = [0, 1, 2, 3]

        # Select action with highest Q-value
        action_idx = torch.argmax(action_values).item()

        action_prob[action_idx] += (1-epsilon)
        action_i = np.random.choice(actions, p=action_prob)
        
    return action_i


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(1200, 700)
    my_mission.setViewpoint(0)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "Dropper" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def get_observation(world_state):

    obs = np.zeros((DEPTH, OBS_SIZE, OBS_SIZE))

    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            # First we get the json from the observation API
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            # Get observation
            grid = observations['floorAll']
            grid_binary = [1 if x == 'obsidian' else 0 for x in grid]
            obs = np.reshape(grid_binary, (DEPTH, OBS_SIZE, OBS_SIZE))
            break
            
    return obs


def prepare_batch(replay_buffer):
    """
    Randomly sample batch from replay buffer and prepare tensors

    Args:
        replay_buffer (list): obs, action, next_obs, reward, done tuples

    Returns:
        obs (tensor): float tensor of size (BATCH_SIZE x obs_size
        action (tensor): long tensor of size (BATCH_SIZE)
        next_obs (tensor): float tensor of size (BATCH_SIZE x obs_size)
        reward (tensor): float tensor of size (BATCH_SIZE)
        done (tensor): float tensor of size (BATCH_SIZE)
    """
    batch_data = random.sample(replay_buffer, BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)

    return obs, action, next_obs, reward, done


def learn(batch, optim, q_network, target_network):
    """
    Update Q-Network according to DQN Loss function

    Args:
        batch (tuple): tuple of obs, action, next_obs, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
    """
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()


def log_returns(steps, returns, fileName, xlabel):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    plt.clf()
    plt.plot(steps, returns)
    
    plt.title('Dropper')
    plt.ylabel('Return')
    plt.xlabel(xlabel)
    plt.savefig(fileName)

def AvgReturn(suc_return):
    df = pd.DataFrame(suc_return, columns = ['num_episode', 'AvgSuccess']) 
    
    if suc_return[-1][0]>= 1999:
        print(df)
        
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table(ax, df, loc='center')
    plt.savefig('AvgReturnff.png')
    
    

def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((DEPTH, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((DEPTH, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    suc_return = []
    steps = []
    stepse = []
    success = 0
    avg_re = []
    avg_ree = []
    eps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False
        if num_episode%500 == 0:
            success = 0

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            action_idx = get_action(obs, q_network, epsilon)
            command = ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(0.2)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS or \
               (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
                     obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
                    command == 'move 1'):
                done = True
                time.sleep(15)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state) 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
                if reward >=90:
                    success += 1
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
            global_step += 1
            if global_step > START_TRAINING and global_step % LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY

                if global_step % TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        steps.append(global_step)
        avg_return = sum(returns)/ (len(returns))
        avg_re.append(avg_return)
        if len(returns) >= 500:
            stepse.append(global_step)
            avg_returne = sum(returns[len(returns)-495:-1])/ (len(returns[len(returns)-495:-1]))
            avg_ree.append(avg_returne)
            eps.append(num_episode)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.4f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 20 == 0:
            log_returns(steps, avg_re, "AllAvg.png", "Steps")
            print()
            
        if num_episode > 0 and num_episode % 500 == 0:
            log_returns(stepse, avg_ree, "500AvgStep.png", "Steps")
            log_returns(eps, avg_ree, "500AvgEps.png", "Episode")
            print()
         
        if num_episode > 0 and num_episode % 500 == 0:
            suc_return.append([num_episode,success/500])
            AvgReturn(suc_return)

    
if __name__ == '__main__':
    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    train(agent_host)
