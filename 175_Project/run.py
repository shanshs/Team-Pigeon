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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from malmo import MalmoPython
except:
    import MalmoPython


# Hyperparameters
SIZE =50
REWARD_DENSITY = .1
PENALTY_DENSITY = .02
OBSTACLE_DENSITY = 0.03
OBS_SIZE = 3
MAX_EPISODE_STEPS = 100
MAX_GLOBAL_STEPS = 10000
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

    def __init__(self, obs_size, action_size, hidden_size=90):
        print(obs_size)
        super().__init__()
        self.net = nn.Sequential(nn.Linear(np.prod(obs_size), hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, action_size)) 

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
    fixed = True

    blocktype = 'wool'
    for elemX in range(5):
        for elemZ in range(5):
            for elemY in range(250):
                if not (elemX == 1 and elemZ == 1 or elemX == 1 and elemZ ==2 or elemX == 1 and elemZ ==3 or elemX == 2 and elemZ ==1 or elemX == 2 and elemZ==2 or elemX == 2 and elemZ==3 or elemX == 3 and elemZ==1 or elemX == 3 and elemZ==2 or elemX == 3 and elemZ==3): 
                    # Removes 3x3 blocks^^^^ This is really ugly lol^^
                    a = genString(elemX, elemY, elemZ, blocktype)
                    #print(a)
                    mapStr += a

    if fixed:
        # x o x
        # o x o
        # x o x   water = o, obsidian = x, for the first floor
        w = genString(1,2,2, 'water')
        mapStr += w
        w = genString(1,1,2, 'glass')
        mapStr += w    
        w = genString(2,2,1, 'water')
        mapStr += w    
        w = genString(2,1,1, 'glass')
        mapStr += w 
        w = genString(2,2,3, 'water')
        mapStr += w  
        w = genString(2,1,3, 'glass')
        mapStr += w     
        w = genString(3,2,2, 'water')
        mapStr += w         
        w = genString(3,1,2, 'glass')
        mapStr += w 
        w = genString(1,2,1, 'water')
        mapStr += w
        w = genString(1,1,1, 'glass')
        mapStr += w
        w = genString(1,2,3, 'water')
        mapStr += w    
        w = genString(1,1,3, 'glass')
        mapStr += w
        w = genString(2,2,2, 'water')
        mapStr += w    
        w = genString(2,1,2, 'glass')
        mapStr += w
        w = genString(3,2,3, 'water')
        mapStr += w       
        w = genString(3,1,3, 'glass')
        mapStr += w
        w = genString(3,2,1, 'water')
        mapStr += w 
        w = genString(3,1,1, 'glass')
        mapStr += w
        
        # Some random obsidian block obstacles
        w = genString(2,100,2, 'obsidian')
        mapStr += w    

        w = genString(2,100,1, 'obsidian')
        mapStr += w    
        
        
        w = genString(2,150,3, 'obsidian')
        mapStr += w     
        
        w = genString(1,150,3, 'obsidian')
        mapStr += w    
        w = genString(3,200,3, 'obsidian')
        mapStr += w       
        w = genString(2,60,2, 'obsidian')
        mapStr += w      
        w = genString(2,80,2, 'obsidian')
        mapStr += w         
        w = genString(1,80,1, 'obsidian')
        mapStr += w      
        w = genString(3,80,3, 'obsidian')
        mapStr += w    
        w = genString(2,80,3, 'obsidian')
        mapStr += w    

        mapStr += gen_air()

    if not fixed:
        w = genString(1,2,1, 'water')
        mapStr += w
        w = genString(2,2,1, 'water')
        mapStr += w
        
        w = genString(1,2,2, 'obsidian')
        mapStr += w
        w = genString(2,2,2, 'obsidian')
        mapStr += w    
        
        for X in range(1,4):
            for Z in range(1,4):
                w = genString(X,1,Z, 'wool')
                mapStr += w
        
        
        for X in range(1,4):
            for Z in range(1,4):
                for Y in range(5, 250, 3):
                    p = np.random.random() 
                    if p <= OBSTACLE_DENSITY:
                        w = genString(X,Y,Z, 'obsidian')
                        mapStr += w

    return mapStr

def gen_air():
    s = ''
    for y in range(250):
        s += "<DrawBlock x='0'  y='"+ str(y) +"' z='0' type='air' />"
    return s

def gen_marker_reward():
    s = ''
    for y in range(250):
        s += "<Marker x='1.5' y='" + str(y) +"' z='1.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='1.5' y='" + str(y) +"' z='2.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='1.5' y='" + str(y) +"' z='3.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='2.5' y='" + str(y) +"' z='1.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='2.5' y='" + str(y) +"' z='2.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='2.5' y='" + str(y) +"' z='3.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='3.5' y='" + str(y) +"' z='1.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='3.5' y='" + str(y) +"' z='2.5' reward='1' tolerance='0.5'/>"
        s += "<Marker x='3.5' y='" + str(y) +"' z='3.5' reward='1' tolerance='0.5'/>"

    return s

def GetMissionXML():

    #<RewardForMissionEnd rewardForDeath="-200">
    #                    <Reward reward="0" description="Mission End"/>
    #                    </RewardForMissionEnd>

    #<RewardForReachingPosition>''' + \
    #                    gen_marker_reward() + \
    #                    '''
    #                    </RewardForReachingPosition>
    
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
                        </RewardForTouchingBlockType>
                        <RewardForTimeTaken initialReward="0" delta="1" density="PER_TICK"/>
                        <ObservationFromGrid>
                            <Grid name="floorAll">
                                <min x="-1" y="-9" z="-1"/>
                                <max x="1" y="0" z="1"/>
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

    obs = np.zeros((10, OBS_SIZE, OBS_SIZE))

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
            #print(observations)
            grid = observations['floorAll']
            grid_binary = [1 if x == 'obsidian' or x == 'water' else 0 for x in grid]
            obs = np.reshape(grid_binary, (10, OBS_SIZE, OBS_SIZE))

            # Rotate observation with orientation of agent
            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))

            break
    #print(obs)
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


def log_returns(steps, returns):
    """
    Log the current returns as a graph and text file

    Args:
        steps (list): list of global steps after each episode
        returns (list): list of total return of each episode
    """
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Dropper')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 


def train(agent_host):
    """
    Main loop for the DQN learning algorithm

    Args:
        agent_host (MalmoPython.AgentHost)
    """
    # Init networks
    q_network = QNetwork((10, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
    target_network = QNetwork((10, OBS_SIZE, OBS_SIZE), len(ACTION_DICT))
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
    steps = []

    # Begin main loop
    loop = tqdm(total=MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

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
            # print(command)
            agent_host.sendCommand(command)
            #print('b')

            # If your agent isn't registering reward you may need to increase this
            time.sleep(0.3)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= MAX_EPISODE_STEPS or \
               (obs[0, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 1 and \
                     obs[1, int(OBS_SIZE/2)-1, int(OBS_SIZE/2)] == 0 and \
                    command == 'move 1'):
                done = True
                time.sleep(10)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state) 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
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
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.4f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, returns)
            print()


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