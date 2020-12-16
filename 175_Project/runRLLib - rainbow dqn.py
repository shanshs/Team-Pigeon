# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import tensorflow
import gym, ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo, dqn, impala

OBSTACLE_DENSITY = 0.20
class DropperSimulator(gym.Env):

    def __init__(self, env_config):  
        # Static Parameters
        self.size = 50
        self.reward_density = .1
        self.penalty_density = .02
        self.obs_size = 3
        self.depth = 30
        self.max_episode_steps = 30
        self.log_frequency = 10
        self.num_episode = 0
        self.xz_coordinate = 2.5,2.5
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'strafe -1',  # Moves left
            2: 'strafe 1',  # Moves right
            3: 'move -1',  # Moves back
            4: 'move 0'  # Moves 0
        }
        
        # Rllib Parameters
        self.action_space = Discrete(len(self.action_dict))
        self.observation_space = Box(0, 1, shape=(np.prod([self.depth, self.obs_size, self.obs_size]), ), dtype=np.int32)

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse( sys.argv )
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # DiamondCollector Parameters
        self.obs = None
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo
        world_state = self.init_malmo()

        # Reset Variables
        self.returns.append(self.episode_return)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)
        self.episode_return = 0
        self.episode_step = 0
        
        # Log
        if len(self.returns) > self.log_frequency and \
            len(self.returns) % self.log_frequency == 0:
            self.log_returns()

        # Get Observation
        self.obs = self.get_observation(world_state)
        
        return self.obs.flatten()

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # Get Action
        
        stateprint = self.agent_host.peekWorldState()
        stateprint = stateprint.is_mission_running
        if(stateprint == True):
            print("true")
            command = self.action_dict[action]
               
            self.agent_host.sendCommand(command)
            print(command)
            self.episode_step+=1
            time.sleep(.15)            
        else:
            print("false")
            done = True
            self.num_episode += 1
            #self.agent_host.sendCommand(" Reward : " + str(self.episode_return))
            print("DONE- Episode Number: " + str(self.num_episode) + " Reward : " + str(self.episode_return))
            time.sleep(4) 
            # Get Observation
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            self.obs = self.get_observation(world_state) 
    
            # Get Reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            self.episode_return += reward
    
            return self.obs.flatten(), reward, done, dict()            
        
        
        # Get Done
        
        done = False
        
        if self.episode_step >= self.max_episode_steps or \
                (self.obs[0, int(self.obs_size/2)-1, int(self.obs_size/2)] == 1 and \
                self.obs[1, int(self.obs_size/2)-1, int(self.obs_size/2)] == 0 and \
                command == 'move 1'):
            done = True
            self.num_episode += 1
            print("DONE- Episode Number: " + str(self.num_episode) + " Reward : " + str(self.episode_return))
            time.sleep(4)  
            
        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs = self.get_observation(world_state) 
        
        # Get Reward
        reward = 0
        for r in world_state.rewards:
            reward += r.getValue()
        self.episode_return += reward

        return self.obs.flatten(), reward, done, dict()

    def genString(self, x,y,z, blocktype):
        """
        Returns string to Draw block of a blocktype at given coordinates
        """
        return '<DrawBlock x="' + str(x) + '" y="' + str(y) + '" z="' + str(z) + '" type="' + blocktype + '"/>'
    
    def genMap(self):
        """
        Generates string of map for the XML
    
        x x x x
        x     x  
        x     x
        x x x x
    
        250 blocks high
        """
        mapStr = ""
    
    
        blocktype = 'wool'
        for elemX in range(5):
            for elemZ in range(5):
                for elemY in range(250):
                    if not (elemX == 1 and elemZ == 1 or elemX == 1 and elemZ ==2 or elemX == 1 and elemZ ==3 or elemX == 2 and elemZ ==1 or elemX == 2 and elemZ==2 or elemX == 2 and elemZ==3 or elemX == 3 and elemZ==1 or elemX == 3 and elemZ==2 or elemX == 3 and elemZ==3): 
                        # Removes 3x3 blocks^^^^ This is really ugly lol^^
                        a = self.genString(elemX, elemY, elemZ, blocktype)
                        #print(a)
                        mapStr += a
        # x o x
        # o x o
        # x o x   water = o, obsidian = x, for the first floor
        
        # Edit: it's just water on the first floor.
        blk = 'water'
        
        i = 2
        w = self.genString(1,i,2, blk)
        mapStr += w
        w = self.genString(1,i,3, blk) #
        mapStr += w    
        w = self.genString(2,i,1, blk)
        mapStr += w    
        w = self.genString(2,i,3, blk)
        mapStr += w  
        w = self.genString(3,i,1, blk) #
        mapStr += w     
        w = self.genString(3,i,2, blk)
        mapStr += w         
        w = self.genString(1,i,1, blk) #
        mapStr += w
        w = self.genString(2,i,2, blk) #
        mapStr += w    
        w = self.genString(3,i,3, blk) #
        mapStr += w        
        
        
        i = 1
        w = self.genString(1,i,2, blk)
        mapStr += w
        w = self.genString(1,i,3, blk) #
        mapStr += w    
        w = self.genString(2,i,1, blk)
        mapStr += w    
        w = self.genString(2,i,3, blk)
        mapStr += w  
        w = self.genString(3,i,1, blk) #
        mapStr += w     
        w = self.genString(3,i,2, blk)
        mapStr += w         
        w = self.genString(1,i,1, blk) #
        mapStr += w
        w = self.genString(2,i,2, blk) #
        mapStr += w    
        w = self.genString(3,i,3, blk) #
        mapStr += w
        
        i = 0
        w = self.genString(1,i,2, 'wool')
        mapStr += w
        w = self.genString(1,i,3, 'wool') #
        mapStr += w    
        w = self.genString(2,i,1, 'wool')
        mapStr += w    
        w = self.genString(2,i,3, 'wool')
        mapStr += w  
        w = self.genString(3,i,1, 'wool') #
        mapStr += w     
        w = self.genString(3,i,2, 'wool')
        mapStr += w         
        w = self.genString(1,i,1, 'wool') #
        mapStr += w
        w = self.genString(2,i,2, 'wool') #
        mapStr += w    
        w = self.genString(3,i,3, 'wool') #
        mapStr += w        
        '''
        # Some random obsidian block obstacles
        w = self.genString(2,100,2, 'obsidian')
        mapStr += w    
    
        w = self.genString(2,100,1, 'obsidian')
        mapStr += w    
        
        
        w = self.genString(2,150,3, 'obsidian')
        mapStr += w     
        
        w = self.genString(1,150,3, 'obsidian')
        mapStr += w    
        w = self.genString(3,200,3, 'obsidian')
        mapStr += w       
        w = self.genString(2,60,2, 'obsidian')
        mapStr += w      
        w = self.genString(2,80,2, 'obsidian')
        mapStr += w         
        w = self.genString(1,80,1, 'obsidian')
        mapStr += w      
        w = self.genString(3,80,3, 'obsidian')
        mapStr += w    
        w = self.genString(2,80,3, 'obsidian')
        '''
        
        for X in range(1,4):
            for Z in range(1,4):
                for Y in range(5, 230, 20):
                    p = np.random.random() 
                    if p <= OBSTACLE_DENSITY:
                        w = self.genString(X,Y,Z, 'obsidian')
                        print(w)
                        mapStr += w        
        mapStr += w    
        
        return mapStr
    
    def gen_air(self):
        s = ''
        for X in range(1,4):
            for Z in range(1,4):        
                for Y in range(2, 240):
                    s += "<DrawBlock x='{}'  y='{}' z='{}' type='air' />".format(X,Y,Z)
        return s
    def gen_marker_reward(self):
        s = ''
        
        s += "<Marker x='1.5' y='1' z='1.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='1.5' y='1' z='2.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='1.5' y='1' z='3.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='2.5' y='1' z='1.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='2.5' y='1' z='2.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='2.5' y='1' z='3.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='3.5' y='1' z='1.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='3.5' y='1' z='2.5' reward='100' tolerance='0.5'/> "
        s += "<Marker x='3.5' y='1' z='3.5' reward='100' tolerance='0.5'/> "
        
        return s    
    def GetMissionXML(self):
        #------------------------------------
        #
        #   TODO: Spawn diamonds
        #   TODO: Spawn lava
        #   TODO: Add diamond reward
        #   TODO: Add lava negative reward
        #
        #-------------------------------------
        
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    
                    <About>
                        <Summary>Drop Simulator</Summary>
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
                                                  self.gen_air() + self.genMap() +\
                                '''
                            </DrawingDecorator>
                            <ServerQuitWhenAnyAgentFinishes/>
                        </ServerHandlers>
                    </ServerSection>
    
                    <AgentSection mode="Survival">
                        <Name>Dropper Simulator</Name>
                        <AgentStart>
                            <Placement x="2.5" y="250" z="2.5" pitch="90" yaw="0"/>
                            
                        </AgentStart>
                        <AgentHandlers>
                            <DiscreteMovementCommands/>
                            <ObservationFromFullStats/>
                            
                            
                            <RewardForTimeTaken initialReward="0" delta="1" density="PER_TICK"/>
                            
                            <RewardForTouchingBlockType>
                            <Block type = "water" reward = '100' behaviour = 'onceOnly'/>
                            <Block type = "obsidian" reward = '0' behaviour = 'onceOnly'/>
                            </RewardForTouchingBlockType>
                            
                            
                            <RewardForMissionEnd rewardForDeath="0">
                                <Reward reward="0" description="Mission End"/>
                            </RewardForMissionEnd>
                        <ObservationFromGrid>
                                <Grid name="floorAll">
                                    <min x="-''' + str(int(self.obs_size/2)) + '''" y="-''' + str(self.depth - 1) + '''" z="-''' + str(int(self.obs_size/2)) + '''"/>
                                    <max x="''' + str(int(self.obs_size/2)) + '''" y="0" z="''' + str(int(self.obs_size/2)) + '''"/>
                                </Grid>
                                </ObservationFromGrid>
                            
                            <AgentQuitFromTouchingBlockType>
                            <Block type = "water"/>
                                <Block type = "obsidian"/>
                            </AgentQuitFromTouchingBlockType>
                            
							
                        </AgentHandlers>
                    </AgentSection>
                </Mission>'''
    
    
    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.GetMissionXML(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(1200, 700)
        my_mission.setViewpoint(0)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        for retry in range(max_retries):
            try:
                self.agent_host.startMission( my_mission, my_clients, my_mission_record, 0, 'DropperSimulator' )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.15)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array>
        """
        obs = np.zeros((self.depth, self.obs_size, self.obs_size))

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)
                
                # Get observation
                grid = observations['floorAll']
                grid_binary = [1 if x == 'obsidian' else 0 for x in grid]
                obs = np.reshape(grid_binary, (self.depth, self.obs_size, self.obs_size))

                # Rotate observation with orientation of agent
                yaw = observations['Yaw']
                if yaw == 270:
                    obs = np.rot90(obs, k=1, axes=(1, 2))
                elif yaw == 0:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif yaw == 90:
                    obs = np.rot90(obs, k=3, axes=(1, 2))
                
                break

        return obs

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns, box, mode='same')
        plt.clf()
        plt.plot(self.steps, returns_smooth)
        plt.title('Dropper Simulator')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps, self.returns):
                f.write("{}\t{}\n".format(step, value)) 

      
        
        
        
if __name__ == '__main__':
    ray.init()
    trainer = dqn.DQNTrainer(env=DropperSimulator, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'tf',       
        
        "n_step": 5, 
        "noisy": True, 
        "num_atoms": 50, 
        "v_min": 10.0, "v_max": 180.0
    })

    while True:
        trainer.train()

