---
layout: default
title:  Status
---

## Project Summary

Our project simulates a “Dropper” map, which is a 5 block wide and 250 block high tunnel made of wool. Inside the tunnel, there are some obsidian obstacles at different locations. At the bottom of the tunnel, there is a pool of water. 

The agent will spawn at the top of the tunnel and begin falling from it. The goal of the project is to let the agent reach the bottom of the tunnel and land in the pool of water safely without taking any damage. In this process, the agent will strafe in the air by moving in different directions to avoid all obsidian obstacles inside the tunnel in order to avoid dying from fall damage and make its way to the bottom. 

## Approach
We are using primarily Reinforcement Learning for our approach for the project.
	
Currently, our agent moves only in discrete actions.
* Move right
* Move left
* Move forward
* Move backwards
	
**Reward Function**
$$
R(s)=\left\{
	\begin{aligned}
	100 &\ (\text{Agent reaches water safely})\\
	1 &\ (\text{Time Agent stays alive (in milliseconds)})\\
	-25 &\ (\text{Damage from obstacle, without dying})\\
	\end{aligned}
	\right.
$$
	
## Evaluation
	Insert graphs and stuff here
	
## Remaining Goals and Challenges
- **Randomness**
 
  Right now, the obsidian obstacles inside the tunnel is hardcoded. The locations of the obstacles stay the same for each episode. In the future, we may generate random location of the obstacles for each episode.

- **Reward and Penalty**

  We still need to find out the way of rewarding the agent. For now, we have three plans: +1 reward for staying alive in the tunnel, +100 reward for landing in the pool and -100 reward  for dying, -1 reward for touching obsidian obstacles. For the future, we need to find out the most reasonable reward for the agents’s performance.

- **Map Complexity**

  Moreover, we are planning to make our map more complex. currently, we only have a vertical tunnel. We may add a horizontal tunnel for the final project, or add more obstacles to the map.
  
- ** Other Goals**
  
  Currently, we are looking to increase our efficiency of the agent to consistently receive the highest reward possible. 
  Additionally, we may try to branch out of discrete actions and try to have our agent move continuously, which simulates how this Minecraft minigame is played by actual players. In this case, the agent would move continuously at varying velocities to avoid obstacles on the map. However, this may be challenging due to the time limit and time window the agent has to send and receive commands. The average time it takes to run one complete episode for the agent is about 7 seconds, with the agent pausing 0.3 seconds before sending another command. We would have to try different reinforcement learning algorithms besides ones that only deal with discrete actions.

## Resources Used
Python Malmo libraries and documentation: 
https://github.com/microsoft/malmo
http://microsoft.github.io/malmo/0.30.0/Documentation/

PyTorch library

CS175 Assignment 2 DQN algorithm
