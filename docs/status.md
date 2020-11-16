---
layout: default
title:  Status
---

## Project Summary

Our project simulates a “Dropper” map, which is a 5 block wide and 250 block high tunnel made of wool. Inside the tunnel, there are some obsidian obstacles at different locations. At the bottom of the tunnel, there is a pool of water. 

The agent will spawn at the top of the tunnel and begin falling from it. The goal of the project is to let the agent reach the bottom of the turnnel and land in the pool of water safely. In this process, the agent will strafe in the air by moving in different directions to avoid all obsidian obstacles inside the tunnel in order to avoid dying from fall damage and make its way to the bottom. 

## Approach

## Evaluation

## Remaining Goals and Challenges
Right now, the obsidian obstacles inside the tunnel is hardcoded. The locations of the obstacles stay the same for each episode. In the future, we may generate random location of the obstacles for each episode.

We still need to find out the way of rewarding the agent. For now, we have three plans: +1 reward for getting one level lower in the tunnel, +100 reward for landing in the pool and -100 reward  for dying, -1 reward for touching obsidian obstacles. For the future, we need to find out the most reasonable reward for the agents’s performance.

Moreover, we are planning to make our map more complex. currently, we only have a verticle tunnel. We may add a horizontal tunnel for the final project.

## Resources Used
Python Malmo libraries and documentation: 
https://github.com/microsoft/malmo
http://microsoft.github.io/malmo/0.30.0/Documentation/

PyTorch library

CS175 Assignment 2
