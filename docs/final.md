---
layout: default 
title: Final Report 
---

## Video

## Project Summary
Our project attempts to have an agent complete a “Dropper” map, which is a map in which a player attempts to fall safely to the bottom while avoiding obstacles. Our agent will be completing a map in which there is a 5 block wide and 250 blocks high tunnel made of wool. Inside the tunnel, there are some randomly generated obsidian obstacles at different locations (By randomly generating the map, the agent sometimes meets a more complex situation: for example, it can go left to avoid hitting an obstacle, but that would cause a dead-end or narrow escape further down). At the bottom of the tunnel, there is a pool of water.

The agent will spawn at the top of the tunnel and begin falling from it. The goal of the mission is to let the agent reach the bottom of the tunnel and land in the pool of water safely without taking any damage. In this process, the agent will strafe in the air by moving in different directions to avoid all obsidian obstacles inside the tunnel to avoid dying from fall damage and make its way to the bottom.

<div style="text-align:center"><img src="summary3.png" width="500" height="330"/></div>

This problem isn’t trivial because the solution can be quite complex. There are a lot of ways to reward and punish the agent and finding the best way takes a lot of time. The falling speed of the agent raises another challenge for the project since we are not able to control the falling speed of the agent. If the sleep time is too long, the agent makes less move so it will dodge obstacles; if the sleep time is too short, the reward of landing in water doesn’t get registered. Also, if the gap between two obstacles is too short, the agent would not have enough time to make a move with unsuitable sleep time.

To find the most proper way to solve the problems above, our team would have to run the model overnight often time. 

## Approaches

## Evaluation

## References
CS175 Assignment 2's DQN algorithm

https://github.com/microsoft/malmo

http://microsoft.github.io/malmo/0.30.0/Documentation/

https://docs.ray.io/en/latest/rllib-algorithms.html#ppo

PyTorch library
