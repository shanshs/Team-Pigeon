---
layout: default
title:  Team
---

## Summary of the Project
Our agent will be attempting to complete a “Dropper” map. The agent will spawn at a high location and begin falling. Its goal will be to reach a location at the bottom in which it will land in a pool of water safely. The agent will need to avoid various obstacles in the air in order to avoid dying from fall damage so it can make its way to the bottom. It will do so by strafing in the air by moving in different directions. The agent will receive info on its immediate surroundings and its distance from the goal and then output a direction to move in.

## AI/ML Algorithms
We anticipate using reinforcement learning/Deep Q learning to figure out the best movements our agent should take.

## Evaluation Plan
A successful AI will be able to safely land in a pool of water from a high location while taking minimum damage. Our metrics will be the survival of our agent, how close our agent is to the goal (height-wise) and the total amount of health lost. Ideally the agent should be able to reach the pool of water without taking any damage, and will be punished for hitting an obstacle even if it survives.

The first sanity check is to make sure our agent can recognize obstacles and react to the environment. Our moonshot/goal would be to reach the bottom without taking any damage. 
We can calculate how well we're doing by tracking how much hp is lost, or by how far down the dropper the agent lives before it dies.

## Appointment with the Instructor
Time: 9:45 am - Wednesday, October 21, 2020
