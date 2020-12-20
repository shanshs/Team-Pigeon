---
layout: default 
title: Final Report 
---

## Video

## Project Summary
Our project attempts to have an agent complete a “Dropper” map, which is a map in which a player attempts to fall safely to the bottom while avoiding obstacles. Our agent will be completing a map in which there is a 5 block wide and 250 blocks high tunnel. Inside the tunnel, there are some randomly generated obsidian obstacles at different locations (By randomly generating the map, the agent sometimes meets a more complex situation: for example, it can go left to avoid hitting an obstacle, but that would cause a dead-end or narrow escape further down). At the bottom of the tunnel, there is a pool of water.

The agent will spawn at the top of the tunnel and begin falling from it. The goal of the mission is to let the agent reach the bottom of the tunnel and land in the pool of water safely without taking any damage. In this process, the agent will strafe in the air by moving in different directions to avoid all obsidian obstacles inside the tunnel to avoid dying from fall damage and make its way to the bottom.

<div style="text-align:center"><img src="summary3.png" width="500" height="330"/></div>

This problem isn’t trivial because the solution can be quite complex. There are a lot of ways to reward and punish the agent and finding the best way takes a lot of time. The falling speed of the agent raises another challenge and it is the biggest challenge of this project since we are not able to control the falling speed of the agent, so we can only adjust the value of other parameters to adapt it. Our team has to find the most suitable obstacle density and optimal timing for the agent to perform actions while falling. If the time between actions is too long, the agent makes less move so it will dodge less obstacles, lowering the probability of success. If the time is too short, the rewards do not get registered correctly by Malmo. Also, if the gap between two obstacles is too short, the agent would not have enough time to make a move. The goal for the map and obstacle creation was to make the obstacles and map for the environment realistic and playable enough that even humans could have a decent chance of completing maps. 

To find the most proper way to solve the problems above, our team had to experiment manipulating our environment and trying out several different reinforcement learning algorithms.

## Approaches
### Activation Function
The activation function is an essential element for neural network. Choosing the right activation function for a layer is important and have a significant impact on the the training process. In the project, we have tried out different activation functions to experiement which of these give us the best result. The three activation functions shown below are used in the project:

#### Sigmoid(): 
Output values is between range 0-1, which can produce very clear prediction. For very high or low values of x, this function almost doesn't affect the prediction, which may result in the network refusing to learn.
￼
#### Tanh(): 
Zero centered, which makes negative, neutral, and strongly positive values easy to distinguish by the model. 
￼
#### ReLU(): 
This function allows the network to converge very fast. However, the gradient of the function becomes zero when inputs are close to zero, or are negative and the network cannot learn.
￼

￼


Our agent's actions are being rewarded with the following:
**Reward Function**
$$
R(s)=\left\{
	\begin{aligned}
	100 &\ (\text{Agent reaches water safely})\\
	1 &\ (\text{+1 For every millisecond the agent is alive})\\
	-25 &\ (\text{Hitting an obstacle})\\
	\end{aligned}
	\right.
$$

The agent is greatly rewarded with 100 points for completeing the objective and reaching the water safely and is punished for being damaged obstacles without fully dying receiving -25 points. However, we wanted to ensure that the agent wouldn't spend time randomly moving until it luckily lands in the water or hits an obstacle so we added another reward for staying alive as long as possible in order to encourage the agent to dodge obstacles and get lower in the tunnel. Thus we gave the agent 1 point for every millisecond of in-game time it remained alive.

To keep simplicity, our agent always faces the north and only takes discrete actions consisting of:

* Strafe right
* Strafe left
* Move forward
* Move backwards
* Stand still


### DQN
We are primarily using Reinforcement Learning for our approach for the project, using the below DQN algorithm.

$$
Q(S_t, A_t)\leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma\max_a Q(S_{t+1},a)- Q(s_t, A_t)]
$$

The algorithm can be interpreted as this:
(Updated Q Value) --> (Current Q Value) + (Learning Rate)[Reward + (Discount Rate)*(Max Expected Reward)- (Current Q Value)]

Our agent will take a 3 x 30 x 3 grid of the blocks around it as observation. Since our map is a vertical tunnel and the agent drops from the top of the tunnel, the falling speed of the agent is fast. Therefore, we set the grid so that the agent can 'see' 30 levels or blocks down in order for it to have enough time to dodge obstacles.

Each episode will see the agent taking the above actions until it reaches a terminal state of either successfully landing in the pool of water or dying from hitting an obstacle.

We used several different algorithmic approaches ranging from DQN learning and PPO to train our agent:

For one of our DQN approaches:

```
For each episode:
   While the agent hasn't reached the pool of water or dies:
      Choose an action by applying epsilon greedy policy from Q network
      Take the action
      Get the next observation by check the 3*30*3 grid around the agent
      Calculate the reward
      Update Q network
```

To get the action of each step, we apply the following function from lecture:

$$
\pi(a|s)=\left\{
	\begin{aligned}
	\epsilon/m+1- \epsilon \text{ }[\text{ if a*} = argmax_{(a \in A)} Q(s,a)]\\
	\epsilon/m  \text{  otherwise}\\
	\end{aligned}
	\right.
$$
 
We created an list called action_prob to save the probabilities for each action. Then, we calculate action_prob based on the formula above and randomly choose an action based on the probabilities in action_prob list.

< Explanation of additional layers to the dqn to be included here >

#### Rainbow DQN

We also experimented with one approach as an extension to DQN: Rainbow DQN, a combination of several improvements DQN algorithms. RLlib provided us with an implementation of a standard DQN algorithm, and with a few given modifications we were able to easily transform it to Rainbow DQN.

Essentially, Rainbow DQN is a dubbed term of a combination of extensions of DQN. These include the following: Double Q-learning, prioritized replay, dueling networks, multi-step learning, distributional reinforcement learning, and noisy neural network stream. A more in-depth explanation of these terms are included in the research paper in our [source](https://arxiv.org/pdf/1710.02298.pdf) .


### PPO

We also experimented another reinforcement learning algorithm called Proximal Policy Optimization (PPO) using the Ray RLlib library. PPO explores by sampling actions according to its latest version of its stochastic policy. This randomness depends on the initial conditions of the environment. The stochastic policy eventually becomes less random and encourages the agent to utilize paths to rewards it has found already.

Below is a standard algorithm for PPO, although we used RLlib's implementation of it for use in our project.
```
For each episode:
	For each agent:
		Run stotastic grading policy in environment for set number of time
		Compute advantage estimates at each timestamp
	Optimize surrogate loss function with respect to old policy parameters, with K epochs and a minibatch size ≤ to the number of agents * timesteps
	Replace the old vector of policy parameters with the newly formed one
```

 
## Evaluation


| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

### DQN

<div style="text-align:left"><img src="returns_torch_dqn.png" width="450" height="290"/></div>

| Episode Range   	| Average return   	| Success rate (%) 	|
|-	|-	|-	|
| 0-500 	| 36.1 	| 4.2 	|
| 500-1000 	| 39.6 	| 6.6 	|
| 1000-1500 	| 40.9 	| 7.4 	|
| ... 	| ... 	| ... 	|
| 5000-5500 	| 48.9 	| 9.9 	|
| 5500-6000 	| 50.6 	| 10.6 	|
| ... 	| ... 	| ... 	|
| 13000-13700 	| 56.8 	| 12.8 	|
| 13500-13900 	| 54.1 	| 12.1 	|

### Rainbow DQN

<div style="text-align:left"><img src="returns-dqn-rainbow.png" width="450" height="290"/></div>

| Episode Range    	| Average return    	| Success rate (%)    	|
|-	|-	|-	|
| 0-200 	| 40.9 	| 10.4 	|
| 200-400 	| 44.9 	| 12 	|
| 400-600 	| 52.6 	| 17.5 	|
| 600-800 	| 67.2 	| 29 	|
| 800-1000 	| 77.1 	| 34.5 	|
| 1000-1200 	| 81.5 	| 39 	|
| ... 	| ... 	| ... 	|
| 3200-3400 	| 102.2 	| 54.5 	|
| 3400-3600 	| 99.8 	| 52.5 	|
| 3600-3800 	| 97.2 	| 49.5 	|
| 3800-4000 	| 92.9 	| 47.5 	|
| 4000-4200 	| 99.6 	| 52 	|

### PPO

<div style="text-align:left"><img src="returns_pponew.png" width="450" height="290"/></div>

| Episode Range   	| Average return   	| Success rate (%) 	|
|-	|-	|-	|
| 0-500 	| 28.0 	| 1.9 	|
| 500-1000 	| 35.5 	| 2.8 	|
| 1000-1500 	| 46.1 	| 6.6 	|
| 1500-2000 	| 48.9 	| 9.6 	|
| ... 	| ... 	| ... 	|
| 5000-5500 	| 62.1 	| 16.3 	|
| 5500-6000 	| 52.8 	| 14.9 	|
| ... 	| ... 	| ... 	|
| 6500-7000 	| 61.9 	| 21.16 	|
| 7000-7250 	| 60.9 	| 18 	|
| 7250-7500 	| 62.4 	| 20.3 	|
| 7500-7800 	| 69.5 	| 26.19 	|


## References
CS175 Assignment 2's DQN algorithm's episilon greedy policy


[Malmo](http://microsoft.github.io/malmo/0.30.0/Documentation/)

[RLlib](https://docs.ray.io/)

[Rainbow DQN](https://arxiv.org/pdf/1710.02298.pdf)

[PPO](https://arxiv.org/pdf/1707.06347.pdf)

PyTorch and TensorFlow library
