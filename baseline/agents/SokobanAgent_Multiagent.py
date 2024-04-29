from collections import defaultdict
import numpy as np
import gymnasium as gym
class SokobanAgentMARL:
    def __init__(
        self,
        name:'ALPHA',
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.name = name
        self.q_values = defaultdict(lambda: np.zeros(9)) # 9 actions

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        #make reachability table
        self.reachabilityT = {}

        #make a list of all states
        self.allSeenStates = {}

    def get_action(self, obs: any, env: gym.Env) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.unwrapped.action_space(self.name).sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            obs = tuple(obs.flatten())
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: any,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: any,
    ):


        #maintain a list of every state seen
        #maintain a reachability table
        obs = str(obs.flatten())
        next_obs = str(next_obs.flatten())

        #if a from or too state is new add it to the reachability table and set all values to -1
        if obs not in self.allSeenStates:
            self.allSeenStates[obs] = len(self.allSeenStates)
            obsi = self.allSeenStates[obs]
            self.reachabilityT[obsi] = {}
            for s in range(len(self.allSeenStates)):
                self.reachabilityT[obsi][s] = np.inf
                self.reachabilityT[s][obsi] = np.inf
            # set reachability to to = 0
            self.reachabilityT[obsi][obsi] = 0

        if next_obs not in self.allSeenStates:
            self.allSeenStates[next_obs] = len(self.allSeenStates)
            next_obsi = self.allSeenStates[next_obs]
            self.reachabilityT[next_obsi] = {}         
            for s in range(len(self.allSeenStates)):
                self.reachabilityT[next_obsi][s] = np.inf
                self.reachabilityT[s][next_obsi] = np.inf
            # set reachability from from = 0
            self.reachabilityT[next_obsi][next_obsi] = 0


        obsi = self.allSeenStates[obs]
        next_obsi = self.allSeenStates[next_obs]

        # set reachability from to = 1
        self.reachabilityT[obsi][next_obsi] = 1


        #in the from row of p
            #fill in the numbers for every to in the from row of n +1 min the number already there

        #update the reachability distance of all of the states from the starting position of the state
        #with the reachibility distance accessible from the ending postition + 1 if it's a shorter path
        for to in range(len(self.allSeenStates)):#for every state
            toto = self.reachabilityT[next_obsi][to]#here is the reachability distance of that state from the ending position
            self.reachabilityT[obsi][to] = min(self.reachabilityT[obsi][to], toto + 1)#this updates the reachability of the starting postition



        #if there is a state that can lead to the starting state to the step we're looking at
        #then update the rachability distance for all of those rows with the reachability list of the starting state + 1
        #if it's less than the current known shortest path
        for fromRow in range(len(self.allSeenStates)):#for every state
            if self.reachabilityT[fromRow][obsi] != np.inf:#check if the starting state is accessible from the state we're looking at
                distToFrom = self.reachabilityT[fromRow][obsi]#distance from this state to the starting state
                for fromTo in range(len(self.allSeenStates)):#for every state accessible from the starting state
                    fromToTo = self.reachabilityT[obsi][fromTo] #the rachibility of a state from the starting state of the action
                    self.reachabilityT[fromRow][fromTo] = min(self.reachabilityT[fromRow][fromTo], fromToTo + distToFrom)#replace the distance if shorter




        """Updates the Q-value of an action."""
        currentValueComponent = (1-self.lr) * self.q_values[obs][action]

        if not terminated:
            future_q_value = np.max(self.q_values[next_obs])
        else:
            future_q_value = 0




        ftSum = 0
        for t in range(len(self.allSeenStates)):
            if self.reachabilityT[next_obsi][t] != np.inf:
                ftSum += (1/len(self.allSeenStates))*(self.discount_factor**self.reachabilityT[next_obsi][t])


        if terminated:
            Dst = 1
        else:
            Dst = 1 - self.discount_factor 

        beta = 1
        raux = beta * Dst * ftSum


        newValueComponentWithAUX = (reward + raux) + self.discount_factor * future_q_value

        newq = currentValueComponent + self.lr * newValueComponentWithAUX

        diff = self.q_values[obs][action] - newq

        self.q_values[obs][action] = newq


        self.training_error.append(diff * 1000)


        #next_obs = tuple(next_obs.flatten())
        #obs = tuple(obs.flatten())
        #"""Updates the Q-value of an action."""
        #future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        #temporal_difference = (
        #    reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        #)

        #self.q_values[obs][action] = (
        #    self.q_values[obs][action] + self.lr * temporal_difference
        #)
        #self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)