import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_table = {}
        self.possible_actions = [None, 'forward', 'left', 'right']
        #探索概率
        self.explore_p = 0.5
        self.count_trail = 0
    
    

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.count_trail = self.count_trail + 1
    
    def get_state(self):
        display_inputs = self.env.sense(self)
        current_states_text = "light: {}, oncoming: {}, left: {}, right: {}, next_waypoint: {}".format(display_inputs['light'],display_inputs['oncoming'],display_inputs['left'],display_inputs['right'],self.next_waypoint)
        return current_states_text


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        
        current_states = (inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'],self.next_waypoint)
        current_action = random.choice([None, 'forward', 'left', 'right'])

        if not self.q_table.has_key((current_states,current_action)):
            self.q_table.setdefault((current_states,current_action))
            self.q_table[(current_states,current_action)] = random.random()
            for act in self.possible_actions:
                if act != current_action:
                    if not self.q_table.has_key((current_states,act)):
                        self.q_table.setdefault((current_states,act))
                        self.q_table[(current_states,act)] = random.random()
        

        
        # TODO: Select action according to your policy
        max_q_value_c = -100.0
        max_q_action_c = current_action
        for (tup0, tup1), value in self.q_table.items():
            if tup0 == current_states:
                for act in self.possible_actions:
                    if self.q_table[(tup0,act)] > max_q_value_c:
                        max_q_value_c = self.q_table[(tup0,act)]
                        max_q_action_c = act
    
    
        if  self.count_trail <= 5:
            if random.random() >= self.explore_p:
                current_action = max_q_action_c
        else:
            current_action = max_q_action_c


        action = current_action #self.next_waypoint

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        next_inputs = self.env.sense(self)
        n_next_waypoint = self.planner.next_waypoint()
        next_action = n_next_waypoint
        #n_deadline = self.env.get_deadline(self)
        next_states = (next_inputs['light'],next_inputs['oncoming'],next_inputs['left'],next_inputs['right'],next_action)
        
        max_q_value_n = -100.0
        max_q_action_n = next_action
        if not self.q_table.has_key((next_states,next_action)):
            self.q_table.setdefault((next_states,next_action))
            self.q_table[(next_states,next_action)] = random.random()
            for act in self.possible_actions:
                if act != max_q_action_n:
                    if not self.q_table.has_key((next_states,act)):
                        self.q_table.setdefault((next_states,act))
                        self.q_table[(next_states,act)] = random.random()
        else:
            for (tup0, tup1), value in self.q_table.items():
                if tup0 == next_states:
                    for act in self.possible_actions:
                        if self.q_table[(tup0,act)] > max_q_value_n:
                            max_q_value_n = self.q_table[(tup0,act)]
                            max_q_action_n = act

        next_action = max_q_action_n

        # TODO: Learn policy based on state, action, reward
        alpha = 0.7 # learning rate 0.4
        gamma = 0.3 # discount factor
        self.q_table[(current_states,current_action)] = (1 - alpha) * self.q_table[(current_states,current_action)] + alpha * (reward + gamma * self.q_table[(next_states,next_action)])
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
