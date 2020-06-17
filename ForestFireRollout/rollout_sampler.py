# Rob was here

import math

class R_ITER():
    """Class for to save a bit of space in creation of iterables with repeated items
    for a iterable object where just the action variable changes, iterating from the
    Action_set.
    Environment must have a .copy() method"""
    def __init__(self, env, H, N_workers, alpha, K, lookahead, N_samples, Action_set=None, H_args=None, Min_obj=True):
        self.env = env
        self.H_ref = H
        self.Min_obj = Min_obj
        self.cpus = N_workers
        self.alpha = alpha
        if Action_set is None:
            Action_set = env.action_set
        assert isinstance(Action_set, (list, set)), "Action_set variable must be a list or set. {} was given".format(type(Action_set))
        self.actions = list(Action_set)
        self.actions_c = len(self.actions)
        self.K = K
        self.lookahead = lookahead
        self.N_samples = N_samples
        self.H_args = H_args
        self.chuncks = math.ceil(self.actions_c / self.cpus) # Size of chuncks
        self.j = lookahead - 1
        self.max_index = []
        self.t_pos = []
        for _ in range(self.lookahead):
            self.t_pos += [0]
            self.max_index += [self.actions_c - 1]
        self.t_pos[self.lookahead - 1] = -1 # A quick fix.
        self.state = True

    def __iter__(self):
        # iterable method
        return self

    def __next__(self):
        if not self.state:
            raise StopIteration
        # iterating through the object
        chunck_trayectories = []
        for _ in range(self.chuncks):
            branch = []
            self.state = self._update_iter_()
            if not self.state:
                break
            for pos in self.t_pos:
                branch += [self.actions[pos]]
            chunck_trayectories += [branch]

        To_H = {
                'env': self.env.copy(), # Always a new copy of the environment. This is a new object.
                'H':self.H_ref,
                'trayectories':chunck_trayectories,
                'alpha':self.alpha,
                'K':self.K,
                'N_SAMPLES':self.N_samples,
                'H_args':self.H_args,
                'min_obj':self.Min_obj,
        }
        return To_H

    def __del__(self):
        return None

    def _update_iter_(self):
        # A recursive counter
        if self.j < 0:
            # Done
            return False
        elif self.t_pos[self.j] < self.max_index[self.j]:
            # This case is when there are still branches to 
            # visit in this level
            self.t_pos[self.j] += 1
            self.j = self.lookahead - 1
            # End of update
            return True
        else:
            self.t_pos[self.j] = 0
            self.j -= 1
            return self._update_iter_()
            

def sample_trayectory(args):
    """
    This function is the sampler for the rollout function design to 
    run in one thread for parallel calling. It supports a list of 
    trayectories. At the end returns the best action with the expected
    cost associated.

    It returns the expect value from taking the action.
    (action type, float)

    args - dict
    -------------
    'env' : Variable to reference the environmet to work with.
        It is expect to be a COPY from the
        object class Helicopter for this version.
    'trayectories' : list
        An order list with the actions to perfomr to make a trayectory of controls
        to sample.
    'H' : Heuristic to perform the next K steps after executing the action.
    'H_args'* : Optional argument dict for H.
    'alpha' : Discount factor for the cost function.
    'K', Integer of number of steps to keep executing the heuristic.
    'N_SAMPLES' : Number of samples required to calculate the expected value of the
        cost function.
    'min_obj' : bool
        The objective of the samples. To minimize or to maximize if false.
    """
    LIMIT_OF_STEPS = 10**3
    # loading variables
    expected_args = ['env', 'trayectories', 'H', 'alpha', 'K', 'N_SAMPLES','min_obj']
    assert isinstance(args, dict), "This function requieres a dict of kwards to perform. Please provide one"
    for arg in expected_args:
        if args.get(arg) is None:
            raise "At least the argument {} is missing! Please do pass correct arguments".format(arg)
    env = args['env']

    trayectories = args['trayectories']
    l_trayectories = len(trayectories)
    l_trayectory = len(trayectories[0])
    assert l_trayectory > 0, "This function should not sample empty trayectories. Please check your inputs."
    
    H = args['H']
    H_args = args['H_args'] #Dictionary of aditional args to the H
    To_H = dict()
    To_H['env'] = env # Dict to pass the Heuristic
    if isinstance(H_args, dict): # There are extra arguments to pass
        for extra in H_args.keys():
            To_H[extra] = H_args[extra]

    alpha = args['alpha']
    
    K = args['K']
    if K < 0: 
        K = LIMIT_OF_STEPS #Executing the heuristic until the limit or when it's done.
    N_SAMPLES = args['N_SAMPLES']

    min_obj = args['min_obj']
    objective = 1
    if not min_obj:
        objective = -1
    min_cost = math.inf*objective
    best_action = None

    # making a checkpoint to load to the start of the sampled trayectory
    initial_checkpoint = env.make_checkpoint()
    # Starting from a trayectory from the list
    for t in range(l_trayectories):
        trayectory = trayectories[t]
        total_cost = 0.0
        for _ in range(N_SAMPLES):
            ALPHA = alpha # Start the path with Alpha
            sample_cost = 0.0
            action_cost = 0.0
            # Executing the trayectory branch
            action = 0
            for action in range(l_trayectory): # Lookahead
                observation, action_cost, done, _ = env.step(trayectory[action])
                sample_cost += ALPHA * action_cost
                ALPHA = ALPHA * alpha
            # Executing the heuristic for k steps
            for _ in range(K):
                if done: # Here to chance to cancel the loop if necesary
                    break #The environment is terminated. Branch is closed.
                to_H = To_H.copy()
                to_H['observation'] = observation
                action_from_H = H(to_H)
                observation, action_cost, done, _ = env.step(action_from_H)
                # Extract heuristic cost
                sample_cost += ALPHA * action_cost
                ALPHA = ALPHA * alpha
            #End of the heuristic running
            # Termination value is estimated to 0 at the moment. 
            # Here could be an approximator from the last observationfrom the environment.
            sample_cost += 0
            # Adding the sample cost to the average cost of the samples
            total_cost += sample_cost / N_SAMPLES
            # Restarting the environment from the initial state.
            env.load_checkpoint(initial_checkpoint)
            # keeping the best only!
            if objective*total_cost < objective*min_cost:
                best_action = trayectory[0] # Saving the action with better sampled cost
                min_cost = total_cost
    # All trayectories sampled!
    del env # Closing the copy of the environment
    # It returns the best cost and action from its samples.
    return (best_action, min_cost)
