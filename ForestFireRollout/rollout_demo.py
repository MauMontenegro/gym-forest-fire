import helicopter
import heuristic
import rollout

# -----------------------Preparing the experiment----------------------------
# Environment parameters
N_ROW = 16              #Grid Rows
N_COL = 16              #Grid Columns
Init_Row=7              #Helicopter initial row
Init_Col=7              #Helicopter initial col
P_FIRE = 0.03           #Probability to turn a green cell into ared cell
P_TREE = 0.1            #Probability to turn an empty cell into a green cell
FREEZE = 1              #Movements of Helicopter after update Automata
# Symbols for cells
TREE = 0
FIRE = 2
EMPTY = 1
# Experiment parameters
N_TRAIN = 1
N_STEPS = 20
N_SAMPLES = 20
K_Rollout = 10
LOOKAHEAD = 2

"""def HH(a):
    env = a['env']
    actions = list(env.action_set)
    return actions[0]
"""

if __name__ == '__main__':
    env = helicopter.EnvMakerForestFire(n_row = N_ROW, n_col = N_COL, 
        p_tree = P_TREE, p_fire = P_FIRE,
        init_pos_row = Init_Row, init_pos_col = Init_Col, 
        moves_before_updating = FREEZE,  
        tree = TREE, empty = EMPTY, fire = FIRE)
    H = heuristic.Heuristic
    exp = rollout.Experiment(env, H, H_ARGS={'vision':1},
        N_TRAIN=N_TRAIN, N_STEPS=N_STEPS,N_SAMPLES=N_SAMPLES, 
        K=K_Rollout, LOOKAHEAD=LOOKAHEAD, MIN_OBJECTIVE=True, N_WORKERS=-1)
    exp.run()#GIF=True)
    #exp.policy_test(N_TEST=5,N_STEPS=15)
    #exp.make_gif(RUN=True) 