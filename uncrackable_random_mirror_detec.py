import numpy as np
import pandas as pd

def uncrackable_rand(observation, configuration):
    np.random.seed()
    return np.random.randint(0,3)
    
def rand(observation, configuration):
    return np.random.randint(0,3)

def copy_opponent_agent(observation, configuration):
    if observation.step > 0:
        return observation.lastOpponentAction
    else:
        return 0

def uncrackable_and_mirror_detector(observation, configuration):
    # this agent's mirror detection portion inspired by this public agent:      https://www.kaggle.com/ilialar/beating-mirror-agent
    # two functions to save data using pandas
    """
    Possible weaknesses:
        If the opponent is a more complex mirror system where it adds in random moves once in a while, then this agent won't detect it
    """
    def save_history(history, file = 'history.csv'):
        pd.DataFrame(history).to_csv(file, index = False)

    def load_history(file = 'history.csv'):
        return pd.read_csv(file).to_dict('records')
    
    # uncrackable_rand function
    def uncrackable_rand(observation, configuration):
        np.random.seed()
        return np.random.randint(0,3)
    
    # main function code
    if observation.step == 0:
        # first round of the game, must define the history dictionary
        # always returns a random move
        move = uncrackable_rand(observation, configuration)
        history = [{'step': move, 'competitorStep': None}]
        save_history(history)
        return move
    elif observation.step <= 20:
        # rounds 2-20 of the game, where 20 is the arbitrary number to determine if the opponent is a mirror bot
        # continues to save information, including this agent's current move and the opponent's move in the last round
        # always returns a random move
        history = load_history()
        history[-1]['competitorStep'] = observation.lastOpponentAction
        move = uncrackable_rand(observation, configuration)
        history.append({'step': move, 'competitorStep': None})
        save_history(history)
        return move
    else:
        # rounds 21+ of the game
        # checks if the last 20 of the opponent's moves were mirrors, if so then handles them appropriately, else keeps random moves
        mirror_indicator = True
        history = load_history()
        history[-1]['competitorStep'] = observation.lastOpponentAction
        for index, dict in enumerate(history):
            if index > 0:
                #print("Step = %s, Index = %d" % (observation.step, index))
                #print("   dict['competitorStep'] = %d" % dict['competitorStep'])
                #print("   history[index-1]['step'] = %d" % history[index-1]['step'])
                if dict['competitorStep'] != history[index-1]['step']:  # if 
                    mirror_indicator = False
        if mirror_indicator:
            #print("Step = %s, detected mirror" % observation.step)
            move = (history[-1]['step'] + 1) % 3
            history.append({'step': move, 'competitorStep': None})
            save_history(history)
            return move
        else:
            #print("Step = %s, no mirror detected" % observation.step)
            move = uncrackable_rand(observation, configuration)
            history.append({'step': move, 'competitorStep': None})
            save_history(history)
            return move