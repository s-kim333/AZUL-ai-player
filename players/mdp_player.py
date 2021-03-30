# Project: comp90054-2020s1-azul
# Group Members: Nathan Batham, Sejin Kim, Rajeong Moon
# Year: 2020
# Agent: MDP model with value iteration & adopted payoff from game theory



from advance_model import *
import operator
from utils import *
import numpy as np

previous_action_values = {}

class myPlayer(AdvancePlayer):

    def __init__(self, _id):
        super().__init__(_id)

    def SelectMove(self, moves, game_state):
        best_move = None
        self.game_state = game_state

        gamma = 0.75
        pos_moves = moves

        action_values = expected_action_value(pos_moves, self.game_state, self.id, gamma)

        val_col = []
        for key in action_values:
            state, point = action_values[key]
            val_col.append(point)

        # Get maximum value in val_col and update value in immediate_values
        best_next = max(val_col)
        for key, value in action_values.items():
            val1,val2 = value
            if val2 == best_next:
                best_move = key


        return best_move

def expected_action_value(moves, game_state, _id, discount_factor):
    if _id ==0:
        partner_id = 1
    else:
        partner_id = 0
    V = {}
    actions = moves


    actions = sorted(actions, key=lambda x: (x[2].num_to_floor_line, -x[2].num_to_pattern_line))
    
    # reduce number of possible next states to avoid time out 
    explore_limit = 10
    if len(actions) > explore_limit:
        actions = actions[:explore_limit]

    # get difference between player and opponent in current state to see what's actual payoff for the player
    
    game_state.players[partner_id].score = game_state.players[_id].ScoreRound()[0]
    game_state.players[partner_id].EndOfGameScore()
    partner_curr_point = game_state.players[partner_id].score

    game_state.players[_id].score = game_state.players[_id].ScoreRound()[0]
    game_state.players[_id].EndOfGameScore()
    curr_point = game_state.players[_id].score

    curr_score = curr_point - partner_curr_point

    for action in actions:
        next_state = copy.deepcopy(game_state)
        next_state.ExecuteMove(_id, action)
        V[action] = (next_state, 0)

    # instead of repeat until convergence repeat is reduced to 10 times to avoid time-out
    repeat = 10

    while repeat > 0:
        repeat -= 1
        for key in V:
            state, score = V[key]
            # calculate expected pay-off
            state.players[_id].score = state.players[_id].ScoreRound()[0]
            state.players[_id].EndOfGameScore()
            next_point = state.players[_id].score

            state.players[partner_id].score = state.players[partner_id].ScoreRound()[0]
            state.players[partner_id].EndOfGameScore()
            next_partner_point = state.players[partner_id].score

            next_point = next_point - next_partner_point


            # Update V value
            point = 0
            for action in actions:
                if action == key:
                    point += 11/20*(curr_score+discount_factor*next_point)
                else:
                    point += 1/20*(curr_score+discount_factor*next_point)

            V[key] = (state, point)

        c_val = [] # current value
        
        # V[action] max -> current update
        for key,value in V.items():
            state, point = value
            c_val.append(point)
        curr_score = max(c_val)

    return V

