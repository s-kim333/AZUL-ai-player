from advance_model import *
import operator
from utils import *
import numpy as np


class myPlayer(AdvancePlayer):

    def __init__(self, _id):
        super().__init__(_id)

    def SelectMove(self, moves, game_state):
        best_move = None
        available_moves = game_state.players[self.id].GetAvailableMoves(game_state)
        moveset = {}

        for move in moves:
            state = copy.deepcopy(game_state)
            state.ExecuteMove(self.id,move)
            score, tiles = state.players[self.id].ScoreRound()
            moveset[move] = score

        best_move = max(moveset.items(), key=operator.itemgetter(1))[0]

        return best_move