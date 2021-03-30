

# Project: comp90054-2020s1-azul
# Group Members: Nathan Batham, Sejin Kim, Rajeong Moon
# Year: 2020
# Agent: Game Theory - Extensive form + greedy algorithm



# Imports
from advance_model import *
from utils import *
import time


# Class that defines the player
class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)

    # Remove strange moves in available moves
    # (e.g. take tile 1 from F1 and move floor line even if there are available pattern lines)
    def remove_strange_moves(self, moves, game_state, player_id):

        # Get number of tiles and tile type of patter line
        lines_number = game_state.players[player_id].lines_number
        lines_tile = game_state.players[player_id].lines_tile

        normal = []

        # For all moves, check if strange
        for move in moves:
            available_line = False

            # Check if available move on pattern line
            for idx in range(5):
                expected_number = lines_number[idx] + move[2].number
                if (lines_tile[idx] == -1 or lines_tile[idx] == move[2].tile_type) and expected_number < idx + 2:
                    available_line = True
                    break

            # If move is available and the destination is not the floor, add to normal set
            if not (available_line == True and move[2].pattern_line_dest == -1):
                normal.append(move)

        # Return vetted moves
        return normal

    # Select the best move using game theory
    def SelectMove(self, moves, game_state):

        # Initialise timer
        start_time = int(round(time.time() * 1000))

        # Set player IDs
        me = self.id
        partner = 0 if self.id == 1 else 1

        # To avoid time_out, set the limitation about len(moves)
        limit_size = 45
        prune = False

        # Get current scores
        current_my_score = game_state.players[me].score
        current_partner_score = game_state.players[partner].score

        # Initialise remaining variables
        best_move = None
        left_tile = 0
        payoff = []

        # Check how many tiles are left across all factories
        for factory in game_state.factories:
            left_tile += factory.total

        # If only one possible move, return move
        if len(moves) == 1:
            best_move = moves[0]
            return best_move

        # During first turn of the first round, focus on completing pattern lines
        if game_state.players[me].score == 0 and sum(game_state.players[me].lines_number) == 0:
            factories = game_state.factories

            nFactory = 0
            max = (0, (None, 0))

            for idx, factory in enumerate(factories):
                tMax = sorted(factory.tiles.items(), key =lambda x: -x[1])[0] #return: [(tile, number), (key, value)...]
                max = (idx, tMax) if max[1][1] < tMax[1] else max

            for factory in factories:
                if max[1][0] in factory.tiles.keys():
                    nFactory = nFactory + 1

            if nFactory < 2:
                for move in moves:
                    if move[1] == max[0] and move[2].tile_type == max[1][0] and \
                            move[2].pattern_line_dest == max[1][1]-1:
                        best_move = move
                        return best_move

            else:
                for move in moves:
                    if move[1] == max[0] and move[2].tile_type == max[1][0] and move[2].pattern_line_dest == max[1][1]:
                        best_move = move
                        return best_move

        # Remove strange moves and assign to temporary set
        tMoves = myPlayer.remove_strange_moves(self, moves, game_state, me)

        # if temp set is not empty, assign to moves
        if len(tMoves) != 0:
            moves = tMoves

        # If the number of moves is greater than the size limit, prune
        if len(moves) > limit_size:
            candidate = []

            # Execute and evaluate move on copy of game state
            for my_move in moves:
                gs = copy.deepcopy(game_state)

                gs.ExecuteMove(me, my_move)
                gs.players[me].ScoreRound()
                gs.players[me].EndOfGameScore()
                candidate.append((my_move, gs.players[me].score, gs))

            moves = sorted(candidate, key=lambda x: -x[1])[:limit_size]
            prune = True


        # Evaluate each remaining move based on potential score - following potential opponent score
        for my_move in moves:

            # Variable initialisation
            gs = None
            potential_score = None
            p_moves = None


            # If move not executed during prune, execute and evaluate. Get potential score
            if prune == False:
                gs = copy.deepcopy(game_state)

                gs.ExecuteMove(me, my_move)
                gs.players[me].ScoreRound()
                gs.players[me].EndOfGameScore()
                potential_score = gs.players[me].score

            else:
                gs = my_move[2]
                potential_score = my_move[1]
                my_move = my_move[0]

            # Get opponent potential moves
            partner_state = gs.players[partner]
            p_moves = partner_state.GetAvailableMoves(gs)

            # If run time greater than 0.9 seconds, break
            time_marker = int(round(time.time() * 1000))
            if (time_marker - start_time) >= 900:
                break

            # If not end of round, evaluate opponent moves
            if len(p_moves) > 0:
                best_score = -999

                # For each move, copy game state, execute move and score
                for move in p_moves:
                    p_gs = copy.deepcopy(gs)

                    p_gs.ExecuteMove(partner, move)
                    p_gs.players[partner].ScoreRound()
                    p_gs.players[partner].EndOfGameScore()
                    potential_p_score = p_gs.players[partner].score

                    # Calculate temporary opponent payoff
                    tPayoff = potential_p_score - current_partner_score

                    # Find opponent score with best payoff
                    if tPayoff > best_score:
                        best_score = tPayoff

                # Add move and difference in player and opponent scores and best score to set
                payoff.append((my_move, potential_score - current_my_score - best_score))

            else:
                # Add move and difference in player and opponent scores to set
                payoff.append((my_move, potential_score - current_my_score))


        # Sort moves based on payoff
        payoff = sorted(payoff, key=lambda x: (-x[1]))

        # Return move with best payoff
        return payoff[0][0]

