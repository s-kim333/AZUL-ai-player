

# Project: comp90054-2020s1-azul
# Group Members: Nathan Batham, Sejin Kim, Rajeong Moon
# Year: 2020
# Agent: Q-learning with Linear Function Approximation




from advance_model import *
from utils import *
import numpy as np

import json


import os
from os import path


class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)



    def SelectMove(self, moves, game_state):

        # Session Variables
        episodes = 75       # Number of episodes used in training
        train = False       # Train weights or use model

        alpha = 0.0001         # Learning Rate
        beta = 1            # Weight size penalty scaling factor

        gamma = 0.9         # Discount factor
        eps = 0.8           # E-Greedy Threshold

        # delta = 10

        # Read weights from file, if file exists


        # if path.exists("./players/team17/weights_early_game.txt") and len(moves) > delta:
        #     # os.write(0, b'Early Game Read\n')
        #     with open('./players/team17/weights_early_game.txt', 'r') as f:
        #         weights = json.load(f)
        #
        #     f.close()
        #
        # elif path.exists("./players/team17/weights_late_game.txt") and len(moves) <= delta:
        #     # os.write(0, b'Late Game Read\n')
        #     with open('./players/team17/weights_late_game.txt', 'r') as f:
        #         weights = json.load(f)
        #
        #     f.close()



        if path.exists("./players/team17/weights.txt"):

            with open('./players/team17/weights.txt', 'r') as f:
                weights = json.load(f)

            f.close()
        # If file does not exist, use predefined initialisation
        else:
            # weights = [14, 7.3, 6.7, 10, 12, 10, 10, 10]
            # weights = [5, 5, 5, 5, 7, 7, 10, 7, 5, 5, 5, 5]
            weights = [0.05, 0.05, 0.15, 0.3, 0.5, 1, 0.5, 0.8, 0.1, 0.2, 0.2, 0.5]


        # Read Past Reward
        if path.exists("./players/team17/old_score.txt") and not path.getsize("./players/team17/old_score.txt") == 0:
            with open('./players/team17/old_score.txt', 'r') as f:
                old_score = json.load(f)

            f.close()
        # If file does not exist, reset reward
        else:
            old_score = 0.0

        # Reset score at beginning of game
        board_clear = 0
        for i in range(5):
            for j in range(5):
                board_clear += game_state.players[self.id].grid_state[i][j]

        if board_clear == 0:
            old_score = 0

        # os.write(0, b'old score: %f\n' % old_score)




        # DEBUG
        # os.write(0, b'----------------------------------- MOVE -----------------------------------\n')

        # Assign root node
        root = QNode(game_state, self.id, root=True)
        root.old_score = old_score


        # Initialise session
        session = QLearn(root, weights)

        # Run session and get best action index and updated weights (if training)
        best_action, weights = session.best_move(episodes, train, alpha, beta, gamma, eps)

        # Select best move using best action index
        best_move = moves[best_action]

        # DEBUG
        # os.write(0, b'Best Move: %d\n' % best_action)

        # Save weights to file if training
        # if train:
        #     if len(moves) > delta:
        #         # os.write(0, b'Early Game Write\n')
        #         with open('./players/team17/weights_early_game.txt', 'w') as f:
        #             json.dump(weights, f)
        #         f.close()
        #     else:
        #         # os.write(0, b'Late Game Write\n')
        #         with open('./players/team17/weights_late_game.txt', 'w') as f:
        #             json.dump(weights, f)
        #         f.close()


        with open('./players/team17/weights.txt', 'w') as f:
            json.dump(weights, f)
            # for i in weights:
            #     f.write("%f," % i)
            f.close()

        new_score = game_state.players[self.id].score
        with open('./players/team17/old_score.txt', 'w') as f:
            json.dump(new_score, f)
            # for i in weights:
            #     f.write("%f," % i)
            f.close()


        # (mid, fid, tgrab) = best_move
        # os.write(0, b'Tile Type = %d, Dest = %d, num pat = %d, num floor = %d\n' % (tgrab.tile_type, tgrab.pattern_line_dest, tgrab.num_to_pattern_line, tgrab.num_to_floor_line))

        return best_move






## Game states represented as nodes to be used in q-learning
class QNode(object):
    def __init__(self, game_state, _id, parent=None, reward = 0, depth = 0, q = 0, root = False):

        self.q = q
        self.GRID_SIZE = 5
        self.game_state = game_state
        self.tgrab = None
        self.fid = None
        self.parent = parent
        self.children = []
        self.num_visits = 0.
        self.reward = reward
        self.potential_moves = None
        self.root = root
        self.terminal = False
        self.expanded = False
        self.id = _id
        self.depth = depth
        self.f = None
        self.opponent_id = np.abs(self.id - 1)
        self.old_score = 0


    # Returns best action from children based on q-value
    def get_best_action(self):

        q_values = np.zeros(len(self.children))

        for i in range(len(self.children)):
            q_values[i] = self.children[i].q

        return np.argmax(q_values)

    def get_best_action_3(self):

        eps = 0.000001
        best_action = -1
        best_q = -1
        q_values = np.zeros(len(self.children))

        for i in range(len(self.children)):
            if self.children[i].q > best_q:
                best_q = self.children[i].q
                best_action = i
            elif np.abs(self.children[i].q - best_q) < eps:
                best_q = self.children[i].q
                best_action = i

        # os.write(0, b'Best Action %d\n' % best_action)
        return best_action

    def get_best_action_2(self):

        r_values = np.zeros(len(self.children))

        for i in range(len(self.children)):
            r_values[i] = self.children[i].reward

        return np.argmax(r_values)

    def rescore(self, weights, beta):

        for c in self.children:
            column = c.tgrab.tile_type + c.tgrab.pattern_line_dest
            if column >= 5:
                column -= 5

            on_board = not self.game_state.players[self.id].grid_state[c.tgrab.pattern_line_dest][column] == 0
            line_full = self.game_state.players[c.id].lines_number[c.tgrab.pattern_line_dest] == \
                        (c.tgrab.pattern_line_dest + 1)
            t1 = self.game_state.players[self.id].lines_tile[c.tgrab.pattern_line_dest]
            t2 = c.tgrab.tile_type
            if (not t1 == -1 and not t1 == t2) or c.tgrab.pattern_line_dest == -1 or line_full or on_board:
                c.q = 0
            else:
                c.q = float(np.sum(np.multiply(weights, c.f)))



    # Method to expand new nodes
    def expand(self, weights, beta):

        # a = 1
        last_turn = False

        # Get potential moves
        #if not self.potential_moves:
        self.potential_moves = self.game_state.players[self.id].GetAvailableMoves(self.game_state)


        # If no potential moves, mark node as terminal
        if not self.potential_moves:
            self.terminal = True
            return

        # For each potential action, create a child node
        for action in range(len(self.potential_moves)):

            # Copy game state and execute potential move
            (mid, fid, tgrab) = self.potential_moves[action]





            new_state = copy.deepcopy(self.game_state)
            new_state.ExecuteMove(self.id, self.potential_moves[action])

            # TEST
            # temp_moves = new_state.players[self.opponent_id].GetAvailableMoves(self.game_state)
            # os.write(0, b'Temp Moves %d\n' % len(temp_moves))

            tiles_left = 0
            for factory in new_state.factories:
                tiles_left += factory.total

            if tiles_left == 0:
                turns_left = 0
                for tile in range(0, 5, 1):
                    if new_state.centre_pool.tiles[tile] >= 1:
                        turns_left += 1

                if turns_left <= 1:
                    # os.write(0, b'End Sim. Turns Left %d\n' % turns_left)
                    # a = 10
                    last_turn = True

            # Create child node using reward policy
            child_node = QNode(new_state, self.id, parent=self,
                                  # reward=(a * self.reward_policy(new_state)),
                                  reward=(self.reward_policy_4(new_state, last_turn)),
                                  depth=self.depth + 1)

            # Assign tile used in action and calculate node features
            child_node.tgrab = tgrab
            child_node.fid = fid
            child_node.f = [
                child_node.f1(),
                child_node.f2(self.game_state),
                child_node.f3(self.game_state),
                child_node.f4(self.game_state),
                child_node.f5(self.game_state, action),
                child_node.f6(self.game_state, action),
                child_node.f7(),
                child_node.f8(),
                child_node.f9(),
                child_node.f10(self.game_state),
                child_node.f11(),
                child_node.f12(self.game_state)

                # child_node.f7(new_state, weights, beta)
            ]

            # Penalty to minimise explosion of weight and q values
            # reg = pow(np.sum(weights), 2)
            # reg = np.abs(np.sum(weights))

            # DEBUG
            # os.write(0, b'F = ')
            # for i in range(len(child_node.f)):
            #     os.write(0, b'%f, ' % child_node.f[i])
            # os.write(0, b'\n')

            # Check if already on board
            column = child_node.tgrab.tile_type + child_node.tgrab.pattern_line_dest
            if column >= 5:
                column -= 5


            on_board = not self.game_state.players[self.id].grid_state[child_node.tgrab.pattern_line_dest][column] == 0

            # DEBUG
            # if on_board:
            #     os.write(0, b'On Board\n')

            line_full = self.game_state.players[child_node.id].lines_number[child_node.tgrab.pattern_line_dest] == \
                        (child_node.tgrab.pattern_line_dest + 1)


            t1 = self.game_state.players[self.id].lines_tile[tgrab.pattern_line_dest]
            t2 = tgrab.tile_type
            if (not t1 == -1 and not t1 == t2) or tgrab.pattern_line_dest == -1 or line_full or on_board:

                child_node.q = float(np.multiply(weights[0], child_node.f[0]))
            else:
                # calculate q value for child
                child_node.q = float(np.sum(np.multiply(weights, child_node.f))) # / (beta * reg)

            # DEBUG
            # os.write(0, b'q = %f\n' % child_node.q)

            # os.write(0, b'q: %f\n' % child_node.q)
            self.children.append(child_node)

        self.expanded = True


    # Policy for determining node reward
    def reward_policy(self, game_state):

        max_eog_score = 95
        new_state = copy.deepcopy(game_state)


        # Get player scores
        score_0, tiles_0 = new_state.players[self.id].ScoreRound()
        score_1, tiles_1 = new_state.players[self.opponent_id].ScoreRound()

        # Add potential bonus scores
        score_0 += new_state.players[self.id].EndOfGameScore()

        # Simulate increase in opponent score based on node depth
        score_1 += 0.1 * self.depth


        # Return normalised scores
        if score_0 + score_1 == 0:
            return 0

        # Give discounted penalty if player loses
        elif score_1 > score_0:
            return - 0.5 * float(score_1) /  (score_0 + score_1 + max_eog_score)

        else:
            return float(score_0) /  (score_0 + score_1 + max_eog_score)



    def reward_policy_2(self, game_state):
        child_state = copy.deepcopy(game_state)
        parent_state = copy.deepcopy(self.game_state)
        child_state.players[self.id].score = child_state.players[self.id].ScoreRound()[0]
        child_state.players[self.id].EndOfGameScore()
        score_child = child_state.players[self.id].score
        score_parent = parent_state.players[self.id].score

        reward = score_child - score_parent

        # DEBUG
        # os.write(0, b'Reward %d\n' % reward)
        return reward


    def reward_policy_3(self, game_state):
        child_state = copy.deepcopy(game_state)
        parent_state = copy.deepcopy(self.game_state)
        score_child = child_state.players[self.id].score
        score_parent = parent_state.players[self.id].score

        reward = score_child

        # DEBUG
        # os.write(0, b'Reward %d\n' % reward)
        return reward

    def reward_policy_4(self, game_state, last_turn):
        child_state = copy.deepcopy(game_state)
        if last_turn:
            child_state.players[self.id].score = child_state.players[self.id].ScoreRound()[0]
            # child_state.players[self.id].EndOfGameScore()

        reward = child_state.players[self.id].score - self.old_score

        # DEBUG
        # os.write(0, b'Reward %d\n' % reward)
        return reward

    # Feature: Tiles to Floor
    def f1(self):

        return 1.0 - float(self.tgrab.num_to_floor_line) / self.tgrab.number


    # # Feature: Tiles to Pattern Line - added +1 to prevent gradient of 1.0
    # def f2(self):
    #     return float(self.tgrab.num_to_pattern_line) / self.tgrab.number

    def f2(self, game_state):
        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] == \
                    (self.tgrab.pattern_line_dest + 1)


        if self.fid == -1 and self.tgrab.num_to_floor_line < self.tgrab.num_to_pattern_line and self.tgrab.num_to_floor_line <= 1 and not line_full == 1:
            most_tiles = 0
            turns_left = 0

            for tile in range(0, 5, 1):
                if game_state.centre_pool.tiles[tile] > most_tiles:
                    most_tiles = game_state.centre_pool.tiles[tile]

                if game_state.centre_pool.tiles[tile] >= 1:
                    turns_left += 1
            if turns_left <= 3:
                return 0
            else:
                # val = float(most_tiles) / 6
                val = float(self.tgrab.num_to_pattern_line) / most_tiles
                if self.tgrab.num_to_floor_line > 0:
                    val = 0.5 * val
                # DEBUG
                # os.write(0, b'Centre Tiles: %f\n' % val)
                return val

        else:
            return 0

    # Feature: Complete Sets
    def f3(self, game_state):

        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] == \
                    (self.tgrab.pattern_line_dest + 1)

        if not self.tgrab.tile_type == -1 and not line_full == 1:
            return float(self.game_state.players[self.id].number_of[self.tgrab.tile_type]) / self.GRID_SIZE
        else:
            return 0

    # Feature: Complete Rows
    def f4(self, game_state):
        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] == \
                    (self.tgrab.pattern_line_dest + 1)


        row_count = 0
        if not self.tgrab.tile_type == -1 and not line_full == 1:

            for j in range(self.GRID_SIZE):
                if not self.game_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][j] == 0:
                    row_count += 1


            return float(row_count) / self.GRID_SIZE
        else:
            return 0

    # Feature: Complete Columns
    def f5(self, game_state, action):
        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] == \
                    (self.tgrab.pattern_line_dest + 1)



        col_count = 0
        if not self.tgrab.tile_type == -1 and not self.tgrab.pattern_line_dest == -1 and not line_full == 1:

            column = self.tgrab.tile_type + self.tgrab.pattern_line_dest
            if column >= 5:
                column -= 5

            for j in range(self.GRID_SIZE):
                if not self.game_state.players[self.id].grid_state[j][column] == 0:
                    col_count += 1




            # tile_type - 1 + tile location
            # os.write(0, b'Columns: %d\n' % col_count)
            # os.write(0, b'Line: %d, Colour: %d, Cols: %d, Column: %d, action: %d\n' %
            #         (self.tgrab.pattern_line_dest, self.tgrab.tile_type, col_count, column, action))
            return float(col_count) / self.GRID_SIZE
        else:
            return 0




    # How many already in a pattern line

    def f6(self, game_state, action):

        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] ==\
                    (self.tgrab.pattern_line_dest + 1)

        #os.write(0, b'Line: %d, Colour: %d, full line: %d, action: %d\n' %
        #         (self.tgrab.pattern_line_dest, self.tgrab.tile_type, line_full, action))

        if self.tgrab.pattern_line_dest == -1 or line_full == 1:
            return 0


        elif self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] ==\
                self.tgrab.pattern_line_dest + 1 and self.tgrab.num_to_floor_line == 0:
            # os.write(0, b'TEST\n')
            return 1

        else:

            val = float(self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] -
                        self.tgrab.num_to_floor_line) / (self.tgrab.pattern_line_dest + 1)
            # os.write(0, b'Val: %f\n' % val)

            return val
            # if val < 0:
            #     return 0
            #
            # else:
            #     return val



    # def f6(self, game_state, action):
    #
    #     line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] ==\
    #                 (self.tgrab.pattern_line_dest + 1)
    #
    #     #os.write(0, b'Line: %d, Colour: %d, full line: %d, action: %d\n' %
    #     #         (self.tgrab.pattern_line_dest, self.tgrab.tile_type, line_full, action))
    #
    #     if self.tgrab.pattern_line_dest == -1 or line_full == 1:
    #         return 0
    #     elif self.tgrab.pattern_line_dest == 4 and self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest]  == 5:
    #         # os.write(0, b'Full last row\n')
    #         return 1
    #
    #     else:
    #         # return float(self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest]) / \
    #         #     (self.tgrab.pattern_line_dest + 1)
    #         return float(self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest]) / \
    #                (5 + 0.25*(self.tgrab.pattern_line_dest + 1))




    # Average going to floor next round - aimed to minimise num to floor at end of round
    def f7(self):
        num = 0
        to_floor = 0

        # Get potential moves
        if not self.potential_moves:
            self.potential_moves = self.game_state.players[self.id].GetAvailableMoves(self.game_state)

        for mid, fid, tgrab in self.potential_moves:
            to_floor += tgrab.num_to_floor_line
            num += tgrab.number

        if num <= 0:
            return 0
        else:
            return 1 - float(to_floor) / num




    # # Adjacent tiles - Could improve by counting all connecting
    # def f8(self):
    #
    #     # Execute score round (to move tiles to board)
    #     # or
    #     # Check existing tiles adjacent to destination
    #     # + tiles with completed pattern lines?
    #
    #     score = 0
    #     if not self.tgrab.tile_type == -1 and not self.tgrab.pattern_line_dest == -1:
    #         new_state = copy.deepcopy(self.game_state)
    #         temp_score, temp_tiles = new_state.players[self.id].ScoreRound()
    #
    #         column = self.tgrab.tile_type + self.tgrab.pattern_line_dest
    #         if column >= 5:
    #             column -= 5
    #
    #         if self.tgrab.pattern_line_dest + 1 < 5:
    #             if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest + 1][column] == 0:
    #                 score += 1
    #         if self.tgrab.pattern_line_dest - 1 >= 0:
    #             if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest - 1][column] == 0:
    #                 score += 1
    #         if column + 1 < 5:
    #             if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][column + 1] == 0:
    #                 score += 1
    #         if column - 1 >= 0:
    #             if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][column - 1] == 0:
    #                 score += 1
    #
    #     return float(score) / 4




    def f8(self):

        # Execute score round (to move tiles to board)
        # or
        # Check existing tiles adjacent to destination
        # + tiles with completed pattern lines?

        score = 0
        if not self.tgrab.tile_type == -1 and not self.tgrab.pattern_line_dest == -1:
            new_state = copy.deepcopy(self.game_state)
            temp_score, temp_tiles = new_state.players[self.id].ScoreRound()

            column = self.tgrab.tile_type + self.tgrab.pattern_line_dest
            if column >= 5:
                column -= 5


            above = 0
            for j in range(column - 1, -1, -1):
                val = new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][j]
                above += val
                if val == 0:
                    break
            below = 0
            for j in range(column + 1, self.GRID_SIZE, 1):
                val = new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][j]
                below += val
                if val == 0:
                    break


            left = 0
            for i in range(self.tgrab.pattern_line_dest - 1, -1, -1):
                val = new_state.players[self.id].grid_state[i][column]
                left += val
                if val == 0:
                    break
            right = 0
            for i in range(self.tgrab.pattern_line_dest + 1, self.GRID_SIZE, 1):
                val = new_state.players[self.id].grid_state[i][column]
                right += val
                if val == 0:
                    break

            if above > 0 or below > 0:
                score += (1 + above + below)

            if left > 0 or right > 0:
                score += (1 + left + right)

            # if self.tgrab.pattern_line_dest + 1 < 5:
            #     if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest + 1][column] == 0:
            #         score += 1
            # if self.tgrab.pattern_line_dest - 1 >= 0:
            #     if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest - 1][column] == 0:
            #         score += 1
            # if column + 1 < 5:
            #     if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][column + 1] == 0:
            #         score += 1
            # if column - 1 >= 0:
            #     if not new_state.players[self.id].grid_state[self.tgrab.pattern_line_dest][column - 1] == 0:
            #         score += 1

        val = float(score) / 8

        # DEBUG
        # os.write(0, b'F8 Val: %f\n' % val)
        return val


    def f9(self):
        if self.game_state.first_player_taken and self.game_state.next_first_player == self.id:
            return 1
        else:
            return 0



    def f10(self, game_state):

        line_full = game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest] ==\
                    (self.tgrab.pattern_line_dest + 1)


        if self.tgrab.pattern_line_dest == -1 or line_full == 1:
            return 0
        else:
            # return float(self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest]) / \
            #     (self.tgrab.pattern_line_dest + 1)

            tiles_left = 0
            for factory in self.game_state.factories:
                tiles_left += factory.tiles[self.tgrab.tile_type]

            tiles_left += self.game_state.centre_pool.tiles[self.tgrab.tile_type]

            val = tiles_left / (self.tgrab.pattern_line_dest + 3 - self.game_state.players[self.id].lines_number[self.tgrab.pattern_line_dest])

            if val > 1:
                val = 1

            return val

    # Diversity
    def f11(self):

        same = 0
        for i in range(5):
            if self.game_state.players[self.id].lines_tile[i] == self.tgrab.tile_type:
                same += 1

            elif self.game_state.players[self.id].lines_tile[i] == -1:
                # Check if already on board
                column = self.tgrab.tile_type + i
                if column >= 5:
                    column -= 5

                if not self.game_state.players[self.id].grid_state[i][column] == 0:
                    same += 1
        val = 1.0 - float(same) / 10
        # DEBUG
        # os.write(0, b'F11 Val: %f\n' % val)
        return val



    def f12(self, game_state):
        child_state = copy.deepcopy(self.game_state)
        parent_state = copy.deepcopy(game_state)
        child_state.players[self.id].score = child_state.players[self.id].ScoreRound()[0]
        child_state.players[self.id].EndOfGameScore()
        score_child = child_state.players[self.id].score
        score_parent = parent_state.players[self.id].score

        reward = score_child - score_parent

        if score_child == 0:
            return 0

        val = float(reward) / score_child

        if val < 0.0:
            val = 0.0
        # DEBUG
        # os.write(0, b'F Reward %f\n' % val)
        return val




    # def f7(self, new_state, weights, beta):
    #     opponent_node = QNode(new_state, self.opponent_id, parent=self, depth=self.depth + 1)
    #     opponent_node.potential_moves = \
    #         opponent_node.game_state.players[opponent_node.id].GetAvailableMoves(opponent_node.game_state)
    #
    #     best_q = 0
    #     sum_q = 0
    #     for action in range(len(opponent_node.potential_moves)):
    #
    #         # Copy game state and execute potential move
    #         (mid, fid, tgrab) = opponent_node.potential_moves[action]
    #         temp_state = copy.deepcopy(opponent_node.game_state)
    #         temp_state.ExecuteMove(opponent_node.id, opponent_node.potential_moves[action])
    #
    #         # Create child node using reward policy
    #         child_node = QNode(new_state, opponent_node.id, parent=opponent_node,
    #                            reward=opponent_node.reward_policy(new_state),
    #                            depth=opponent_node.depth + 1)
    #
    #         # Assign tile used in action and calculate node features
    #         child_node.tgrab = tgrab
    #         child_node.f = [
    #             child_node.f1(),
    #             child_node.f2(),
    #             child_node.f3(),
    #             child_node.f4(),
    #             child_node.f5(),
    #             child_node.f6(),
    #         ]
    #
    #         reg = np.abs(np.sum(weights))
    #
    #         # calculate q value for child
    #         child_node.q = float(np.sum(np.multiply(weights[0:6], child_node.f))) / (beta * reg)
    #
    #         #os.write(0, b'Child q: %f\n' % child_node.q)
    #
    #         # self.children.append(child_node)
    #
    #         if child_node.q > best_q:
    #             best_q = child_node.q
    #
    #         sum_q += child_node.q
    #
    #
    #     #os.write(0, b'Potential Moves: %d\n' % len(opponent_node.potential_moves))
    #     #os.write(0, b'Best q: %f\n' % best_q)
    #     #os.write(0, b'Sum q:  %f\n' % sum_q)
    #
    #
    #     if sum_q <= 0:
    #         return 0
    #
    #     else:
    #         return 1 - best_q



## Class to run weight training and q calculation
class QLearn(object):

    def __init__(self, node, weights):
        self.root = node
        self.weights = weights

    # Run session and return index for best move
    def best_move(self, episodes, train, alpha, beta, gamma, eps):

        # If not training, use model to evaluate first layer nodes and return best result
        if not train:
            node = self.root
            node.expand(self.weights, beta)
            return node.get_best_action(), self.weights

        # If training, explore possible actions according to e-greed multi-armed bandit for desired episodes
        else:
            for i in range(episodes):

                # DEBUG
                # os.write(0, b'Episode: %d\n' % i)

                node = self.root

                temp_id = self.root.id

                # Explore until terminal node is found
                while not node.terminal:

                    # Expand node unless already expanded
                    if not node.expanded:
                        node.expand(self.weights, beta)

                    # Break if terminal
                    if node.terminal:
                        break

                    # Select best child node based on e-greedy
                    temp_node = node.children[self.e_greedy(node, eps)]

                    if temp_node.id == temp_id:
                        temp_node.id = temp_node.opponent_id
                        temp_node.opponent_id = temp_id
                        temp_id = temp_node.id


                    # Update weights
                    if (not node.root) and train:
                        self.update_weights(node, alpha, gamma)

                    # Progress to next node
                    node = temp_node


            # TEST: Rescore
            # os.write(0, b'Old Q: %f, child: %d\n' % (self.root.children[self.root.get_best_action()].q, self.root.get_best_action()))
            # self.root.rescore(self.weights, beta)
            # os.write(0, b'New Q: %f, child: %d\n' % (self.root.children[self.root.get_best_action()].q, self.root.get_best_action()))

            # Get best action based on q_value
            best_move = self.root.get_best_action_3()

            # os.write(0, b'Best q = %f, Reward = %f | Best AQ = %f, Best Reward = %f\n' % (self.root.children[best_move].q,
            #         self.root.children[best_move].reward, self.root.children[self.root.get_best_action_2()].q,
            #         self.root.children[self.root.get_best_action_2()].reward))

            # Return best move and updated weights
            return best_move, self.weights


    # Update weights during training
    def update_weights(self, node, alpha, gamma):

        # DEBUG
        # os.write(0, b'Reward: %f\n' % node.reward)
        measured = node.reward + gamma * node.children[node.get_best_action()].q
        error = measured - node.q

        # DEBUG
        # os.write(0, b'Error = %f\n' % error)

        for i in range(len(self.weights)):

            # Calculate error from predicted reward and measured reward


            # Update weights using a double discount based on the depth of the node making the update
            # if node.depth == 0:
            #     self.weights[i] += alpha * error * node.f[i]
            #
            # else:
            #     self.weights[i] += float(pow(alpha, float((node.depth + 1)) / 2.0) * error * node.f[i])

            change = alpha * error * node.f[i]

            # DEBUG
            # os.write(0, b'Change: %f\n' % change)
            self.weights[i] += change

            # Restart weight if negative
            if self.weights[i] < 0:
                self.weights[i] = 0.0001



    # E-Greedy multi-armed bandit
    def e_greedy(self, node, eps):

        val = np.random.rand()

        if val >= eps:
            return np.random.randint(0, len(node.children))
        else:

            return node.get_best_action()
