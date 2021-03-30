# IDEAS
# Input for NN: Needs to be 1 hot encoded
# Each set of 5 represents flag for colour of tile, then 4 sets to represent  what is in that factory
# e.g. [1,0,0,0,0,  0,1,0,0,0,  0,1,0,0,0,  0,0,1,0,0]
# = [Blue, Yellow, Yellow, red] present at that particular factory
# Using:
    # BLUE = 0
    # YELLOW = 1
    # RED = 2
    # BLACK = 3
    # WHITE = 4


# Output needs to equal the number of moves and represent Q(s,a), or the expected reward.



#### IDEA ####
# Could use statistics of tiles taken by each player, and what is left, and probability of completing rows, col, sets
##############

# Game Dynamics:

# Structure recording the number, type, and destination of tiles
# collected by a player. Note that the sum of 'num_to_pattern_line'
# and 'num_to_floor_line' must equal 'number'.

    # tgrab:
        # self.tile_type = -1 # <- this equals colour!!!
        # self.number = 0 # <- This is the total number of tiles of the above type taken from the factory / middle
        # self.pattern_line_dest = -1 # <- This is which pattern line the tiles will go to
        # self.num_to_pattern_line = 0 # <- This is how many of the picked up tiles will go to the board
        # self.num_to_floor_line = 0 # <- This is how many of the picked up tiles will go to the floor

    # fid:
        # Factory ID number
        # -1 if taken from centre

    # mid:
        # Move ID
        # Taken from centre or taken from factory



    # game_state
        # game_state.first_player_taken
        # game_state.factories
        # game_state.players[id].score
            # plr_state = self.game_state.players[i]
            # plr_state.EndOfGameScore()
            # player_traces[i] = (plr_state.score, plr_state.player_trace)
        # game_state.first_player
        # game_state.ExecuteMove(i, selected)
        # game_state.TilesRemaining()
        # game_state.ExecuteEndOfRound()
        # game_state.SetupNewRound()
        #       for plr in self.game_state.players:
        #       plr.player_trace.StartRound()
        # game_state.centre_pool.tiles[tile]


    # PlayerState
        # Variables
            # GRID_SIZE = 5
            # FLOOR_SCORES = [-1,-1,-2,-2,-2,-3,-3]
            # ROW_BONUS = 2
            # COL_BONUS = 7
            # SET_BONUS = 10

            # self.lines_number[line] += number
            # self.lines_tile[line] = tile_type
            # self.number_of[tile] <- returns number of this tile type on board

        # Methods
            # GetCompletedRows
            # GetCompletedColumns
            # GetCompletedSets
            # GetAvailableMoves <- Already in "moves"








##################################################################
## Try using Monte Carlo Tree Search Instead of Q-learning      ##
## 17/05/2020                                                   ##
##################################################################


# Basic Player that minimises tiles to floor
from advance_model import *
from utils import *
import numpy as np

###### MUST REMOVE LATER ######
import os
#sys.stdout


class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)


    def SelectMove(self, moves, game_state):

        best_move = None
        simulations = 20

        root = MCTSNode(game_state, self.id)
        search = MCTSeach(root)

        action = search.best_move(simulations)
        best_move = moves[action]
        return best_move











    # Only move across incomplete tiles from move that was used
    def getPotentialScore(self, player):
        used_tiles = []

        score_inc = 0
        gamma = 1

        # 1. Move tiles across from pattern lines to the wall grid
        for i in range(player.GRID_SIZE):
            # Is the pattern line full? If not it persists in its current
            # state into the next round.

            #if (player.lines_number[i] == i + 1 or i == pattern_line_dest):
            #if player.lines_number[i] == i + 1:

                #if i == pattern_line_dest and player.lines_number[i] != i + 1:
                #    gamma = score_multiplier

                tc = player.lines_tile[i]
                col = int(player.grid_scheme[i][tc])

                # Record that the player has placed a tile of type 'tc'
                # player.number_of[tc] += 1

                # Clear the pattern line, add all but one tile into the
                # used tiles bag. The last tile will be placed on the
                # players wall grid.
                for j in range(i):
                    used_tiles.append(tc)

                player.lines_tile[i] = -1
                player.lines_number[i] = 0

                # Tile will be placed at position (i,col) in grid
                player.grid_state[i][col] = 1

                # count the number of tiles in a continguous line
                # above, below, to the left and right of the placed tile.
                above = 0
                for j in range(col - 1, -1, -1):
                    val = player.grid_state[i][j]
                    above += val
                    if val == 0:
                        break
                below = 0
                for j in range(col + 1, player.GRID_SIZE, 1):
                    val = player.grid_state[i][j]
                    below += val
                    if val == 0:
                        break
                left = 0
                for j in range(i - 1, -1, -1):
                    val = player.grid_state[j][col]
                    left += val
                    if val == 0:
                        break
                right = 0
                for j in range(i + 1, player.GRID_SIZE, 1):
                    val = player.grid_state[j][col]
                    right += val
                    if val == 0:
                        break

                # If the tile sits in a contiguous vertical line of
                # tiles in the grid, it is worth 1*the number of tiles
                # in this line (including itself).
                if above > 0 or below > 0:
                    score_inc += gamma * (1 + above + below)

                # In addition to the vertical score, the tile is worth
                # an additional H points where H is the length of the
                # horizontal contiguous line in which it sits.
                if left > 0 or right > 0:
                    score_inc += gamma * (1 + left + right)

                # If the tile is not next to any already placed tiles
                # on the grid, it is worth 1 point.
                if above == 0 and below == 0 and left == 0 \
                        and right == 0:
                    score_inc += gamma * 1

                #score_inc = score_inc * (1 + 2*player.GetCompletedRows())
                #score_inc = score_inc * (1 + 2*player.GetCompletedColumns())
                #score_inc = score_inc * (1 + 2*player.GetCompletedSets())

        # Score penalties for tiles in floor line
        penalties = 0
        for i in range(len(player.floor)):
            penalties += player.floor[i] * player.FLOOR_SCORES[i] * 3
            player.floor[i] = 0

        used_tiles.extend(player.floor_tiles)
        player.floor_tiles = []

        # Players cannot be assigned a negative score in any round.
        score_change = score_inc + penalties
        change_in_score = score_change
        if score_change < 0 and player.score < -score_change:
            score_change = -player.score


        player.score += score_change
        player.player_trace.round_scores[-1] = score_change



        return (player.score, change_in_score, penalties)









class MCTSNode(object):
    def __init__(self, game_state, id, parent=None, reward = -9999.0, depth = 0):

        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.num_visits = 0.
        self.reward = reward
        self.potential_moves = None
        self.root = False
        self.terminal = False
        self.expanded = False
        self.id = id
        self.depth = depth

    # Method to expand new nodes
    def expand(self):
        
        
        
        self.potential_moves = self.game_state.players[self.id].GetAvailableMoves(self.game_state)
        if not self.potential_moves:
            # DEBUG
            # os.write(0, b'No potential_moves\n')
            self.terminal = True
            return

        for action in range(len(self.potential_moves)):
            new_state = copy.deepcopy(self.game_state)
            new_state.ExecuteMove(self.id, self.potential_moves[action])
            #(mid, fid, tgrab) = self.potential_moves[action]
            
            child_node = MCTSNode(game_state=new_state, id=self.id, parent=self,
                                  reward = self.reward_policy_3(self.potential_moves[action], new_state), depth=self.depth+1)
            self.children.append(child_node)

        self.expanded = True


    def reward_policy_1(self, move):

        (mid, fid, tgrab) = move
        reward = tgrab.num_to_pattern_line * 100 - tgrab.num_to_floor_line * 200
        return reward

    def reward_policy_2(self, move):
        alpha = 1
        (mid, fid, tgrab) = move
        reward = (tgrab.pattern_line_dest + 1) - \
                 (self.game_state.players[self.id].lines_number[tgrab.pattern_line_dest] -
                  tgrab.num_to_pattern_line) * 1.0

        return (reward - alpha*tgrab.num_to_floor_line) * 100
        # return (reward - 0.7*tgrab.num_to_floor_line*self.game_state.players[self.id].FLOOR_SCORES[tgrab.num_to_floor_line]) * 100

    def reward_policy_3(self, move, new_state):
        alpha = 0.5
        (mid, fid, tgrab) = move
        reward = (tgrab.pattern_line_dest + 1) - \
                 (self.game_state.players[self.id].lines_number[tgrab.pattern_line_dest] -
                  tgrab.num_to_pattern_line) * 1.0


        # DEBUG - TESTING GREEDY
        if new_state.players[self.id].lines_number[tgrab.pattern_line_dest] == (tgrab.pattern_line_dest + 1):
            # DEBUG
            # os.write(0, b'TEST\n')
            reward +=10

        return (reward - alpha * tgrab.num_to_floor_line * new_state.players[self.id].lines_number[-1])





        # node = root
        # while node.expanded:
        #     node = node.children[self.UCB(node)]
        #
        # if node.visited == 0:
        #     reward = self.simulate(node)
        # else:
        #     self.get_children(node)
        #     node.expanded = True
        #     reward = simulate(node.children[rand(len(node.children))])
        # node.reward = reward

    ## Might need to modify iteration for node ##
    # def ucb(self, cp):
    #     weights = np.zeros(len(self.children))
    #
    #     for i in range(len(self.children)):
    #         weights[i] = (self.children[i].reward / len(self.children[i])) + cp * \
    #                      np.sqrt((2*np.log(self.num_visits)) / self.children[i].visits)
    #
    #     return np.argmax(weights)


    def ucb(self, cp):
        weights = np.zeros(len(self.children))
        i = 0
        for c in self.children:

            # DEBUG
            #os.write(0, b'UCB: c.reward = %d, c.num_visits = %d, self.num_visits = %d\n' % (c.reward, c.num_visits, self.num_visits))

            if c.num_visits == 0:
                weights[i] = 999999
            else:
                weights[i] = (c.reward / c.num_visits) + cp * np.sqrt((2*np.log(self.num_visits)) / c.num_visits)
            # DEBUG
            #os.write(0, b'UCB: i = %d, w = %d\n' % (i, weights[i]))

            i += 1

        return np.argmax(weights)

    def ucb_best_action(self, cp):
        weights = np.zeros(len(self.children))
        i = 0
        for c in self.children:
            # Debug
            #os.write(0, b'UCB: c.reward = %d, c.num_visits = %d, self.num_visits = %d\n' % (c.reward, c.num_visits, self.num_visits))
            if c.num_visits == 0:
                weights[i] = 0
            else:
                # DEBUG - remove "test"
                test = (c.reward / c.num_visits) + cp * np.sqrt((2*np.log(self.num_visits)) / c.num_visits)
                weights[i] = test

                # DEBUG
                #os.write(0, b'UCB: i = %d, w = %d\n' % (i, test))
            i += 1

        return np.argmax(weights)

    def get_best_action(self):
        weights = np.zeros(len(self.children))
        i = 0
        for c in self.children:
            weights[i] = c.reward
            i+=1

        return np.argmax(weights)

    def simulate(self, y):

        temp_state = copy.deepcopy(self.game_state)
        possible_moves = temp_state.players[self.id].GetAvailableMoves(temp_state)
        steps = 0

        if possible_moves:
            while len(possible_moves) > 1:
                action = self.simulation_policy_4(possible_moves) ## simulation policy
                temp_state.ExecuteMove(self.id, possible_moves[action])
                possible_moves = temp_state.players[self.id].GetAvailableMoves(temp_state)
                steps += 1



        temp_player = myPlayer(self.id)

        score, reward, penalties =  temp_player.getPotentialScore(temp_state.players[self.id])
        score2, used_tiles = temp_state.players[self.id].ScoreRound()


        # score, reward = temp_state.players[self.id].getPotentialScore(temp_state.players[self.id])

        # DEBUG
        # os.write(0, b'Simulate: score = %d, reward = %d, scaled reward = %d, penalties = %d\n' % (score, reward, np.power(y,steps) * reward, penalties))
        # os.write(0, b'Simulate - Actual: score = %d\n' % (score2))

        if score2 > 0 and score2 > reward:
            return np.power(y,steps) * score2
        return np.power(y,steps) * reward
        # return np.power(y,steps) * score


    def simulation_policy_1(self, possible_moves):
        action = np.random.randint(len(possible_moves))

        return action

    def simulation_policy_2(self, possible_moves):
        most_to_line = -1
        corr_to_floor = 100000

        action = -1
        i = 0

        for mid, fid, tgrab in possible_moves:
            if most_to_line == 100000:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
                continue

            if tgrab.num_to_floor_line < corr_to_floor:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            elif tgrab.num_to_floor_line == corr_to_floor and \
                    tgrab.num_to_pattern_line > most_to_line:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            i+=1

        return action

    def simulation_policy_3(self, possible_moves):
        most_to_line = -1
        corr_to_floor = 0

        action = -1
        i = 0


        for mid, fid, tgrab in possible_moves:
            if most_to_line == -1:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
                continue

            if tgrab.num_to_pattern_line > most_to_line:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            elif tgrab.num_to_pattern_line == most_to_line and \
                    tgrab.num_to_pattern_line < corr_to_floor:
                action = i
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            i+=1

        return action

    def simulation_policy_4(self, possible_moves):
        most_to_line = -1
        corr_to_floor = 0

        action = -1
        i = 0
        most_point = -1

        # self.lines_number[line] += number
        # self.lines_tile[line] = tile_type



        for mid, fid, tgrab in possible_moves:

            points = (tgrab.pattern_line_dest + 1) - \
                     (self.game_state.players[self.id].lines_number[tgrab.pattern_line_dest] -
                      tgrab.num_to_pattern_line)
            if most_point == -1:
                action = i
                most_point = points
                corr_to_floor = tgrab.num_to_floor_line
                continue

            if points > most_point:
                action = i
                most_point = points
                corr_to_floor = tgrab.num_to_floor_line

            elif points == most_point and \
                    tgrab.num_to_pattern_line < corr_to_floor:
                action = i
                corr_to_floor = tgrab.num_to_floor_line
            i += 1

        return action

    # Pass reward up the tree, iff it is better than current reward
    def backpropagate(self, reward):
        self.num_visits += 1
        if self.reward < reward:
            self.reward = reward
        
        if self.parent:
            self.parent.backpropagate(reward)



    # # Need to rewrite for node!!!
    # def simulate(self, game_state):
    #     current_state = game_state
    #     possible_moves = current_state.players[self.id].GetAvailableMoves(current_state)
    #     steps = 0
    #     while len(possible_moves) > 0:
    #         action = np.random.randint(len(possible_moves))  ## simulation policy
    #         current_state.ExecuteMove(self.id, possible_moves[action])
    #         possible_moves = current_state.players[self.id].GetAvailableMoves(current_state)
    #         steps += 1
    #
    #     score, reward = self.getPotentialScore(current_state.players[self.id])
    #     self.num_visits += 1
    #     return score, steps
    #
    #     choose_child = argmax(q_children + 2*cp*sqrt(2*lg(parent.numvisits)/(child.numvisits)))
    #     q_children = total_reward / num_visits
    #     q_sim = y^steps * reward
    #     y = 0.8



class MCTSeach(object):

    def __init__(self, node):
        self.root = node

    # SIMPLIFY TO BE LIKE EXAMPLE
    def mcts_search(self, y, cp):
        node = self.root
        while node.expanded and not node.terminal:
            node = node.children[node.ucb(cp)]

        if node.num_visits == 0:
            reward = node.simulate(y)
        else:
            #node.potential_moves = node.game_state.players[node.id].GetAvailableMoves(node.game_state)
            node.expand()
            node.expanded = True

            ## FIX ##
            if not node.potential_moves:
                node.terminal = True
                node.reward = node.simulate(y)
                return node

            reward = node.children[np.random.randint(0, len(node.children))].simulate(y)
        node.reward = reward
        return node

    def mcts_search2(self, y, cp):
        node = self.root
        while not node.terminal:

            if not node.expanded:
                return node.expand()
            else:
                node = node.children[node.ucb(cp)]
        return node

    def mcts_search3(self, y, cp):
        node = self.root
        while node.expanded:
            tmp_node = node.children[node.ucb(cp)]
            if tmp_node.terminal:
                break
            else:
                node = tmp_node

        node.expand()
        return node




    def best_move(self, simulations):
        i = 0
        y = 0.7
        cp = 1.4


        while i < simulations:
            v = self.mcts_search3(y, cp)

            # DEBUG
            # os.write(0, b'V Potential Moves: %d\n' % len(v.potential_moves))


            # Check if all children of root are terminal if first child is terminal

            if self.root.children[0].terminal:
                terminal = 0
                for c in self.root.children:
                    if c.terminal:
                        terminal += 1
                # DEBUG
                # os.write(0, b'Terminal Nodes: %d\n' % terminal)
                # os.write(0, b'Number of Potential Moves: %d\n' % len(self.root.potential_moves))

                if terminal >= len(self.root.potential_moves):

                    # DEBUG
                    # os.write(0, b'Break: Terminal\n')
                    break


            if len(self.root.potential_moves) == 1:
                # DEBUG
                # os.write(0, b'Best Move: 0 - Trigger\n')
                # os.write(0, b'Number of Potential Moves: %d\n' % len(self.root.potential_moves))
                return 0
            v.reward += v.simulate(y)

            # DEBUG
            #os.write(0, b'Reward: %d, Depth: %d\n' % (v.reward, v.depth))
            #if not v.terminal:
            v.backpropagate(v.reward)
            i+=1

            # DEBUG
            #os.write(0, b'%d\n' % i)


        # j = 0
        # best_move = -500000
        # best_reward = 0
        # for j in range(len(self.root.children)):
        #     if best_reward < self.root.children[i].reward:
        #         best_move = j
        #         best_reward = self.root.children[i].reward
        #
        #     j += 1
        # return best_move

        # DEBUG
        #j = 0
        #os.write(0, b'Parent has reward %d\n' % self.root.reward)
        #for c in self.root.children:
        #    os.write(0, b'child %d has reward %d\n' % (j, c.reward))
        #    j+=1

        # best_move = self.root.ucb_best_action(cp)
        best_move = self.root.get_best_action()
        # os.write(0, b'ID: %d\n' % self.root.id)
        # os.write(0, b'Number of Children: %d\n' % len(self.root.children))
        # os.write(0, b'Number of Potential Moves: %d\n' % len(self.root.potential_moves))


        if len(self.root.potential_moves) < len(self.root.children):
            os.write(0, b'ERROR: Too many nodes\n')

        return best_move