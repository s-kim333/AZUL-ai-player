# Written by Michelle Blom, 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

# Clone of naive player used for testing purposes



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

########################################################################################################################
######### Need to access available points some how!! Maybe through game_state and calculating? or inbuilt fn? ##########
########################################################################################################################


## NEED TO KNOW ##
# Current board status
# Current points
# Future move status
# Future move points



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

    # mid:
        # Move ID
        # Taken from centre or taken from factory



# Basic Player that minimises tiles to floor
from advance_model import *
from utils import *

class myPlayer(AdvancePlayer):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectMove(self, moves, game_state):
        # Select move that involves placing the most number of tiles
        # in a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 100000

        best_move = None


        for mid,fid,tgrab in moves:
            if most_to_line == 100000:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
                continue

            if tgrab.num_to_floor_line < corr_to_floor:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line
            elif tgrab.num_to_floor_line == corr_to_floor and \
                tgrab.num_to_pattern_line > most_to_line:
                best_move = (mid,fid,tgrab)
                most_to_line = tgrab.num_to_pattern_line
                corr_to_floor = tgrab.num_to_floor_line


        return best_move



