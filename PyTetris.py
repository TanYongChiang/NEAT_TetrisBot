
import random
from copy import deepcopy
import neat.checkpoint
import numpy as np
import neat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from collections import defaultdict
import statistics

generation = 0
best_genome_id = 1 # for plotting purposes

class Tetris():
    
    def __init__(self):
        self.n_cols, self.n_rows = 10, 15+4 # add 4 buffer rows
        self.board = [[0 for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        self.is_alive = True
        self.lines_cleared = 0
        
    def landing_position(self, board, block, left):
        top = 0
        while True:
            # if reach end of block
            if top == len(board) - len(block) + 1:
                return top - 1
            
            # if overlapped, return top-1
            for y_ind, y in enumerate(block):
                for x_ind, x in enumerate(y):
                    if board[top+y_ind][left+x_ind] + x > 1: 
                        return top - 1
        
            top += 1
            
    def place_block(self, board, block, left, top):
        boardtemp = deepcopy(board)
        for y_ind, y in enumerate(block):
            for x_ind, x in enumerate(y):
                boardtemp[top+y_ind][left+x_ind] += x
    
        return boardtemp
    
    def landing_params(self, board, block, left, top) -> list:
        # temporarily place the block
        boardtemp = deepcopy(board)
        boardtemp = self.place_block(boardtemp, block, left, top)

        # calculate heights
        heights = []
        for col_index in range(self.n_cols):
            for row_index in range(self.n_rows + 1): # +1 buffer for empty cols
                if row_index == self.n_rows or boardtemp[row_index][col_index] == 1:
                    heights.append(len(boardtemp) - row_index)
                    break
        
        # height differences between adjacent blocks
        height_differences = []
        for i in range(self.n_cols - 1):
            height_differences.append(abs(heights[i]-heights[i+1]))

        # max height difference between 2 adjacent blocks
        max_diff = max(height_differences)
        
        # number of holes
        holes = 0
        for col_index in range(self.n_cols):
            # first find topmost block, then screen down to get holes
            found_top = False
            for row_index in range(self.n_rows):
                if boardtemp[row_index][col_index] == 1:
                    found_top = True
                else: # if empty
                    if found_top == True: # and found topmost, means holes
                        holes += 1
        
        # min, max height of block and number of empty cols
        max_height = max(heights)
        min_height = min(heights)
        empty_cols = len([i for i in heights if i == 0])
                    
        # holes when comparing rect with max_height to board
        rect_area = max_height * self.n_cols
        flooded_holes = rect_area - sum(map(sum, self.board))
        
        # average height
        average_height = statistics.mean(heights)
        
        # average height diff
        average_height_differences = statistics.mean(height_differences)
        
        # lines clearable on placement
        lines_clearable = 0
        for row in range(self.n_rows):
            if sum(self.board[row]) == self.n_cols:
                lines_clearable += 1
        
        return [max_diff, holes, max_height, empty_cols, flooded_holes, 
                average_height, average_height_differences, lines_clearable]

    def check_overload(self):
        if self.lines_cleared > 1000:
            self.is_alive = False
            return
        for row in range(4):
            if sum(self.board[row]) > 0: 
                # if there's anything in the buffer layer
                self.is_alive = False
                
    def get_alive(self):
        return self.is_alive
                
    def get_data(self, block, left):
        # get landing params if dropping 'block' at 'left' position
        top = self.landing_position(self.board, block, left)
        return self.landing_params(self.board, block, left, top)
    
    def update_board(self):
        # update board and return score by line clear
        cleared_rows = 0
        index = 0
        while index < len(self.board):
            while sum(self.board[index]) == self.n_cols:
                del self.board[index]
                self.board = [[0 for _ in range(self.n_cols)]] + self.board
                cleared_rows += 1
                self.lines_cleared += 1
            index += 1
        score_dict = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        score = score_dict[cleared_rows]
        return score
    
def run_tetris(genomes, config):
    
    def pad_block(block, layer): # for plot hold block
        for _ in range(layer):
            for i, __ in enumerate(block):
                block[i] = [0] + block[i] + [0] # pad horizontally
        for _ in range(layer):
            block = [[0]*len(block[0])] + block + [[0]*len(block[0])] # pad vertically
        return block
    
    # init NEAT
    nets = []
    tetrises = []
    
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        
        # init tetris
        tetrises.append(Tetris())
    
    # main loop
    global best_genome_id
    global generation
    generation += 1
    hold = None # block_index
    while True:
        # random a block
        blockdict = {
        0: [[1, 1, 1],
            [0, 1, 0]],
        
        1: [[0, 1, 1],
            [1, 1, 0]],
        
        2: [[1, 1, 0],
            [0, 1, 1]],
        
        3: [[1, 0, 0],
            [1, 1, 1]],
        
        4: [[0, 0, 1],
            [1, 1, 1]],
        
        5: [[1, 1, 1, 1]],
        
        6: [[1, 1],
            [1, 1]],
        }
        rotations = {
            0: 3,
            1: 1,
            2: 1,
            3: 3,
            4: 3,
            5: 1,
            6: 0
        }
        block_index = random.randint(0, 6)
        block = blockdict[block_index]
        if hold == None:
            hold = block_index
            continue
        
        # input data and get score from NEAT
        # iterate thru genomes
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive(): # only proceed for alive tetris
                max_score = (block, 0, float('-inf')) # (block, left, output_score)
                
                # store rotation variants
                rotated_blocks = [block]
                for _ in range(rotations[block_index]):
                    rotated_blocks.append(np.rot90(rotated_blocks[-1]))
                # iterate thru rotation variants
                for rot_index, rot_block in enumerate(rotated_blocks):
                    # iterate left to right
                    for left in range(tetris.n_cols - len(rot_block[0]) + 1):
                        # get data for harddropping here, send to NEAT get output
                        landing_params = tetris.get_data(rot_block, left)
                        output_score = nets[index].activate(landing_params)[0]
                        max_score = (rot_block, left, output_score) if output_score > max_score[-1] else max_score
                        
                # check also for hold block, same as above
                hold_flag = False # swap blocks at the end if True
                if block_index != hold:
                    rotated_blocks = [blockdict[hold]]
                    for _ in range(rotations[hold]):
                        rotated_blocks.append(np.rot90(rotated_blocks[-1]))
                    for rot_index, rot_block in enumerate(rotated_blocks):
                        for left in range(tetris.n_cols - len(rot_block[0]) + 1):
                            landing_params = tetris.get_data(rot_block, left)
                            output_score = nets[index].activate(landing_params)[0]
                            
                            if output_score > max_score[-1]:
                                hold_flag = True
                                max_score = (rot_block, left, output_score)
                    if hold_flag:
                        hold, block_index = block_index, hold # swap out block for hold
                
                # place block
                top = tetris.landing_position(tetris.board, max_score[0], max_score[1])
                tetris.board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                genomes[index][1].fitness += 1
                # kill if overload
                tetris.check_overload()
                
                # plot something
                if genomes[index][0] == best_genome_id:
                    global plot_board, plot_hold, ax0, ax1
                    plot_board.remove()
                    plot_hold.remove()
                    
                    highlighted_board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                    highlighted_board = tetris.place_block(highlighted_board, [[2]*10]*3, 0, 0) # hide buffer layer
                    plot_board = ax0.imshow(highlighted_board, cmap='gray', aspect='auto')
                    ax0.axes.set_title('genome id:' + str(genomes[index][0]) + 
                                        '\nlines_cleared:' + str(tetris.lines_cleared) +
                                        '\nblocks_placed:' + str(genomes[index][1].fitness))
                    plot_hold = ax1.imshow(pad_block(blockdict[hold], 2), cmap='gray', aspect='auto')
                    plot_hold.axes.set_title('hold piece')
                    
                    plt.pause(0.01)
                
        # update board to clear lines, and update fitness
        remain_tetrises = 0
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive():
                remain_tetrises += 1
                tetris.update_board() # genomes[index][1].fitness += tetris.update_board()
                
        if remain_tetrises == 0:
            print('all dead')
            max_fitness = (0, float('-inf'))
            for index, tetris in enumerate(tetrises):
                genomes[index][1].fitness += tetris.lines_cleared * 5
                max_fitness = (genomes[index][0], genomes[index][1].fitness) if genomes[index][1].fitness > max_fitness[1] else max_fitness
            
            best_genome_id = max_fitness[0]
            print('best genome id is', best_genome_id)
            break
            
# if __name__ == "__main__":
#     # Set configuration file
#     config_path = "./config-feedforward.txt"
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

#     # Create core evolution algorithm class
#     p = neat.Population(config)

#     # Add reporter for fancy statistical result
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
    
#     # draw first plot
#     outer = gridspec.GridSpec(1,2,width_ratios=[10,3])
#     gs0 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer[0])
#     gs1 = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=outer[1], height_ratios=[3,4,12])
#     ax0 = plt.subplot(gs0[0])
#     init_board = [[1 for _ in range(10)] for _ in range(4)] + [[0 for _ in range(10)] for _ in range(15)]
#     plot_board = ax0.imshow(init_board, cmap='gray', aspect='auto')
#     ax1 = plt.subplot(gs1[1])
#     plot_hold = ax1.imshow([[0 for _ in range(4)] for _ in range(4)], cmap='gray', aspect='auto')
#     ax0.set_axis_off(), ax1.set_axis_off()
#     plot_board.axes.set_title('board \ninitializing')
#     plot_hold.axes.set_title('hold piece')
#     plt.pause(1)
    
#     # Run NEAT
#     winner = p.run(run_tetris, None)
#     with open("winner-pickle", "wb") as f:
#         pickle.dump(winner, f)
    
# play once
if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # draw first plot
    outer = gridspec.GridSpec(1,2,width_ratios=[10,3])
    gs0 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=outer[0])
    gs1 = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=outer[1], height_ratios=[3,4,12])
    ax0 = plt.subplot(gs0[0])
    init_board = [[1 for _ in range(10)] for _ in range(4)] + [[0 for _ in range(10)] for _ in range(15)]
    plot_board = ax0.imshow(init_board, cmap='gray', aspect='auto')
    ax1 = plt.subplot(gs1[1])
    plot_hold = ax1.imshow([[0 for _ in range(4)] for _ in range(4)], cmap='gray', aspect='auto')
    ax0.set_axis_off(), ax1.set_axis_off()
    plot_board.axes.set_title('board \ninitializing')
    plot_hold.axes.set_title('hold piece')
    plt.pause(1)
    
    with open('./winner-pickle', 'rb') as f:
        winner = pickle.load(f)
    genomes = [(1, winner)]
    
    run_tetris(genomes, config)
    plt.pause(1000)

# # single-use plotting
# import matplotlib.pyplot as plt
# import matplotlib.ticker as plticker

# tet = Tetris()
# block = [[1,1,1,1,1]]
# top = tet.landing_position(tet.board, block, 0)
# tet.board = tet.place_block(tet.board, block, 0, top)
# for i in range(3):
#     block = [[1,1,1,1,1,1,1,1,1,1]]
#     top = tet.landing_position(tet.board, block, 0)
#     tet.board = tet.place_block(tet.board, block, 0, top)
# for i in range(3):
#     block = [[1,1,1]]
#     top = tet.landing_position(tet.board, block, 0)
#     tet.board = tet.place_block(tet.board, block, 0, top)
# block = [[1,1,1,1,1]]
# data = tet.get_data(block, 5)
# top = tet.landing_position(tet.board, block, 5)
# tet.board = tet.place_block(tet.board, block, 5, top)

# print(data)

# plot = plt.imshow(tet.board)
# loc = plticker.MultipleLocator(base=1.0)
# plot.axes.xaxis.set_major_locator(loc)
# plot.axes.yaxis.set_major_locator(loc)
# plt.show()
# tet.update_board()
# plt.imshow(tet.board)
# plt.show()