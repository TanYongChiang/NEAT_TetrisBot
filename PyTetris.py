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
import time  # Using time to enforce a generation limit

generation = 0
best_genome_id = 1  # for plotting purposes

class Tetris():
    
    def __init__(self):
        self.n_cols, self.n_rows = 10, 15+4  # add 4 buffer rows
        self.board = [[0 for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        self.is_alive = True
        self.lines_cleared = 0
        self.combo_counter = 0
        self.combo_score = 0
        
    def landing_position(self, board, block, left):
        top = 0
        while True:
            if top == len(board) - len(block) + 1:
                return top - 1
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
    
    def normalize_features(self, features: list) -> list:
        # Define fixed minimum and maximum values for normalization:
        mins = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        maxs = [10, 50, 20, 20, 10, 200, 20, 10, 4]
        normalized = []
        for f, m, M in zip(features, mins, maxs):
            normalized.append((f - m) / (M - m) if M - m != 0 else f)
        return normalized

    def landing_params(self, board, block, left, top) -> list:
        boardtemp = deepcopy(board)
        boardtemp = self.place_block(boardtemp, block, left, top)
        heights = []
        for col_index in range(self.n_cols):
            for row_index in range(self.n_rows + 1):
                if row_index == self.n_rows or boardtemp[row_index][col_index] == 1:
                    heights.append(len(boardtemp) - row_index)
                    break
        height_differences = [abs(heights[i] - heights[i+1]) for i in range(self.n_cols - 1)]
        max_diff = max(height_differences)
        holes = 0
        for col_index in range(self.n_cols):
            found_top = False
            for row_index in range(self.n_rows):
                if boardtemp[row_index][col_index] == 1:
                    found_top = True
                else:
                    if found_top:
                        holes += 1
        max_height = max(heights)
        min_height = min(heights)
        empty_cols = len([i for i in heights if i == 0])
        rect_area = max_height * self.n_cols
        flooded_holes = rect_area - sum(map(sum, self.board))
        average_height = statistics.mean(heights)
        average_height_differences = statistics.mean(height_differences)
        lines_clearable = sum(1 for row in range(self.n_rows) if sum(self.board[row]) == self.n_cols)
        features = [max_diff, holes, max_height, min_height, empty_cols,
                    flooded_holes, average_height, average_height_differences, lines_clearable]
        return self.normalize_features(features)

    def check_overload(self):
        for row in range(4):
            if sum(self.board[row]) > 0:
                self.is_alive = False
                
    def get_alive(self): 
        return self.is_alive
                
    def get_data(self, block, left):
        top = self.landing_position(self.board, block, left)
        return self.landing_params(self.board, block, left, top)
    
    def update_board(self):
        cleared_rows = 0
        index = 0
        while index < len(self.board):
            if sum(self.board[index]) == self.n_cols:
                del self.board[index]
                self.board = [[0 for _ in range(self.n_cols)]] + self.board
                cleared_rows += 1
                self.lines_cleared += 1
            else:
                index += 1
        
        if cleared_rows > 0:
            self.combo_counter += 1
            if cleared_rows == 1:
                base_bonus = 0
            elif cleared_rows == 2:
                base_bonus = 1
            elif cleared_rows == 3:
                base_bonus = 2
            elif cleared_rows == 4:
                base_bonus = 4
            else:
                base_bonus = 0
            # Scale extra bonus with combo counter to emphasize combos:
            extra_bonus = cleared_rows * self.combo_counter
            self.combo_score += base_bonus + extra_bonus
        else:
            self.combo_counter = 0

def run_tetris(genomes, config):
    
    def pad_block(block, layer):
        for _ in range(layer):
            for i, __ in enumerate(block):
                block[i] = [0] + block[i] + [0]
        for _ in range(layer):
            block = [[0]*len(block[0])] + block + [[0]*len(block[0])]
        return block
    
    nets = []
    tetrises = []
    
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        tetrises.append(Tetris())
    
    global best_genome_id, generation
    generation += 1
    hold = None
    start_time = time.time()      # Record the start time of the generation
    max_time = 600                # 3 minutes = 180 seconds
    
    while True:
        # Check if time limit has been reached:
        if time.time() - start_time >= max_time:
            print("Time limit reached (3 mins), ending generation.")
            break
        
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
        rotations = {0: 3, 1: 1, 2: 1, 3: 3, 4: 3, 5: 1, 6: 0}
        block_index = random.randint(0, 6)
        block = blockdict[block_index]
        if hold is None:
            hold = block_index
            continue
        
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive():
                max_score = (block, 0, float('-inf'))
                rotated_blocks = [block]
                for _ in range(rotations[block_index]):
                    rotated_blocks.append(np.rot90(rotated_blocks[-1]))
                    
                for rot_block in rotated_blocks:
                    for left in range(tetris.n_cols - len(rot_block[0]) + 1):
                        landing_params = tetris.get_data(rot_block, left)
                        output_score = nets[index].activate(landing_params)[0]
                        if output_score > max_score[-1]:
                            max_score = (rot_block, left, output_score)
                            
                hold_flag = False
                if block_index != hold:
                    rotated_blocks = [blockdict[hold]]
                    for _ in range(rotations[hold]):
                        rotated_blocks.append(np.rot90(rotated_blocks[-1]))
                    for rot_block in rotated_blocks:
                        for left in range(tetris.n_cols - len(rot_block[0]) + 1):
                            landing_params = tetris.get_data(rot_block, left)
                            output_score = nets[index].activate(landing_params)[0]
                            if output_score > max_score[-1]:
                                hold_flag = True
                                max_score = (rot_block, left, output_score)
                    if hold_flag:
                        hold, block_index = block_index, hold
                
                top = tetris.landing_position(tetris.board, max_score[0], max_score[1])
                tetris.board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                genomes[index][1].fitness += 1  # Reward for each block placement
                tetris.check_overload()
                
                if genomes[index][0] == best_genome_id:
                    global plot_board, plot_hold, ax0, ax1
                    plot_board.remove()
                    plot_hold.remove()
                    
                    highlighted_board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                    highlighted_board = tetris.place_block(highlighted_board, [[2]*10]*3, 0, 0)
                    plot_board = ax0.imshow(highlighted_board, cmap='gray', aspect='auto')
                    ax0.axes.set_title('genome id:' + str(genomes[index][0]) +
                                         '\nlines_cleared:' + str(tetris.lines_cleared) +
                                         '\nblocks_placed:' + str(genomes[index][1].fitness))
                    plot_hold = ax1.imshow(pad_block(blockdict[hold], 2), cmap='gray', aspect='auto')
                    plot_hold.axes.set_title('hold piece')
                    
                    plt.pause(0.01)
        
        remain_tetrises = 0
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive():
                remain_tetrises += 1
                tetris.update_board()
                
        if remain_tetrises == 0:
            break
            
    # End-of-generation fitness adjustment
    COMBO_MULTIPLIER = 2  # Emphasize combos by scaling the combo score
    max_fitness = (0, float('-inf'))
    for index, tetris in enumerate(tetrises):
        remaining_board = (15*10) - sum(map(sum, tetris.board[4:]))
        genomes[index][1].fitness -= min(remaining_board * 0.5, genomes[index][1].fitness)
        genomes[index][1].fitness += tetris.combo_score * COMBO_MULTIPLIER
        genomes[index][1].fitness += tetris.lines_cleared * 0.5
        if genomes[index][1].fitness > max_fitness[1]:
            max_fitness = (genomes[index][0], genomes[index][1].fitness)
    
    best_genome_id = max_fitness[0]
    print('best genome id is', best_genome_id)
    
# if __name__ == "__main__":
#     config_path = "Neat_TetrisBot/config-feedforward.txt"
#     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
#     p = neat.Population(config)
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
    
#     outer = gridspec.GridSpec(1, 2, width_ratios=[10, 3])
#     gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
#     gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], height_ratios=[3, 4, 12])
#     ax0 = plt.subplot(gs0[0])
#     init_board = [[1 for _ in range(10)] for _ in range(4)] + [[0 for _ in range(10)] for _ in range(15)]
#     plot_board = ax0.imshow(init_board, cmap='gray', aspect='auto')
#     ax1 = plt.subplot(gs1[1])
#     plot_hold = ax1.imshow([[0 for _ in range(4)] for _ in range(4)], cmap='gray', aspect='auto')
#     ax0.set_axis_off(), ax1.set_axis_off()
#     plot_board.axes.set_title('board \ninitializing')
#     plot_hold.axes.set_title('hold piece')
#     plt.pause(1)
    
#     winner = p.run(run_tetris, 100 )
#     with open("winner-pickle", "wb") as f:
#         pickle.dump(winner, f)

# # play once
if __name__ == "__main__":
    # Set configuration file
    config_path = "NEAT_TetrisBot/config-feedforward.txt"
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