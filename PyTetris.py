
import random
from copy import deepcopy
import neat.checkpoint
import numpy as np
import neat
import matplotlib.pyplot as plt
import pickle

plt.ion()
generation = 0

class Tetris():
    
    def __init__(self):
        self.n_cols, self.n_rows = 10, 15+4 # add 4 buffer rows
        self.board = [[0 for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        self.is_alive = True
        
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
        for y_ind, y in enumerate(block):
            for x_ind, x in enumerate(y):
                board[top+y_ind][left+x_ind] += x
    
        return board
    
    def landing_params(self, board, block, left, top) -> list:
        # temporarily place the block
        board_temp = deepcopy(board)
        board_temp = self.place_block(board_temp, block, left, top)
        
        # max height difference between 2 adjacent blocks
        max_diff = 0
        prev_height = 0
        for col_index in range(self.n_cols):
            for row_index in range(self.n_rows+1):
                
                if row_index == self.n_rows or board[row_index][col_index] == 1:
                    cur_height = len(board) - row_index
                    
                    if col_index != 0:    
                        height_diff = abs(prev_height - cur_height)
                        max_diff = max(max_diff, height_diff)
                        
                    prev_height = cur_height
                    break
                    
        # number of holes
        holes = 0
        for col_index in range(self.n_cols):
            # first find topmost block, then screen down to get holes
            found_top = False
            for row_index in range(self.n_rows):
                if board[row_index][col_index] == 1:
                    found_top = True
                else: # if empty
                    if found_top == True: # and found topmost, means holes
                        holes += 1
        
        # max height of block and number of empty cols
        max_height = 0
        empty_cols = 0
        for col_index in range(self.n_cols):
            for row_index in range(self.n_rows):
                if board[row_index][col_index] == 1:
                    max_height = max(max_height, len(board) - row_index)
                    break
                if row_index == self.n_rows - 1 and board[row_index][col_index] == 0:
                    empty_cols += 1
        
        return [max_diff, holes, max_height, empty_cols]

    def check_overload(self):
        self.is_alive = True
        for row in range(4):
            if sum(self.board[row]) > 0: # if there's anything in the buffer layer
                self.is_alive = False
                
    def get_alive(self):
        return self.is_alive
                
    def get_data(self, block, left):
        # get landing params if dropping 'block' at 'left' position
        board = deepcopy(self.board)
        top = self.landing_position(board, block, left)
        return self.landing_params(board, block, left, top)
    
    def update_board(self):
        # update board and return score by line clear
        cleared_rows = 0
        for index, row in enumerate(self.board):
            if sum(row) == self.n_cols:
                del self.board[index]
                self.board = [0 for _ in range(self.n_cols)] + self.board
                cleared_rows += 1
        score_dict = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        score = score_dict[cleared_rows]
        return score
    
def run_tetris(genomes, config):
    
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
    global generation
    generation += 1
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
            6: 1
        }
        block_index = random.randint(0, 6)
        block = blockdict[block_index]
        
        # input data and get score from NEAT
        # iterate thru genomes
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive: # only proceed for alive tetris
                max_score = (block, 0, float('-inf')) # (block, left, output_score)
                
                # store rotation variants
                rotated_blocks = [block]
                for _ in range(rotations[block_index]):
                    rotated_blocks.append(np.rot90(rotated_blocks[-1]))
                # iterate thru rotation variants
                for rot_index, rot_block in enumerate(rotated_blocks):
                    # iterate left to right
                    for left in range(tetris.n_cols - len(rot_block[0])):
                        # get data for harddropping here, send to NEAT get output
                        landing_params = tetris.get_data(rot_block, left)
                        output_score = nets[index].activate(landing_params)[0]
                        max_score = (block, left, output_score) if output_score > max_score[1] else max_score
                        if max_score == float('-inf'): print('max_score error: ', output_score)
                
                # place block
                top = tetris.landing_position(tetris.board, max_score[0], max_score[1])
                tetris.board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                
                # update plot
                global graphs
                global axs
                graphs[index].remove()
                highlighted_board = tetris.place_block(tetris.board, max_score[0], max_score[1], top)
                graphs[index] = axs[index].imshow(highlighted_board)
                
                # kill if overload
                tetris.check_overload()
                
        # update board to clear lines, and update fitness
        remain_tetrises = 0
        for index, tetris in enumerate(tetrises):
            if tetris.get_alive():
                remain_tetrises += 1
                genomes[index][1].fitness += tetris.update_board()
        
        if remain_tetrises == 0:
            print('all dead')
            fig.suptitle('generation '+str(generation))
            plt.pause(3)
            break
            
if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # draw first plot
    fig, axs = plt.subplots(5,6, layout="compressed")
    axs = [j for i in axs for j in i]
    graphs = []
    fig.suptitle('initializing')
    for id, ax in enumerate(axs):
        graphs.append(ax.imshow([[0 for _ in range(10)] for _ in range(19)]))
        ax.set_axis_off()
    plt.pause(1)
    
    # Run NEAT
    best_genome = p.run(run_tetris, 2)
    # with open("winner.pkl", "wb") as f:
    #     pickle.dump(best_genome, f)
    #     f.close()
    

# import matplotlib.pyplot as plt
 
# tet = Tetris()
# block = [[1, 0, 0],
#          [1, 1, 1]]
# tet.board = [[0 for _ in range(tet.n_cols)] for _ in range(tet.n_rows)]
# top = tet.landing_position(tet.board, block, 0)
# print(top)
# tet.board = tet.place_block(tet.board, block, 0, top)
# block = [[1, 1, 0],
#          [0, 1, 1]]
# top = tet.landing_position(tet.board, block, 0)
# plt.imshow(tet.place_block(tet.board, block, 0, top))
# plt.show()
# print(tet.landing_params(tet.board, block, 0, top))
# plt.pause(1000)