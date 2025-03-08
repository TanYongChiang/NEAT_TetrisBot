import pyautogui
import cv2
import numpy as np
from PyTetris import Tetris
import pickle
import neat
import time

pyautogui.PAUSE = 0.03

class Read_Screen():
    
    def __init__(self):
        
        self.topleft = (817, 250)
        self.bottomright = (1156, 925)
        self.extratopleft_search = (817, 117)
        self.extrabottomright_search = (1156, 322)
        self.n_cols = 10
        self.n_rows = 20
        self.hold_topleft =  (627, 287) #(658, 286)
        self.hold_bottomright = (784, 367) #(805, 365)
        self.next_topleft = (1164, 285) #(1189, 283)
        self.next_bottomright = (1318, 374) #(1348, 374)
        self.blockdict = {
            0: [[0, 1, 0],
                [1, 1, 1]],
            
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
        
    def screenshot(self):
        topleft = self.topleft
        bottomright = self.bottomright
        width = bottomright[0]-topleft[0]
        length = bottomright[1]-topleft[1]
        playground = pyautogui.screenshot(region=(topleft[0], topleft[1], width, length))

        img = np.array(playground)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array_split(img, self.n_rows)
        row_centered = [row[len(row)//2] for row in img]
        img = [np.array_split(row, self.n_cols) for row in row_centered]
        col_centered = [[col[len(col)//2] for col in row] for row in img]
        img = [[0 if j < 50 else 1 for j in i] for i in col_centered]
        self.scrnsht = img
        
    def find_landed_blocks(self):

        landed_blocks = []

        i = -1 # iterate upwards
        while True:
            if sum(self.scrnsht[i]) == 0:
                break
            landed_blocks.append(self.scrnsht[i])
            i -= 1
            
        # pad new rows to allow space for dropping blocks
        for _ in range(19-len(landed_blocks)): 
            landed_blocks.append([0]*self.n_cols)
            
        landed_blocks = landed_blocks[::-1]
        self.bottom_index = i # topmost index
        return landed_blocks

    def find_dropping_blocks(self):
        # bottom: topmost index of landed blocks
        
        dropping_blocks = self.scrnsht[:self.bottom_index]
        if np.sum(dropping_blocks) != 4:
            # # block not fully shown
            # topleft, bottomright = self.extratopleft_search, self.extrabottomright_search
            # width, length = bottomright[0]-topleft[0], bottomright[1]-topleft[1]
            # img = pyautogui.screenshot(region=(topleft[0], bottomright[1], width, length))
            # id = self.identify_block_from_image_erosion(img)
            # if id == None: pass
            # else: return id
            return None

        def trim_zeros_vertically(matrix):
            return [i for i in matrix if sum(i) > 0]
        dropping_blocks = trim_zeros_vertically(dropping_blocks)
        dropping_blocks = trim_zeros_vertically(np.array(dropping_blocks).T)
        dropping_blocks = np.array(dropping_blocks).T.tolist()
        for index in self.blockdict:
            if dropping_blocks == self.blockdict[index]:
                return index
    
    def find_holding_blocks(self):
        # return block index, None if nothing
        topleft = self.hold_topleft
        bottomright = self.hold_bottomright
        width = bottomright[0]-topleft[0]
        length = bottomright[1]-topleft[1]
        img = pyautogui.screenshot(region=(topleft[0], topleft[1], width, length))
        return self.identify_block_from_image(img)
    
    def find_next_blocks(self):
        # return block index, None if nothing
        topleft = self.next_topleft
        bottomright = self.next_bottomright
        width = bottomright[0]-topleft[0]
        length = bottomright[1]-topleft[1]
        img = pyautogui.screenshot(region=(topleft[0], topleft[1], width, length))
        return self.identify_block_from_image(img)
    
    def trim_surround(self, matrix):
        vert = [i for i in matrix if sum(i) > 0]
        verthori = np.array([i for i in np.array(vert).T if sum(i) > 0]).T
        return verthori
    
    def downscale(self, img, rows, cols):
        img = np.array_split(img, rows)
        row_centered = [row[len(row)//2] for row in img]
        img = [np.array_split(row, cols) for row in row_centered]
        col_centered = [[col[len(col)//2] for col in row] for row in img]
        img = col_centered
        return img
    
    def identify_block_from_image(self, img):
        # used for holding blocks images, return block index
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = [[0 if j < 7 else 1 for j in i] for i in img]
        img = self.trim_surround(img)
        if img.size == 0:
            return None
        
        block_types = [
                [[0, 1, 0],
                [1, 1, 1]],
                
                [[0, 1, 1],
                [1, 1, 0]],
                
                [[1, 1, 0],
                [0, 1, 1]],
                
                [[1, 0, 0],
                [1, 1, 1]],
                
                [[0, 0, 1],
                [1, 1, 1]]
                ]
        ds_img = self.downscale(img, 2,3)
        for index, block in enumerate(block_types):
            if ds_img == block:
                return index
        
        if ds_img == [[1, 1, 1],
                      [1, 1, 1]]: # either cube or bar
            height, width = len(img), len(img[0])
            return 5 if abs(width/height-4) < abs(width/height-2) else 6
        
    def identify_block_from_image_erosion(self, img):
        # used for extra search dropping block
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = [[0 if j < 100 else 1 for j in i] for i in img]
        
        kernel = np.ones((1, 5), np.uint8)
        img = np.array(img).astype('uint8')
        img = cv2.erode(img, kernel)
        
        img = self.trim_surround(img)
        if img.size == 0:
            return None
        
        block_types = [
                [[0, 1, 0],
                [1, 1, 1]],
                
                [[0, 1, 1],
                [1, 1, 0]],
                
                [[1, 1, 0],
                [0, 1, 1]],
                
                [[1, 0, 0],
                [1, 1, 1]],
                
                [[0, 0, 1],
                [1, 1, 1]]
                ]
        ds_img = self.downscale(img, 2,3)
        for index, block in enumerate(block_types):
            if ds_img == block:
                return index
        
        if ds_img == [[1, 1, 1],
                      [1, 1, 1]]: # either cube or bar
            height, width = len(img), len(img[0])
            return 5 if abs(width/height-4) < abs(width/height-2) else 6
        
class Tetrio(Read_Screen):
    def __init__(self):
        super().__init__()
    
    def hold_block(self):
        pyautogui.press('c')
    
    def rotate_ccw(self):
        pyautogui.press('z')
            
    def move_left(self):
        pyautogui.press('left')
    
    def move_right(self):
        pyautogui.press('right')
    
    def hard_drop(self):
        pyautogui.press('space')
    
    def soft_drop(self):
        pyautogui.press('down')
    
    def generate_rotation_variants(self, block_index):
        rotations = {
                    0: 3,
                    1: 1,
                    2: 1,
                    3: 3,
                    4: 3,
                    5: 1,
                    6: 0
                }
        rotated_blocks = [self.blockdict[block_index]]
        for _ in range(rotations[block_index]):
            rotated_blocks.append(np.rot90(rotated_blocks[-1]))
        return rotated_blocks
    
    def get_rotation_left_displacement(self, block_index, rotation):
        # rotation 0 means no rotate, 1 means ccw once, return disp, positive to right
        displacement = {
            0:{
                0:0, 1:0, 2:0, 3:1
                },
            1:{
                0:0, 1:0
                },
            2:{
                0:0, 1:0
                },
            3:{
                0:0, 1:0, 2:0, 3:1
                },
            4:{
                0:0, 1:0, 2:0, 3:1
                },
            5:{
                0:0, 1:1
                },
            6:{
                0:0
                }
        }
        return displacement[block_index][rotation]
    
    def get_block_spawn_left_coor(self, block_index):
        left = {
            0: 3,
            
            1: 3,
            
            2: 3,
            
            3: 3,
            
            4: 3,
            
            5: 3,
            
            6: 4,
            }
        return left[block_index]
    
    def run_tetrio(self, genome, config):
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        tetris = Tetris()
        blocks_placed = 0
        
        while True:
            self.screenshot()
            tetris.board = self.find_landed_blocks()
            holding_blocks_id = self.find_holding_blocks()
            if holding_blocks_id == None:
                self.hold_block()
                print('hold block')
                curr_block = self.find_next_blocks()
                continue
            
            max_score = (0, 0, 0, float('-inf')) # (rotation, block, left, output_score)
            for rot_index, rot_block in enumerate(self.generate_rotation_variants(curr_block)): # get rotation variants
                for left in range(self.n_cols - len(rot_block[0]) + 1): # iterate left to right
                    landing_params = tetris.get_data(rot_block, left)
                    output_score = net.activate(landing_params)[0]
                    max_score = (rot_index, rot_block, left, output_score) if output_score > max_score[-1] else max_score

            # repeat for hold block
            hold_flag = False # swap blocks at the end if True
            if curr_block != holding_blocks_id:
                for rot_index, rot_block in enumerate(self.generate_rotation_variants(holding_blocks_id)):
                    for left in range(self.n_cols - len(rot_block[0]) + 1):
                        landing_params = tetris.get_data(rot_block, left)
                        output_score = net.activate(landing_params)[0]
                        
                        if output_score > max_score[-1]:
                            hold_flag = True
                            max_score = (rot_index, rot_block, left, output_score)
            
            # update tetris
            block_index = holding_blocks_id if hold_flag else curr_block
            top = tetris.landing_position(tetris.board, max_score[1], max_score[2])
            tetris.board = tetris.place_block(tetris.board, max_score[1], max_score[2], top)
            tetris.update_board()
            blocks_placed += 1
            
            # now keyboard work
            if hold_flag:
                self.hold_block()
                print('hold block')
            
            for i in range(max_score[0]):
                self.rotate_ccw()
            
            spawn_left_after_rotate = self.get_block_spawn_left_coor(block_index) + self.get_rotation_left_displacement(block_index, max_score[0])
            distance_to_move = spawn_left_after_rotate - max_score[2]
            if distance_to_move > 0:
                for _ in range(abs(distance_to_move)):
                    self.move_left()
            elif distance_to_move < 0:
                for _ in range(abs(distance_to_move)):
                    self.move_right()
            
            curr_block = self.find_next_blocks()
            
            self.hard_drop()
            print('place block', max_score[1], 'at left', max_score[2])
             
if __name__ == "__main__":
    
    with open('./winner-pickle', 'rb') as f:
        winner = pickle.load(f)
     
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    tetr = Tetrio()
    tetr.run_tetrio(winner, config)
    # print(tetr.get_block_spawn_left_coor(5) + tetr.get_rotation_left_displacement(5, 1))

