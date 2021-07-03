import os
class Board:
    def __init__(self):
        self.board = []
        self.ROW_SIZE = 8
        self.COL_SIZE = 8
        self.base_path  = os.getcwd()
        self.game_mode = None
        self.playing_color = None
        self.rem_time = None

    def get_board(self):
        return self.board

    def get_game_mode(self):
        return self.game_mode
    
    def get_playing_color(self):
        return self.playing_color
    
    def get_rem_time(self):
        return self.rem_time

    def read_board(self):
        input_path = os.path.join(self.base_path, 'input.txt')
        with open(input_path, 'r') as f:
            self.game_mode = f.readline().rstrip('\n')
            self.playing_color = f.readline().rstrip('\n')
            self.rem_time = float(f.readline().rstrip('\n'))
            
            for i in range(self.ROW_SIZE):
                row = f.readline().rstrip('\n')
                self.board.append([])
                for ch in row:
                    self.board[i].append(ch)
