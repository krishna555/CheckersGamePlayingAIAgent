from board import Board
import constant
from playGame import PlayGame
import os
class CheckersOrchestrator:
    def __init__(self):
        self.b = Board()
        self.read_input_file()

    def read_input_file(self) -> None:
        self.b.read_board()

    def write_move(self, seq: str) -> None:
        ofp = open(os.path.join(os.getcwd(), 'output.txt'), "w")
        n = len(seq)
        for (i, s) in enumerate(seq):
            end_line = "\n" if i != n - 1 else ""
            ofp.write(s + end_line)
        ofp.close()

    def get_depth(self, rem_time: float) -> int:
        if self.b.get_game_mode() == constant.SINGLE:
            return 2
        if rem_time > 100:
            return 9
        elif rem_time > 50:
            return 7
        elif rem_time > 10:
            return 5
        else:
            return 2

    def run(self):
        p = PlayGame(self.b.get_board())
        p.set_max_player(self.b.get_playing_color())
        depth = self.get_depth(self.b.get_rem_time())
        p.set_depth(depth)
        p.set_piece_counts()
        v, jump_seq = p.alphabeta(None, None, False, 0, True, -constant.INFINITY, constant.INFINITY)
        self.write_move(jump_seq)

def main():
    s = CheckersOrchestrator()
    s.run()

if __name__ == "__main__":
    main()