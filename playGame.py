import constant
from collections import defaultdict
import os
class PlayGame:
    def __init__(self, board):
        self.board = board
        self.player = None
        self.black_piece = 0
        self.white_piece = 0
        self.black_king = 0
        self.white_king = 0
        self.counter = {
            constant.BLACK_PIECE: 0,
            constant.BLACK_KING: 0,
            constant.WHITE_KING: 0,
            constant.WHITE_PIECE: 0
        }
        self.black_piece_list = set()
        self.white_piece_list = set()
        self.depth_limit = None

    def in_board(self, row: int, col: int) -> bool:
        return 0 <= row < constant.ROW_SIZE and 0 <= col < constant.COL_SIZE

    def set_depth(self, depth):
        self.depth_limit = depth

    def check_jump_allowed(self, row: int, col: int, op_row: int, op_col: int, blank_row: int, blank_col: int) -> bool:
        if (not self.in_board(op_row, op_col)) or (not self.in_board(blank_row, blank_col)):
            return False
        curr_piece = self.board[row][col]
        if self.board[op_row][op_col] in constant.OPPONENT_PIECES[curr_piece] and self.board[blank_row][blank_col] == constant.EMPTY_SQUARE:
            return True
        return False

    def is_jump_possible(self, row: int, col: int) -> bool:
        if self.board[row][col] == constant.EMPTY_SQUARE:
            return False

        curr_piece = self.board[row][col]
        for (dx, dy) in constant.JUMP_MOVES[curr_piece]:
            op_row = row + dx
            op_col = col + dy
            jump_to_row = row + 2 * dx
            jump_to_col = col + 2 * dy 
            if self.check_jump_allowed(row, col, op_row, op_col, jump_to_row, jump_to_col):
                return True
        return False

    def is_promotion_state(self, row: int, col: int) -> bool:
        curr_piece = self.board[row][col]
        if curr_piece == constant.BLACK_PIECE and row == 7:
            return True
        if curr_piece == constant.WHITE_PIECE and row == 0:
            return True
        return False

    def get_promoted_piece(self, piece: str) -> str:
        if piece == constant.BLACK_PIECE:
            return constant.BLACK_KING
        if piece == constant.WHITE_PIECE:
            return constant.WHITE_KING
        return ''
    
    def set_piece_counts(self) -> None:
        for i in range(constant.ROW_SIZE):
            for j in range(constant.COL_SIZE):
                if self.board[i][j] == constant.EMPTY_SQUARE:
                    continue
                if self.board[i][j] == constant.BLACK_PIECE or self.board[i][j] == constant.BLACK_KING:
                    self.black_piece_list.add((i, j))
                else:
                    self.white_piece_list.add((i, j))
                self.counter[self.board[i][j]] += 1

    def check_is_king_adjacent(self, row: int, col: int, king_piece: str) -> bool:
        if not self.in_board(row, col):
            return False
        return True if self.board[row][col] == king_piece else False

    def get_evaluation(self, piece_list: list, color: str) -> tuple:
        back_row = 0
        middle_box = 0
        middle_two_rows = 0
        protected_cnt = 0
        vulnerable_cnt = 0
        advanced_pawn_count = 0
        safe_piece_count = 0
        if color == constant.BLACK:
            for curr_piece in piece_list:
                (i, j) = curr_piece
                if i == 0:
                    back_row += 1
                if 3 <= i <= 4 and 3 <= j <= 4:
                    middle_box += 1
                if 3 <= i <= 4:
                    middle_two_rows += 1
                if i >= 4:
                    advanced_pawn_count += 1
                if j == 0 or j == 7:
                    safe_piece_count += 1
                protection = [[-1, -1], [-1, 1]]
                p1_x, p1_y = i + protection[0][0], j + protection[0][1]
                p2_x, p2_y = i + protection[1][0], j + protection[1][1]
                if (p1_x, p1_y) in self.black_piece_list or (p2_x, p2_y) in self.black_piece_list:
                    protected_cnt += 1
                attackers = [[1, -1], [1, 1]]
                a1_x, a1_y = i + attackers[0][0], j + attackers[0][1]
                a2_x, a2_y = i + attackers[1][0], j + attackers[1][1]
                if (a1_x, a1_y) in self.white_piece_list or (a2_x, a2_y) in self.white_piece_list:
                    vulnerable_cnt += 1
                if self.check_is_king_adjacent(p1_x, p1_y, constant.WHITE_KING) or \
                    self.check_is_king_adjacent(p2_x, p2_y, constant.WHITE_KING):
                    vulnerable_cnt += 1
        else:
            for curr_piece in piece_list:
                (i, j) = curr_piece
                if i == 7:
                    back_row += 1
                if 3 <= i <= 4 and 3 <= j <= 4:
                    middle_box += 1
                if 3 <= i <= 4:
                    middle_two_rows += 1
                if i < 4:
                    advanced_pawn_count += 1
                if j == 0 or j == 7:
                    safe_piece_count += 1
                protection = [[1, -1], [1, 1]]
                p1_x, p1_y = i + protection[0][0], j + protection[0][1]
                p2_x, p2_y = i + protection[1][0], j + protection[1][1]
                if (p1_x, p1_y) in self.white_piece_list or (p2_x, p2_y) in self.white_piece_list:
                    protected_cnt += 1
                attackers = [[-1, -1], [-1, 1]]
                a1_x, a1_y = i + attackers[0][0], j + attackers[0][1]
                a2_x, a2_y = i + attackers[1][0], j + attackers[1][1]
                if (a1_x, a1_y) in self.black_piece_list or (a2_x, a2_y) in self.black_piece_list:
                    vulnerable_cnt += 1
                if self.check_is_king_adjacent(p1_x, p1_y, constant.BLACK_KING) or \
                    self.check_is_king_adjacent(p2_x, p2_y, constant.BLACK_KING):
                    vulnerable_cnt += 1

        return (back_row, middle_box, middle_two_rows, protected_cnt, vulnerable_cnt, advanced_pawn_count, safe_piece_count)

    def eval_core(self) -> float:
        black_regular_piece = self.counter[constant.BLACK_PIECE]
        black_king_piece = self.counter[constant.BLACK_KING]
        white_regular_piece = self.counter[constant.WHITE_PIECE]
        white_king_piece = self.counter[constant.WHITE_KING]
        black_piece_count = black_regular_piece + black_king_piece
        white_piece_count = white_regular_piece + white_king_piece

        (black_back_row, black_middle_box, black_middle_two_rows, black_protected_piece, black_vulnerable_piece, black_advanced_pawn_count, black_safe_cnt) = \
            self.get_evaluation(self.black_piece_list, constant.BLACK)
        (white_back_row, white_middle_box, white_middle_two_rows, white_protected_piece, white_vulnerable_piece, white_advanced_pawn_count, white_safe_cnt) = \
            self.get_evaluation(self.white_piece_list, constant.WHITE) 
        
        if black_piece_count > 0:
            black_aggr_advanced_pawn = (black_advanced_pawn_count / black_piece_count)
            black_aggr_safe_cnt = (black_safe_cnt / black_piece_count)
        else:
            black_aggr_advanced_pawn = 0
            black_aggr_safe_cnt = 0
        if white_piece_count > 0:
            white_aggr_advanced_pawn = (white_advanced_pawn_count / white_piece_count)
            white_aggr_safe_cnt = (white_safe_cnt / white_piece_count)
        else:
            white_aggr_advanced_pawn = 0
            white_aggr_safe_cnt = 0

        if self.get_max_player() == constant.BLACK:
            white_piece_eval = (5 * white_regular_piece) + (7.75 * white_king_piece) + (2 * white_back_row) + (1 * white_middle_two_rows) + (3 * white_middle_box) + (-3 * white_vulnerable_piece) + (3 * white_protected_piece) + (white_aggr_advanced_pawn * 3) + (white_aggr_safe_cnt * 2)
            black_piece_eval = (5 * black_regular_piece) + (7.75 * black_king_piece) + (2 * black_back_row) + (1 * black_middle_two_rows) + (3 * black_middle_box) + (-3 * black_vulnerable_piece) + (3 * black_protected_piece) + (black_aggr_advanced_pawn * 3) + (black_aggr_safe_cnt * 2)
        else:
            white_piece_eval = (5 * white_regular_piece) + (7.75 * white_king_piece) + (3 * white_back_row) + (1 * white_middle_two_rows) + (2 * white_middle_box) + (-3 * white_vulnerable_piece) + (3 * white_protected_piece) + (white_aggr_advanced_pawn * 4)
            black_piece_eval = (5 * black_regular_piece) + (7.75 * black_king_piece) + (3 * black_back_row) + (1 * black_middle_two_rows) + (2 * black_middle_box) + (-3 * black_vulnerable_piece) + (3 * black_protected_piece) + (black_aggr_advanced_pawn * 4)
        
        if self.get_max_player() == constant.WHITE:
            return white_piece_eval - black_piece_eval
        else:
            return black_piece_eval - white_piece_eval

    def evaluate_alphabeta(self) -> float:
        return self.eval_core()

    def is_winner_found(self) -> bool:
        if self.counter[constant.BLACK_PIECE] + self.counter[constant.BLACK_KING] <= 0 or \
            self.counter[constant.WHITE_PIECE] + self.counter[constant.WHITE_KING] <= 0:
            return True
        return False
    
    def get_board_position(self, row: int, col: int) -> str:
        return chr(97 + col)  + str(8 - row)

    def get_max_player(self) -> str:
        return self.player
    
    def set_max_player(self, player: str) -> None:
        self.player =  player

    def get_player(self, is_max_player: bool) -> str:
        if is_max_player:
            return self.get_max_player()
        else:
            return constant.PIECE_COLOR_INVERSE[self.get_max_player()]

    def get_jump_string(self, init_row: int, init_col: int, final_row: int, final_col: int) -> str:
        final_pos = self.get_board_position(final_row, final_col)
        start_pos = self.get_board_position(init_row, init_col)
        return "J " + start_pos + " " + final_pos

    def get_simple_move_string(self, init_row: int, init_col: int, final_row: int, final_col: int) -> str:
        final_pos = self.get_board_position(final_row, final_col)
        start_pos = self.get_board_position(init_row, init_col)
        return "E " + start_pos + " " + final_pos

    def jump_remove_piece_from_set(self, row: int, col: int, piece) -> None:
        if piece in constant.PIECES[constant.BLACK]:
            self.black_piece_list.discard((row, col))
        else:
            self.white_piece_list.discard((row, col))

    def jump_add_piece_in_set(self, row: int, col: int, piece) -> None:
        if piece in constant.PIECES[constant.BLACK]:
            self.black_piece_list.add((row, col))
        else:
            self.white_piece_list.add((row, col))

    def jump_update_set(self, init_row: int, init_col: int, op_row: int, op_col: int, final_row: int, final_col: int, \
        cp_piece: str, op_piece: str, is_removal: bool) -> None:
        if is_removal:
            self.jump_remove_piece_from_set(init_row, init_col, cp_piece)
            self.jump_remove_piece_from_set(op_row, op_col, op_piece)
            self.jump_add_piece_in_set(final_row, final_col, cp_piece)
        else:
            self.jump_add_piece_in_set(init_row, init_col, cp_piece)
            self.jump_add_piece_in_set(op_row, op_col, op_piece)
            self.jump_remove_piece_from_set(final_row, final_col, cp_piece)

    def simple_move_update_set(self, init_row: int, init_col: int, final_row: int, final_col: int, piece: str, is_removal: bool) -> None:
        if is_removal:
            self.jump_remove_piece_from_set(init_row, init_col, piece)
            self.jump_add_piece_in_set(final_row, final_col, piece)
        else:
            self.jump_remove_piece_from_set(final_row, final_col, piece)
            self.jump_add_piece_in_set(init_row, init_col, piece)

    def alphabeta(self, row: int, col: int, is_jump_pending: bool, depth: int, is_max_player: bool, alpha: float, beta: float) -> tuple:
        if (depth == self.depth_limit) or (self.is_winner_found()):
            return (self.evaluate_alphabeta(), [])

        if is_jump_pending:
            curr_piece = self.board[row][col]
            
            if is_max_player:
                best_eval = -constant.INFINITY
            else:
                best_eval = constant.INFINITY
            
            best_jump_seq = None
            is_more_jumps_possible = False
            for (dx, dy) in constant.JUMP_MOVES[curr_piece]:
                op_row = row + dx
                op_col = col + dy
                jump_to_row = row + 2 * dx
                jump_to_col = col + 2 * dy
                
                if self.check_jump_allowed(row, col, op_row, op_col, jump_to_row, jump_to_col):
                    is_more_jumps_possible = True
                    op_piece = self.board[op_row][op_col]
                    self.board[op_row][op_col] = constant.EMPTY_SQUARE
                    self.board[row][col] = constant.EMPTY_SQUARE
                    self.board[jump_to_row][jump_to_col] = curr_piece
                    self.counter[op_piece] -= 1
                    self.jump_update_set(row, col, op_row, op_col, jump_to_row, jump_to_col, curr_piece, op_piece, True)
                    
                    was_promoted = False
                    curr_jump = self.get_jump_string(row, col, jump_to_row, jump_to_col)
                    next_jump_seq = None
                    if self.is_promotion_state(jump_to_row, jump_to_col):
                        was_promoted = True
                        promoted_piece = self.get_promoted_piece(curr_piece)
                        self.counter[promoted_piece] += 1
                        self.counter[curr_piece] -= 1
                
                        self.board[jump_to_row][jump_to_col] = promoted_piece
                        (v, next_jump_seq) = self.alphabeta(None, None, False, depth + 1, not is_max_player, alpha, beta)
                    else:
                        # Call alphabeta while checking for next jumps inside it.
                        (v, next_jump_seq) = self.alphabeta(jump_to_row, jump_to_col, True, depth, is_max_player, alpha, beta)
                    if is_max_player:
                        if v >= best_eval:
                            best_eval = v
                            if was_promoted:
                                best_jump_seq = [curr_jump]
                            else:
                                best_jump_seq = [curr_jump] + next_jump_seq
                    else:
                        if v <= best_eval:
                            best_eval = v
                            if was_promoted:
                                best_jump_seq = [curr_jump]
                            else:
                                best_jump_seq = [curr_jump] + next_jump_seq

                    if was_promoted:
                        self.counter[curr_piece] += 1
                        self.counter[promoted_piece] -= 1
                        was_promoted = False
                    self.jump_update_set(row, col, op_row, op_col, jump_to_row, jump_to_col, curr_piece, op_piece, False)
                    self.counter[op_piece] += 1
                    self.board[op_row][op_col] = op_piece
                    self.board[jump_to_row][jump_to_col] = constant.EMPTY_SQUARE
                    self.board[row][col] = curr_piece

                    # Alpha Beta Pruning Logic. We have to unset first before we prune.
                    if is_max_player:
                        if best_eval >= beta:
                            return (best_eval, best_jump_seq)
                        alpha = max(alpha, best_eval)
                    else:
                        if best_eval <= alpha:
                            return (best_eval, best_jump_seq)
                        beta = min(beta, best_eval)
            if is_more_jumps_possible == False:
                (best_eval, next_jump_seq) = self.alphabeta(None, None, False, depth + 1, not is_max_player, alpha, beta)
                return (best_eval, [])
            return (best_eval, best_jump_seq)
        else:
                if is_max_player:
                    best_eval = -constant.INFINITY
                else:
                    best_eval = constant.INFINITY
                best_jump_seq = None
                any_jump_made = False
                backup_single_jump_pieces = []
                for i in range(constant.ROW_SIZE):
                    for j in range(constant.COL_SIZE):
                            if not (self.board[i][j] != constant.EMPTY_SQUARE and \
                                self.board[i][j] in constant.PIECES[self.get_player(is_max_player)]):
                                continue
                            backup_single_jump_pieces.append((i, j))
                            # Carry our Jump Moves
                            if self.is_jump_possible(i, j):
                                curr_piece = self.board[i][j]
                                for (dx, dy) in constant.JUMP_MOVES[curr_piece]:
                                    
                                    op_row = i + dx
                                    op_col = j + dy
                                    jump_to_row = i + 2 * dx
                                    jump_to_col = j + 2 * dy
                                    if self.check_jump_allowed(i, j, op_row, op_col, jump_to_row, jump_to_col):
                                        op_piece = self.board[op_row][op_col]
                                        self.board[op_row][op_col] = constant.EMPTY_SQUARE
                                        self.board[i][j] = constant.EMPTY_SQUARE
                                        self.board[jump_to_row][jump_to_col] = curr_piece
                                        self.counter[op_piece] -= 1
                                        self.jump_update_set(i, j, op_row, op_col, jump_to_row, jump_to_col, curr_piece, op_piece, True)

                                        any_jump_made = True
                                        was_promoted = False
                                        if self.is_promotion_state(jump_to_row, jump_to_col):
                                            was_promoted = True
                                            promoted_piece = self.get_promoted_piece(curr_piece)
                                            # Adjust counter for promoted piece
                                            self.counter[promoted_piece] += 1
                                            self.counter[curr_piece] -= 1

                                            self.board[jump_to_row][jump_to_col] = promoted_piece
                                            (v, next_jump_seq) = self.alphabeta(None, None, False, depth + 1, not is_max_player, alpha, beta)
                                        else:
                                            # Call alphabeta while checking for next jumps inside it.
                                            (v, next_jump_seq) = self.alphabeta(jump_to_row, jump_to_col, True, depth, is_max_player, alpha, beta)
                                        if is_max_player:
                                            if v >= best_eval:
                                                best_eval = v
                                                if was_promoted:
                                                    best_jump_seq = [self.get_jump_string(i, j, jump_to_row, jump_to_col)]
                                                else:
                                                    best_jump_seq = [self.get_jump_string(i, j, jump_to_row, jump_to_col)] + next_jump_seq
                                        else:
                                            if v <= best_eval:
                                                best_eval = v
                                                if was_promoted:
                                                    best_jump_seq = [self.get_jump_string(i, j, jump_to_row, jump_to_col)]
                                                else:
                                                    best_jump_seq = [self.get_jump_string(i, j, jump_to_row, jump_to_col)] + next_jump_seq
                                        if was_promoted:
                                            self.counter[promoted_piece] -= 1
                                            self.counter[curr_piece] += 1
                                        
                                        self.jump_update_set(i, j, op_row, op_col, jump_to_row, jump_to_col, curr_piece, op_piece, False)
                                        self.counter[op_piece] += 1
                                        self.board[op_row][op_col] = op_piece
                                        self.board[jump_to_row][jump_to_col] = constant.EMPTY_SQUARE
                                        self.board[i][j] = curr_piece
                                        # Alpha Beta Pruning Logic. We have to unset first before we prune.
                                        if is_max_player:
                                            if best_eval >= beta:
                                                return (best_eval, best_jump_seq)
                                            alpha = max(alpha, best_eval)
                                        else:
                                            if best_eval <= alpha:
                                                return (best_eval, best_jump_seq)
                                            beta = min(beta, best_eval)
                was_simple_move_made = False
                if not any_jump_made:
                    num_pieces = len(backup_single_jump_pieces)
                    for ind in range(num_pieces):
                        (i, j) = backup_single_jump_pieces[ind]
                        curr_piece = self.board[i][j]
                        # Carry out simple moves
                        for (dx, dy) in constant.JUMP_MOVES[curr_piece]:
                            x = i + dx
                            y = j + dy
                            if not self.in_board(x, y):
                                continue
                            
                            if self.board[x][y] == constant.EMPTY_SQUARE:
                                was_simple_move_made = True
                                curr_jump = self.get_simple_move_string(i, j, x, y)
                                self.board[x][y] = curr_piece
                                self.board[i][j] = constant.EMPTY_SQUARE
                                self.simple_move_update_set(i, j, x, y, curr_piece, True)

                                any_move_made = True
                                was_promoted = False
                                if self.is_promotion_state(x, y):
                                    was_promoted = True
                                    promoted_piece = self.get_promoted_piece(curr_piece)
                                    self.counter[curr_piece] -= 1
                                    self.counter[promoted_piece] += 1
                                    self.board[x][y] = promoted_piece

                                (v, next_jump_seq) = self.alphabeta(None, None, False, depth + 1, not is_max_player, alpha, beta)
                                if is_max_player:
                                    if v >= best_eval:
                                        best_eval = v
                                        best_jump_seq = [curr_jump]
                                else:
                                    if v <= best_eval:
                                        best_eval = v
                                        best_jump_seq = [curr_jump]
                                
                                if was_promoted:
                                    self.counter[curr_piece] += 1
                                    self.counter[promoted_piece] -= 1
                                    was_promoted = False

                                self.simple_move_update_set(i, j, x, y, curr_piece, False)
                                self.board[i][j] = curr_piece
                                self.board[x][y] = constant.EMPTY_SQUARE
                                # Alpha Beta Pruning Logic. We have to unset first before we prune.
                                if is_max_player:
                                    if best_eval >= beta:
                                        return (best_eval, best_jump_seq)
                                    alpha = max(alpha, best_eval)
                                else:
                                    if best_eval <= alpha:
                                        return (best_eval, best_jump_seq)
                                    beta = min(beta, best_eval)
                if not any_jump_made and not was_simple_move_made:
                    if is_max_player:
                        return (-constant.INFINITY, [])
                    else:
                        return (constant.INFINITY, [])
                
                return (best_eval, best_jump_seq)