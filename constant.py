ROW_SIZE = 8
COL_SIZE = 8
EMPTY_SQUARE = '.'
WHITE_KING = 'W'
BLACK_KING = 'B'
WHITE_PIECE = 'w'
BLACK_PIECE = 'b'
WHITE = 'WHITE'
BLACK = 'BLACK'
SINGLE = 'SINGLE'
INFINITY = 1000000000

JUMP_MOVES = {
    'B': [(1, 1), (-1, 1), (1, -1), (-1, -1)],
    'W': [(1, 1), (-1, 1), (1, -1), (-1, -1)],
    'b': [(1, -1), (1, 1)],
    'w': [(-1, -1), (-1, 1)]
}
PIECES = {
    'WHITE': ['w', 'W'],
    'BLACK': ['b', 'B']
}
PIECE_COLOR_INVERSE = {
    'WHITE': 'BLACK',
    'BLACK': 'WHITE'
}
OPPONENT_PIECES = {
    'B': ['w', 'W'],
    'W': ['B', 'b'],
    'b': ['w', 'W'],
    'w': ['B', 'b']
}