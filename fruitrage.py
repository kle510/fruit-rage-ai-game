import time
import math
import copy
import random
from collections import namedtuple


curr_state = namedtuple('curr_state', 'utility, board, turn')


class FruitRage:

    def __init__(self, board_dimensions, fruit_types, remaining_time):
        self.board_dimensions = board_dimensions
        self.fruit_types = fruit_types
        self.remaining_time = remaining_time
        curr_state(utility=0, board={}, turn='MAX')

    def retrieve_possible_moves(self, state):
        # list of possible moves
        possible_moves = []

        # get range of alphabetical letters for col
        alphabetical_col = []
        num_cols = len(state.board[0])
        for curr_col in range(num_cols):
            alphabetical_col.append(chr(curr_col + 65))

        # retrieve possible moves
        for curr_row, row in enumerate(state.board):
            for curr_col, col in enumerate(state.board):
                if state.board[curr_row][curr_col] != "*":
                    curr_move = alphabetical_col[curr_col] + str(curr_row + 1)
                    possible_moves.append(curr_move)

        return possible_moves

    def possible_moves(self, state):

        possible_moves = self.retrieve_possible_moves(state)
        # sort possible moves by greatest value
        #print (state.board)
        #print ("possible moves: ", possible_moves)
        potential_util = [self.outcome(
            state, move).utility for move in possible_moves]
        #print ("potential util: ", potential_util)

        if state.turn == 'MAX':
            sorted_moves = [x for _, x in sorted(
                zip(potential_util, possible_moves), key=lambda pair: pair[0], reverse=True)]

        else:
            sorted_moves = [x for _, x in sorted(
                zip(potential_util, possible_moves), key=lambda pair: pair[0])]

        #print ("sorted moves:", sorted_moves)

        # top x moves:
        top_x_moves = sorted_moves[:8]

        return top_x_moves

    def outcome(self, state, move):

        #print ("move picked: ", move )

        # convert move from alphabetical to row, col format
        row = int(move[1:]) - 1
        col = int(ord(move[0])-65)

        board = copy.deepcopy(state.board)

        fruit = board[row][col]

        # cover all squares
        def cover_all_squares(row, col, fruit):

            fruit_count = 0
            fruit_stack = [(row, col, fruit)]

            while len(fruit_stack) > 0:
                row, col, fruit = fruit_stack.pop()


                if row < 0 or col < 0 or row >= len(board) or col >= len(board[0]):
                    continue

                if board[row][col] == fruit:
                    fruit_count += 1
                    # replace fruit with *
                    board[row][col] = '*'

                    fruit_stack.append((row - 1, col, fruit))
                    fruit_stack.append((row + 1, col, fruit))
                    fruit_stack.append((row, col - 1, fruit))
                    fruit_stack.append((row, col + 1, fruit))

            return fruit_count

        number_of_fruits = cover_all_squares(row, col, fruit)
        # here, we should have number of fruits as well as an updated state with *'s in it

        score = number_of_fruits**2

        # gravity drop
        # for loop from bottom up
        for row, rows in enumerate((board)):
            for col, cols in enumerate(board[0]):
                if board[-row - 1][col] == '*':
                    # if not top row
                    if row != len(board) - 1:

                        if board[-row - 2][col] != '*':
                            # gravity drop
                            r = row
                            c = col
                            while board[-r - 1][c] == '*' and r >= 0:
                                board[-r - 1][c] = board[-r - 2][c]
                                board[-r - 2][c] = '*'
                                r -= 1

        # insert to list
        # account for different depths?

        # print(board)
        # print(score)

        return curr_state(utility=(state.utility+score if state.turn == 'MAX' else state.utility-score), board=board, turn=('MIN' if state.turn == 'MAX' else 'MAX'))

    def utility(self, state):
        return state.utility

    def terminal_test(self, state):
        # just check the bottom row only
        for col, last_row in enumerate(state.board):
            if state.board[-1][col] != '*':
                #print("terminal test is false")
                return False

        #print ("terminal test is true")
        return True



def mini_max(state, game):

    depth = 0

    def max_value(state, depth):
        if game.terminal_test(state) or depth == 3 or game.remaining_time < 15:
            return game.utility(state)
        v = -math.inf
        for move in game.possible_moves(state):
            v = max(v, min_value(game.outcome(state, move), depth + 1))

        return v

    def min_value(state, depth):
        if game.terminal_test(state) or depth == 3 or game.remaining_time < 15:
            return game.utility(state)
        v = math.inf
        for move in game.possible_moves(state):
            v = min(v, max_value(game.outcome(state, move), depth + 1))

        return v

    best_move = None
    best_state = None
    best_score = -math.inf

    # return the successor with max value
    # (so, take the max value of depth 1)
    for move in game.possible_moves(state):
        score = min_value(game.outcome(state, move), depth + 1)
        if score > best_score:
            best_move = move
            best_state = game.outcome(state, move)

    return [best_move, best_state]


def alpha_beta(state, game):

    alpha = -math.inf
    beta = math.inf
    depth = 0
    start_curr_search = time.time()

    def cutoff_test(state, depth):

        elapsed_time = time.time() - start_curr_search
        play_remaining_time = game.remaining_time - elapsed_time

        # hardcode depth limit

        if game.fruit_types < 3 or game.board_dimensions < 12:
            depth_limit = 4
            if play_remaining_time < 125:
                depth_limit = 3
            if play_remaining_time < 100:
                depth_limit = 2
            if play_remaining_time < 75:
                depth_limit = 1
            if play_remaining_time < 50:
                depth_limit = 0

        else:
            depth_limit = 0
        if play_remaining_time < 297:
            depth_limit = 4
        if play_remaining_time < 125:
            depth_limit = 3
        if play_remaining_time < 100:
            depth_limit = 2
        if play_remaining_time < 75:
            depth_limit = 1
        if play_remaining_time < 50:
            depth_limit = 0

        # if depth limit reached or empty board
        if depth > depth_limit or game.terminal_test(state):
            return True
        return False

    # run the utility test
    def eval(state):
        return game.utility(state)

    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval(state)
        for move in game.possible_moves(state):
            alpha = max(alpha, min_value(game.outcome(
                state, move), alpha, beta, depth + 1))
            if alpha >= beta:
                return beta
        return alpha

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval(state)
        for move in game.possible_moves(state):
            beta = min(beta, max_value(game.outcome(
                state, move), alpha, beta, depth + 1))
            if beta <= alpha:
                return alpha
        return beta

    best_move = None
    best_state = None

    # return the successor with max value
    # (so, take the max value of depth 1)
    for move in game.possible_moves(state):
        score = min_value(game.outcome(state, move), alpha, beta, depth + 1)
        if score > alpha:
            alpha = score
            best_move = move
            best_state = game.outcome(state, move)
        elif score == alpha and random.randint(0, 100) > 50 and game.board_dimensions > 10:
            alpha = score
            best_move = move
            best_state = game.outcome(state, move)

        if game.remaining_time < 12:
            return [best_move, best_state]

    return [best_move, best_state]



def read_file(in_file):
    input = open(in_file, "r")

    board_dimensions = int(input.readline().strip())
    fruit_types = int(input.readline().strip())
    remaining_time = float(input.readline().strip())

    board = []
    row_scanner = input.readline().strip()

    while row_scanner:
        curr_row = [[str(i) for i in row_scanner]]
        board += curr_row
        row_scanner = input.readline().strip()

    input.close()

    return (board_dimensions, fruit_types, remaining_time, board)


def write_file(best_move, best_state_board):
    output = open("output.txt", "w")

    # write best move (col letter, row number)
    output.write(best_move + "\n")

    # write best state
    for index, row in enumerate(best_state_board):
        if index == len(best_state_board) - 1:
            output.write(''.join(str(i) for i in row))
        else:
            output.write(''.join(str(i) for i in row) + "\n")

    output.close()


def write_play_file(board_dimensions, fruit_types, remaining_time, best_state_board):

    output = open("game.txt", "w")
    output.write(str(board_dimensions) + "\n")
    output.write(str(fruit_types) + "\n")
    output.write(str(remaining_time) + "\n")

    # write best state
    for index, row in enumerate(best_state_board):
        if index == len(best_state_board) - 1:
            output.write(''.join(str(i) for i in row))
        else:
            output.write(''.join(str(i) for i in row) + "\n")

    output.close()



def play_game():

    p1_score = 0
    p2_score = 0
    p2_remaining_time = 300
    p2_elapsed_time = 0
    in_file = "input.txt"
    p1_output = open("p1.txt", "w")
    p2_output = open("p2.txt", "w")
    board_dimensions, fruit_types, p1_remaining_time, board = read_file(
        in_file)

    p2_state = curr_state(utility=0, board=board, turn='MAX')
    # play the game
    while True:

        board_dimensions, fruit_types, p1_remaining_time, board = read_file(
            in_file)
        in_file = "game.txt"

        # player 1's turn
        p1_game = FruitRage(board_dimensions, fruit_types, p1_remaining_time)
        #p1_result = alpha_beta(curr_state(utility=0, board=p2_state.board, turn='MAX'), p1_game)
        p1_result = mini_max(curr_state(
            utility=0, board=p2_state.board, turn='MAX'), p1_game)
        p1_state = p1_result[1]
        p1_score += p1_state.utility
        print("P1 Score: ", p1_score)
        p1_output.write(str(p1_score) + "\n")

        if p1_game.terminal_test(p1_state):
            break

        curr_time = time.time()
        p1_elapsed_time = curr_time - start
        p1_remaining_time -= p1_elapsed_time - p2_elapsed_time
        print(p1_remaining_time, " seconds left for p1")
        write_play_file(board_dimensions, fruit_types,
                        p2_remaining_time, p1_state.board)

        # player 2's turn

        board_dimensions, fruit_types, p2_remaining_time, board = read_file(
            in_file)

        p2_game = FruitRage(board_dimensions, fruit_types, p2_remaining_time)
        #p2_result = mini_max(curr_state(utility=0, board=p1_state.board, turn='MAX'), p2_game)
        p2_result = alpha_beta(curr_state(
            utility=0, board=p1_state.board, turn='MAX'), p2_game)

        p2_state = p2_result[1]
        p2_score += p2_state.utility
        print("P2 Score: ", p2_score)
        p2_output.write(str(p2_score) + "\n")

        if p2_game.terminal_test(p2_state):
            break

        curr_time = time.time()
        p2_elapsed_time = curr_time - start
        p2_remaining_time -= p2_elapsed_time - p1_elapsed_time
        print(p2_remaining_time, " seconds left for p2")
        write_play_file(board_dimensions, fruit_types,
                        p1_remaining_time, p2_state.board)


    print("Final P1 Score: ", p1_score)
    print("Final P2 Score: ", p2_score)

    if p1_score > p2_score:
        print("P1 wins")
    else:
        print("P2 wins")

    p1_output.close()
    p2_output.close()


def main():

    in_file = "input.txt"
    board_dimensions, fruit_types, remaining_time, board = read_file(
        in_file="input.txt")

    state = curr_state(utility=0, board=board, turn='MAX')
    game = FruitRage(board_dimensions, fruit_types, remaining_time)

    # returns best move, best state
    solution = alpha_beta(state, game)
    best_move = solution[0]
    best_state_board = solution[1].board

    write_file(best_move, best_state_board)


if __name__ == '__main__':
    start = time.time()

    main()

    # play_game()

    #board_dimensions = 3
    #fruit_types = 2
    #remaining_time = 300
    #test = FruitRage(board_dimensions, fruit_types, remaining_time)
    #board = [['1', '1', '1'], ['0', '0', '0'], ['1', '0', '1']]
    #board = [['1', '1', '1'], ['1', '1', '1'], ['1', '1', '1']]
    #test_state = curr_state(utility=0, board=board, turn='MAX')
    # test.terminal_test(test_state)
    # test.possible_moves(test_state)
    #test.outcome(test_state, 'B3')

    end = time.time()
    print(end-start, " total seconds for entire program runtime")
