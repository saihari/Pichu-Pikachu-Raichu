#
# raichu.py : Play the game of Raichu
#
# PLEASE PUT YOUR NAMES AND USER IDS HERE!
#
# Based on skeleton code by D. Crandall, Oct 2021
#
import sys
import time
from typing import final
import copy
import math
import numpy as np

ALL_MOVES = None
CAPABILITY = {
    "w": {
        "w" : "b",
        "W": "bB",
        "@": "bB$" 
    } ,
    "b": {
        "b" : "w",
        "B": "wW",
        "$": "wW@"
    }
}

def genMoves(N):
    row,col = 0,0
    fw = [(row+i,col) for i in range(1,N)]
    bk = [(row-1,col) for i in range(1,N)]

    left = [(row,col-i) for i in range(1,N)]
    right = [(row,col+i) for i in range(1,N)]

    diag_lr_fw = [(i,i) for i in range(1,N)]
    diag_lr_bk = [(-i,-i) for i in range(1,N)]
    diag_rl_bk = [(-i,i) for i in range(1,N)]
    diag_rl_fw = [(i,-i) for i in range(1,N)]

    return { "pichu" : diag_lr_fw[0:1] + diag_rl_fw[0:1],
             "pikachu" : fw[0:2] + right[0:2] + left[0:2],
             "raichu": fw[:] + right[:] + left[:] + bk[:] + diag_lr_bk[:] + diag_lr_fw[:] + diag_rl_bk[:] + diag_rl_fw[:] }

#My code Taken From Assignment 0
def slopeAngle(p1,p2):   
    """Returns the minimum angle the line connecting two points makes with the x-axis.
    Args:
        p1 ([tuple]): point p1
        p2 ([tuple]): point p2
    Returns:
        [float]: minimum angle between x-axis and line segment joining p1 and p2
    """
    try :
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
    except ZeroDivisionError:
        m = 0
    return abs(math.degrees(math.atan(m)))


#partially my code Taken From Assignment 0
def all_cells_in_bw(init,final):
    slope = slopeAngle(init,final)
      
    if slope in [0,90]:
        #points are in either same column or same row
        if init[0] == final[0]: #checking for same column
            points_between_them = [(final[0],i) for i in range(min(init[1],final[1])+1, max(init[1],final[1]))]

        elif init[1] == final[1]: #checking for same row
            points_between_them = [(i,final[1]) for i in range(min(init[0],final[0])+1, max(init[0],final[0]))]

        else: 
            raise ValueError("slope == 90 but no commanility between row and column; init = ("+", ".join(init)+")"+" final = ("+", ".join(final)+")"+" Slope: "+str(slope))

    elif slope == 45:
        #points are diagonal to each other
        m = (final[1]-init[1])/(final[0]-init[0])
        intercept = init[1] - init[0]*m

        x_points = [i for i in range(min(init[0],final[0])+1, max(init[0],final[0]))]
        points_between_them = [(x,int(intercept+m*x)) for x in x_points]

    else:
        raise("points are not in visible line of sight")
    
    return points_between_them

def nextPoint(init,final,N):
    m = slopeAngle(init,final) 
    if m in [0,90]:
        if final[1] == init[1]:
            point = ((int(final[0]+(final[0] - init[0])/abs((final[0] - init[0]))), final[1]))
        elif final[0] == init[0]:
            point = ((final[0], int(final[1]+(final[1] - init[1])/abs((final[1] - init[1]))) ))
    elif m in [45]:
        point = (((int(final[0]+(final[0] - init[0])/abs((final[0] - init[0])))), int(final[1]+(final[1] - init[1])/abs((final[1] - init[1]))) ))
    
    return point if valid_index(point,N,N) else None



def board_to_string(board, N):
    return "\n".join(board[i:i+N] for i in range(0, len(board), N))

def state_to_string(board, N):
    return "".join("".join(board[i]) for i in range(N))

def print_board(board,N):
    for i in range(N):
        print("".join(board[i]))

def string_to_board(board, N):
    number_board = []
    for i in range(N):
            number_board.append([p for p in board[N*i:N*(i+1)]]) 
    return number_board

#Below Function Taken From Assignment 0
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

def move_piece(board, player, piece, initial, final, jumpFlag=False, jumpPosition = None):
    # print("hjhgghj")
    # print(final)
    # print("ghghghghgh")

    # print("Moving Coin from init", initial , " to final", final )

    #alters the board
    assert ((jumpFlag) and (jumpPosition != None)) or ((not jumpFlag) and (jumpPosition == None)), "Jump position not passed but jump flag is on"

    #Checking for creation of Raichu
    if final[0] == len(board) - 1:
        if player == "w":
            if player in "wW":
                piece = "@"
        elif player == "b":
            if player in "bB":
                piece = "$"

    temp = copy.deepcopy(board)
    temp[initial[0]][initial[1]] = "."

    if jumpFlag:
        temp[jumpPosition[0]][jumpPosition[1]] = "."


    temp[final[0]][final[1]] = piece

    return temp if player == "w" else np.flip(temp,0).tolist()

def make_move(board,N,player,piece,init,final):
    #if valid move between init and final  and final is empty then moves the piece and returns board
    in_between = all_cells_in_bw(init,final)
    content = [board[pt[0]][pt[1]] for pt in in_between]
    
    # print("probable move from init", init , " to final", final )
    # if final == (1,4):
    #     print(init)
    #     print(final)
    #     print("in_bw",in_between)
    #     print("Con",content)
    #     print("board",(board[final[0]][final[1]] == "."))
    
    
    if (len(content) == 0) and  (board[final[0]][final[1]] == "."):
        #either no intersections and and final cell empty
        
        return move_piece(board=board, player=player, piece=piece, initial=init, final=final)

    elif (len(content) == 0) and (board[final[0]][final[1]] != "."):
        #no intersection but final cell is not empty 
        # check next cell in the same direction if empty and final cell content in capability then jump else immpossible move return None
        if board[final[0]][final[1]] in CAPABILITY[player][piece]:
            #check for next cell content
            destination = nextPoint(init,final,N) #position where the piece will be placed if vacant
            if destination and board[destination[0]][destination[1]] == ".":
                return move_piece(board=board, player=player, piece=piece, initial=init, final=destination, jumpFlag=True, jumpPosition = final)
            else:
                return None
        else:
            return None

    elif (len(content) != 0) and (board[final[0]][final[1]] != "."):
        #intersections present and final notcell  empty :
        #more thank one piece in path cannot jump 
        if (content.count(".") == len(content)) and (board[final[0]][final[1]] in CAPABILITY[player][piece]):
            destination = nextPoint(init,final,N) #position where the piece will be placed if vacant
            if destination and board[destination[0]][destination[1]] == ".":
                return move_piece(board=board, player=player, piece=piece, initial=init, final=destination, jumpFlag=True, jumpPosition = final)
            else:
                return None
        else:
            return None
    
    elif (len(content) != 0) and (board[final[0]][final[1]] == "."):
        #2 cases possible
        #one piece in intersection'
        if content.count(".") == len(content):
            return move_piece(board=board, player=player, piece=piece, initial=init, final=final)

        if content.count(".") == len(content)-1:
            #intersection piece in capability or not capability or same team piece 
            try:
                #handles if intersection piece is in capability 
                return move_piece(board=board, player=player, piece=piece, initial=init, final=final, jumpFlag=True, jumpPosition = in_between[[i for i,j in enumerate(content) if j != "." and j in CAPABILITY[player][piece]][0]])
            except IndexError:
                #either the intersection piece is out of capability or same team piece 
                return None 
                
        #more than one opposite piece in intersection or either one opposite piece and one same team piece  or 1 or more same team pieces implies no move
        else:
            return None


    elif (len(content) != 0) and (content.count(".") == len(content)) and (board[final[0]][final[1]] == "."):
        #there are intersections but all intersection empty and final cell is also empty
        return move_piece(board=board, player=player, piece=piece, initial=init, final=final)


    
    else:
        NotImplementedError("Case Missing in make_move")

def pichu_move(board,player,row,col):
    #listing single,double jump moves
    piece = player

    final_moves = []
    for move in ALL_MOVES["pichu"]:
        move = (move[0]+row,move[1]+col)
        
        #if invalid move skip
        if not valid_index(move, len(board), len(board[0])):
            continue
        
        else:
            config = make_move(board,N,player,piece,(row,col),move)

            if config != None:
                final_moves.append(config)

    return final_moves

def pikachu_move(board,player,row,col):
    #listing single,double jump moves
    piece = player.upper()

    final_moves = []
    for move in ALL_MOVES["pikachu"]:
        move = (move[0]+row,move[1]+col)
        
        #if invalid move skip
        if not valid_index(move, len(board), len(board[0])):
            continue
        
        else:
            config = make_move(board,N,player,piece,(row,col),move)

            if config != None:
                final_moves.append(config)

    return final_moves

def raichu_move(board,player, piece, row, col):
    #listing single,double jump moves

    final_moves = []
    for move in ALL_MOVES["raichu"]:
        move = (move[0]+row,move[1]+col)
        
        #if invalid move skip
        if not valid_index(move, len(board), len(board[0])):
            continue
        
        else:
            config = make_move(board,N,player,piece,(row,col),move)

            if config != None:
                final_moves.append(config)

    return final_moves


def pichuMoves(board,player):
    possible_boards = []
    pichu_locs=[(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i]==player]
    for each_pichu in pichu_locs:
        new_boards = pichu_move(board, player, each_pichu[0], each_pichu[1])
        
        if len(new_boards) == 0:
            continue
        possible_boards.extend(new_boards)
    
    return possible_boards

def pikachuMoves(board,player):
    possible_boards = []
    pikachu_locs=[(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i]==player.upper()]
    for each_pikachu in pikachu_locs:
        new_boards = pikachu_move(board, player, each_pikachu[0], each_pikachu[1])
        
        if len(new_boards) == 0:
            continue
        possible_boards.extend(new_boards)
    
    return possible_boards
    
def raichuMoves(board,player):
    piece = "@" if player == "w" else "$"


    possible_boards = []
    raichu_locs=[(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i]==piece]
    for each_raichu in raichu_locs:
        new_boards = raichu_move(board, player, piece, each_raichu[0], each_raichu[1])
        
        if len(new_boards) == 0:
            continue
        possible_boards.extend(new_boards)
    
    return possible_boards
    
def actions(board,player):
    board = np.flip(board,0).tolist() if player == "b" else board 
    return pichuMoves(board,player) + pikachuMoves(board,player) + raichuMoves(board,player)

def utility(board,main_player):
    white = sum([j in "wW@" for i in board for j in i])
    black = sum([j in "bB$" for i in board for j in i])
    # print(white,black)
    # if white-black == 3:
    #     print (white - black)
        # print("&&&&&&&&&&&&&&&&&")
        # print_board(board,len(board))
        # print("&&&&&&&&&&&&&&&&&")
    # return white-black   
    return white-black if main_player == "w" else black-white

def min_value(board, alpha, beta, d, max_depth, player, main_player):
    nextPlayer = "b" if player == "w" else "w"
    if d == max_depth:
        return utility(board,main_player)
    else:
        value = np.inf
        for a in actions(board,player):
            value = min(value, max_value(a, alpha, beta, d+1, max_depth, nextPlayer, main_player))
            if value <= alpha:
                return value
            beta = min(beta,value)
        return value

def max_value(board, alpha, beta, d, max_depth, player, main_player):
    nextPlayer = "b" if player == "w" else "w"
    if d == max_depth:
        return utility(board,main_player)
    else:
        value = -1*np.inf
        for a in actions(board,player):
            value = max(value, min_value(a, alpha, beta, d+1, max_depth, nextPlayer, main_player))
            if value >= beta:
                return value
            alpha = max(alpha,value) 
        return value

def mini_max(board, player, max_depth):
    successors = actions(board,player)
    values = [max_value(action, -1*np.inf, np.inf, 1, max_depth, player, player) for action in successors]
    max_value_index = np.argmax(values)
    return successors[max_value_index]
    
    # max_val = max(values)
    
    # for i,val in enumerate(values):
    #     if val == max_val:
    #         # print("$$$$$$$$$$$$$$$$$$$") 
    #         print_board(successors[i],N)
    #         # print("????????????????????")
    #         # print (val)
    #         print("$$$$$$$$$$$$$$$$$$$")
    # print(max_val)

def find_best_move(board, N, player, timelimit):
    global ALL_MOVES
    # This sample code just returns the same board over and over again (which
    # isn't a valid move anyway.) Replace this with your code!
    ALL_MOVES = genMoves(N)
    board = string_to_board(board,N)
    curr_depth = 0
    while True:
        curr_depth+=1
        yield state_to_string(mini_max(board,player, curr_depth),N)



if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")
        
    (_, N, player, board, timelimit) = sys.argv
    N=int(N)
    timelimit=int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N*N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print("Searching for best move for " + player + " from board state: \n" + board_to_string(board, N))
    print("Here's what I decided:")
    for new_board in find_best_move(board, N, player, timelimit):
        print(new_board)
