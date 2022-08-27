#
# raichu.py : Play the game of Raichu
#
# Contributors: Aashay Gondalia (@Aashay7), Harsh Atha (@atha333), Sai Hari Chandan Morapakala (@saihari)
#
# Based on skeleton code by D. Crandall, Oct 2021
#
#I have used the numpy library extensively in the code 
# and reffered its whenever needed from https://numpy.org/doc/stable/reference/index.html

import sys
import time
from typing import final
import copy
import math
import numpy as np
from scipy.spatial import distance

#Below Code Taken From https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
import warnings
warnings.filterwarnings("ignore")

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
    """
    generates moves for pichu, pikachu and raichu as if thery were at (0,0) given the size of the board
    Args:
        N ([int]): size of board

    Returns:
        [dict]: returns all mpossible moves if the piece was at (0,0)
    """
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
    """
    returns list of all the cells in between 2 points.

    Args:
        init ([int]): initial point
        final ([int]): final point

    Raises:
        ValueError: raises error describing the situation

    Returns:
        [list]: list of all points in between two points
    """
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
    """
    Calculates the next point that lies

    Args:
        init ([type]): [description]
        final ([type]): [description]
        N ([type]): [description]

    Returns:
        [type]: [description]
    """
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

#Below Function Taken From Assignment 0 Skeletal Code
def valid_index(pos, n, m):
        return 0 <= pos[0] < n  and 0 <= pos[1] < m

def move_piece(board, player, piece, initial, final, jumpFlag=False, jumpPosition = None):
    """
    Moves a piece on the board and modifies the cell values as required.
    """
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
    """
    Checks if a given move is valid according to game logic, if so makes the move
    """
    #if valid move between init and final  and final is empty then moves the piece and returns board
    in_between = all_cells_in_bw(init,final)
    content = [board[pt[0]][pt[1]] for pt in in_between]
    
    
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
    """Generates Pichu Successors given its current position"""
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
    """Generates Pikachu Successors given its current position"""
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
    """"Generates Raichu Successors given its current position"""
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
    """"Generate All Pichu Successors"""
    possible_boards = []
    pichu_locs=[(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i]==player]
    for each_pichu in pichu_locs:
        new_boards = pichu_move(board, player, each_pichu[0], each_pichu[1])
        
        if len(new_boards) == 0:
            continue
        possible_boards.extend(new_boards)
    
    return possible_boards

def pikachuMoves(board,player):
    """"Generate All pikachu Successors"""
    possible_boards = []
    pikachu_locs=[(row_i,col_i) for col_i in range(len(board[0])) for row_i in range(len(board)) if board[row_i][col_i]==player.upper()]
    for each_pikachu in pikachu_locs:
        new_boards = pikachu_move(board, player, each_pikachu[0], each_pikachu[1])
        
        if len(new_boards) == 0:
            continue
        possible_boards.extend(new_boards)
    
    return possible_boards
    
def raichuMoves(board,player):
    """"Generate All raichu Successors"""
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
    """Generates All Successors"""
    board = np.flip(board,0).tolist() if player == "b" else board 
    return pichuMoves(board,player) + pikachuMoves(board,player) + raichuMoves(board,player)
#### Utility Function and Supporting Functions

def getCount(board):
    """gets counts for all pieces and empty moves present on the board"""
    #For fast counting Implemented this Idea 
    #https://stackoverflow.com/questions/10741346/numpy-most-efficient-frequency-counts-for-unique-values-in-an-array
    unique, counts = np.unique(np.array(board), return_counts=True)
    return dict(zip(unique,counts))

def remaining(counts,player):
    """Calculates the sum of remaining opponent pieces"""
    remaining = 0
    player_info = {"w": "wW@","b":"bB$"}
    for each in player_info["w" if player=="b" else "b"]:
        if each in counts.keys():
            remaining+=counts[each]
    return remaining

def weightedTypeDiff(board,counts,main_player):
    """Weighted Difference between each type of piece b/w opponent and team"""
    weights = [1,2,3]
    
    
    #Pichus
    w_pichu = 0 if "w" not in counts.keys() else counts["w"]
    b_pichu = 0 if "b" not in counts.keys() else counts["b"]
    pichu =  w_pichu-b_pichu if main_player == "w" else b_pichu-w_pichu
    
    #pikachus
    w_pikachu = 0 if "W" not in counts.keys() else counts["W"]
    b_pikachu = 0 if "B" not in counts.keys() else counts["B"]
    pikachu =  w_pikachu-b_pikachu if main_player == "w" else b_pikachu-w_pikachu
    
    #raichus
    w_raichu = 0 if "@" not in counts.keys() else counts["@"]
    b_raichu = 0 if "$" not in counts.keys() else counts["$"]
    raichu =  w_raichu-b_raichu if main_player == "w" else b_raichu-w_raichu
    
    return weights[0]*pichu + weights[1]*pikachu + weights[2]*raichu

def isThreat(board, a,b,threat_piece,N):
    """ Checks if an opponents piece posesses a threat to the players piece """
    max_range_info = {
        "w":1,"b":1,"W":2,"B":2,"$":np.inf,"@":np.inf
    }
    
    angle_info = {
        "w":[45],"b":[45],"W":[90,0],"B":[90,0],"$":[0,90,45],"@":[0,90,45]
    }
    
    
    if (slopeAngle(a,b) in angle_info[threat_piece]) and (distance.euclidean(a , b) <= max_range_info[threat_piece]):
        next_point = nextPoint(b,a,N)
        if next_point != None:
            if board[next_point[0]][next_point[1]] == ".":
                content = [board[pt[0]][pt[1]] for pt in all_cells_in_bw(b,next_point)]
                if len(content) == 0:
                    return True
                elif content.count(".") == len(content) - 1:
                    return True
        
    
    return False

def distToRaichu(board,player):
    distance = 0
    state = np.array(board)
    player_info = {"w": "wW","b":"bB"}
    if player == "b":
        state = np.flip(state,0)
    for each in player_info[player]:
        for each_loc in np.argwhere(state == each):
            distance+= (len(board) - each_loc[0]) - 1
    return distance

#Argwhere function in np referenced from https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
def Threats(board, counts, player):
    state = np.array(board)
    player_info = {"w": "wW@","b":"bB$"}
    player_threat_info = {"b": "wW@","w":"bB$","W":"B$","B":"W@","@":"$","$":"@"}
    threat_weights = {"w":1,"W":2,"@":3,"b":1,"B":2,"$":3}
    threatSum = 0
    for each in player_info[player]:
        if each in counts.keys():
            for each_position in np.argwhere(state == each):
                for each_threat in player_threat_info[each]:
                    if each_threat in counts.keys():
                        for threat_position in np.argwhere(state == each_threat):
                            threatSum += threat_weights[each] if isThreat(board, each_position,threat_position,each_threat, len(board)) else 0

    return threatSum

def utility(board,main_player, inital_remaining_my_pieces,inital_remaining_enemies):
    counts = getCount(board)

    return (4*weightedTypeDiff(board,counts,main_player) + 
            2*(inital_remaining_enemies - remaining(counts,main_player)) +
            -1*(inital_remaining_my_pieces - remaining(getCount(board),"b" if main_player == "w" else "w")) + # (1.5* Threats(board,counts,"w" if main_player=="b" else "b")) + 
            -1*Threats(board,counts,main_player) - 1.5*distToRaichu(board,main_player))



###########End of Utility and support functions


def min_value(board, alpha, beta, d, max_depth, player, main_player, inital_remaining_my_pieces,inital_remaining_enemies):
    nextPlayer = "b" if player == "w" else "w"
    if d == max_depth:
        return utility(board,main_player, inital_remaining_my_pieces,inital_remaining_enemies)
    else:
        value = np.inf
        for a in actions(board,player):
            value = min(value, max_value(a, alpha, beta, d+1, max_depth, nextPlayer, main_player, inital_remaining_my_pieces,inital_remaining_enemies))
            if value <= alpha:
                return value
            beta = min(beta,value)
        return value

def max_value(board, alpha, beta, d, max_depth, player, main_player, inital_remaining_my_pieces,inital_remaining_enemies):
    nextPlayer = "b" if player == "w" else "w"
    if d == max_depth:
        return utility(board,main_player, inital_remaining_my_pieces,inital_remaining_enemies)
    else:
        value = -1*np.inf
        for a in actions(board,player):
            value = max(value, min_value(a, alpha, beta, d+1, max_depth, nextPlayer, main_player,inital_remaining_my_pieces,inital_remaining_enemies))
            if value >= beta:
                return value
            alpha = max(alpha,value) 
        return value

def mini_max(board, player, max_depth,inital_remaining_my_pieces,inital_remaining_enemies):
    successors = actions(board,player)
    values = [max_value(action, -1*np.inf, np.inf, 1, max_depth, player, player, inital_remaining_my_pieces,inital_remaining_enemies) for action in successors]
    max_value_index = np.argmax(values)
    return successors[max_value_index],np.max(values)

def find_best_move(board, N, player, timelimit):
    global ALL_MOVES
    # This sample code just returns the same board over and over again (which
    # isn't a valid move anyway.) Replace this with your code!
    ALL_MOVES = genMoves(N)
    board = string_to_board(board,N)
    inital_remaining_my_pieces = remaining(getCount(board),"b" if player == "w" else "w")
    inital_remaining_enemies = remaining(getCount(board),player)
    past_suggestions = []
    past_suggestion_values = []
    curr_depth = 0
    while True:
        curr_depth+=1
        suggestion, val = mini_max(board, player, curr_depth, inital_remaining_my_pieces,inital_remaining_enemies)
        past_suggestions.append(state_to_string(suggestion,N))
        past_suggestion_values.append(val)

        # print(val)
        # print(state_to_string(suggestion,N))
        yield past_suggestions[np.argmax(val)]



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
