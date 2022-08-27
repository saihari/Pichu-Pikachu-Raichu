# Pichu, Pikachu \& Raichu: AI Game Playing Bot

## 1.1 Problem Statement

Raichu is a popular childhood game played on an n ×n grid (where n ≥ 8 is an even number) with three kinds of pieces (Pichus, Pikachus, and Raichus) of two different colors (black and white).

W: White Pikachu

w: White Pichu

@: White Raichu

B: Black Pikachu

b: Black Pichu

$: Black Raichu

Initial State of board is given as:

![image](https://media.github.iu.edu/user/18130/files/9050f680-3ddb-11ec-8347-26b2fed78d05)

Possible Moves by pieces:

A. Pichu: (1 empty square forward diagonally) or (2 squares forward diagonally if the landing spot is empty and Pichu has jumped over an enemy.) Like a bishop in Chess, but with limitations that it only moves forward, has limit on steps and can jump.

B. Pikachu: (1 or 2 empty squares forward, left or right) or (Jump over enemy piece, where landing spot is empty and only 1 enemy piece in jump. 2 to 3 moves). Like a Rook in Chess but with no backward movement, can jump and has limits on step.

C. Raichu: Created only when pichu/pikachu reach opposite row of the board. (any number of squares forward, backward, left, right, diagonal) or (jump over an enemy piece where in landing square is empty and there is only one enemy jumped over). Like a Queen in Chess, but can jump over 1 piece.

Jumping over an enemy piece removes the enemy piece from the board. Pichus can only defeat Pichus, Pikachus can defeat Pichus and Pikachus, Raichus can defeat all types of pieces.

The objective is to defeat the opponent. When there is no enemy piece left, player wins. Output is a string of the format of the board at the extent.

## 1.2 Approaches Used

### Computing successors and Validity

We have used the principles of coordinate geometry to fast compute the successors.
At first, we pre-compute movements of each piece assuming the piece was at (0,0). We translate the coordinates of each pre-computed value to generate final possible moves of the piece. Once we get list of all possible moves, we check for validity of the moves.

Additional check on top of this is if the given move is possible or not. To check if the move is possible or not, we abstract the game logic to a generalized pattern that is similar for all kinds of pieces. Drawback of this approach is the possibility of generating duplicate states in the successor set. Further work can be done on this to remove duplicates.

### 1.2.1 Minimax Algorithm

We started off with the most basic version of the Minimax Algorithm. Initially the utility function being used was the difference in number of pieces between player team and opponent team.
This approach worked pretty well till level 2 of the Minimax tree, but as we reach depth 3 and below the time consumed was too high and sometimes it failed to generate the better move in given timeout limit.

### 1.2.2 Minimax with Alpha-Beta Pruning

In this approach, we tried to reduce the number of nodes by using alpha-beta pruning to the already above applied Minimax algorithm. This process made the code a bit faster as compared to our previous approach.

### 1.2.3 Modified Minimax with Iterative deepening and Alpha-Beta Pruning:

We created a tweaked version of Minimax Algorithm that gave us a better successor as compared to above methods. In this version we loop over the infinite depth, till the timer stops us to go iteratively down the tree. In some test cases, as it went down the tree we realized that the best successor at a deeper level of the tree migt not be the best move. So we stored the move that had the best alpha value at each level and return the move with max value at end of each move timeout. This ensures that we get a best possible move explored in that limited time.

## 1.3 Utility Function

In order to compute the utility function, we have chosen various sub-heuristics and assigned weights to them based on their importance.
We have used the following functions:

    1. Difference in number of each type of piece between opponent and player team
    2. Threat to piece: Given a piece and left untouched the opponent would have a chance to kill it in the next turn.
    3. Number of kills our piece commits without losing our own piece.
    4. Distance between pichu/pikachu to become a Raichu.
    5. Number of player's pieces reduced.

Weights to each utility function: [4, -1, 2, -1.5, -1]

Using only weights would have been okay if each piece had the same value. As Raichu is able to move in all directions as well kill all pieces, it should have the highes weight. Pikachu will have the middle value and pichu has the least value.

Weights for Pieces:

1. Raichu:3
2. Pikachu:2
3. Pichu: 1

Using a heuristic that is a weighted sum of all above sub-heuristics, we compute the next possible state.

## References Used

[1] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.39.742&rep=rep1&type=pdf
