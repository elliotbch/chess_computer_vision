
#*-------------------------------------------Imports---------------------------------------------*#

import copy
import numpy as np
# from detect_move import *

#*------------------------------------------Functions--------------------------------------------*#

def ascending_recognition(row_diff, col_diff, pieces_number, side):
    """
    Determine the possible values of a piece that moved in an ascending way
    
    :param initial_game_state: the initial game state
    :param initial_indices: the indices of the initial position
    :param row_diff: the difference of row between the initial and new position
    :param col_diff: the difference of column between the initial and new position
    :param pieces_number: the pieces number dictionary
    
    :return possible_values: the possible values of the piece that moved
    """
    
    # Set the possible values as an empty array by default
    possible_values = np.array([], dtype='int16')

    # If the piece has moved in a L-shape motion (i.e. knight), then add value 3 or -3
    if (abs(row_diff) == 2 and abs(col_diff) == 1) or (abs(row_diff) == 1 and abs(col_diff) == 2):
        
        if pieces_number[side * 3][0] > 0:
            
            possible_values = np.append(possible_values, side * 3)
        
    # If the piece has moved in a diagonal motion
    if abs(row_diff) == abs(col_diff):
        
        # If the piece has moved more than one square (i.e. bishop or queen), then add values 4, 6 or -4, -6
        if abs(row_diff) > 1:
                
            if pieces_number[side * 4][0] > 0:
                
                possible_values = np.append(possible_values, side * 4)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)
                
        # If the piece has moved only one square (i.e. pawn, bishop, king or queen), then add values 1, 4, 5 or 6 and -1, -4, -5, -6
        elif abs(row_diff) == 1:
            
            if pieces_number[side * 1][0] > 0:
                
                possible_values = np.append(possible_values, side * 1)
                
            if pieces_number[side * 4][0] > 0:
                
                possible_values = np.append(possible_values, side * 4)
                
            if pieces_number[side * 5][0] > 0:
                
                possible_values = np.append(possible_values, side * 5)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)
                
    # If the piece has moved in an horizontal motion    
    if row_diff == 0:
        
        # If the piece has moved more than one square (i.e. rook or queen), then add values 2, 6 or -2, -6
        if abs(col_diff) > 1:
            
            if pieces_number[side * 2][0] > 0:
                
                possible_values = np.append(possible_values, side * 2)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)
                
        # If the piece has moved only one square (i.e. rook, king or queen), then add values 2, 5 or 6 and -2, -5, -6
        elif abs(col_diff) == 1:
            
            if pieces_number[side * 2][0] > 0:
                
                possible_values = np.append(possible_values, side * 2)
                
            if pieces_number[side * 5][0] > 0:
                
                possible_values = np.append(possible_values, side * 5)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)

    # If the piece has moved in a vertical motion
    if col_diff == 0:
        
        # If the piece has moved more than one square (i.e. pawn, rook or queen), then add values 1, 2, 6 or -1, -2, -6
        if abs(row_diff) > 1:
            
            if pieces_number[side * 1][0] > 0:
                
                possible_values = np.append(possible_values, side * 1)
                
            if pieces_number[side * 2][0] > 0:
                
                possible_values = np.append(possible_values, side * 2)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)
                
        # If the piece has moved only one square (i.e. pawn, rook, king or queen), then add values 1, 2, 5 or 6 and -1, -2, -5, -6
        elif abs(row_diff) == 1:
            
            if pieces_number[side * 1][0] > 0:
                
                possible_values = np.append(possible_values, side * 1)
                
            if pieces_number[side * 2][0] > 0:
                
                possible_values = np.append(possible_values, side * 2)
                
            if pieces_number[side * 5][0] > 0:
                
                possible_values = np.append(possible_values, side * 5)
                
            if pieces_number[side * 6][0] > 0:
                
                possible_values = np.append(possible_values, side * 6)
            
    return possible_values

def context_update(game_state, game_dict, pieces_number, side):
    """
    Update the whole game state according to the context
    
    :param game_state: the game state
    :param game_dict: the game dictionary
    :param pieces_number: the pieces number dictionary
    :param side: the side of the piece
    :param new_indices: the indices of the new position
    
    :return updated_game_state: the new game state
    """
    
    # Set the updated game state as the game state by default
    updated_game_state = game_state
    
    # Set a flag to call the context update function again
    call_again = False
    
    # Set a flag to break the double loop
    break_flag = False
    
    # Loop through each piece of the game dictionary
    for piece_name in game_dict.keys():
        
        # Get the value of the piece
        piece_value = game_dict[piece_name][0]
        
        # If there is only one possible value
        if abs(piece_value) < 7:
            
            # Check if the piece is already in the unique pieces dictionary
            for piece in pieces_number.keys():
                if piece_name in pieces_number[piece]:
                    
                    # Break the double loop
                    break_flag = True
                    break
                
            if break_flag == True:
                
                break_flag = False
                continue
                
            # Get the piece values as a list
            piece_number_list = list(pieces_number[piece_value])
            
            # Get the index of the void value
            void_index = piece_number_list.index("")
            
            # Set the piece in the list
            piece_number_list[void_index] = piece_name
            
            # Update the pieces number dictionary
            pieces_number[piece_value] = piece_number_list
            
            # And update its value
            pieces_number[piece_value][0] -= 1
            
            # If the number of this piece is 0
            if pieces_number[piece_value][0] == 0:
                
                # Loop through the game dictionary
                for a_piece in game_dict.keys():
                    
                    # If the piece is different from the current piece and it contains multiple possible values
                    if a_piece != piece_name and abs(game_dict[a_piece][0]) > 7:
                        
                        # Get the possible values of the piece
                        possible_values = get_piece_possible_values(a_piece, game_dict)
                        
                        # Remove the piece from the possible values
                        possible_values = np.delete(possible_values, np.where(possible_values == piece_value))
                        
                        # If there is only one possible value
                        if len(possible_values) == 1:
                            
                            # Set a flag to call the context update function again
                            call_again = True
                            
                            # Update the game state
                            updated_game_state[game_dict[a_piece][1]] = possible_values[0]
                            
                        # Set the possible values of the piece
                        set_piece_possible_values(a_piece, game_dict, possible_values, game_dict[a_piece][1], game_dict[a_piece][2], side)
        
    # If all values in the pieces number dictionary are 0 except the values 1 and -1, then all unknown pieces are pawns
    unique_white_values = [2, 3, 4, 5, 6]
    unique_black_values = [-2, -3, -4, -5, -6]
    
    cpt = 0
    for a_unique_value in unique_white_values:
        
        if pieces_number[a_unique_value][0] == 0:
            
            cpt += 1
    
    if cpt == 5:
        
        updated_game_state = set_piece_all_pawns(updated_game_state, game_dict, pieces_number, 1)
    
    cpt = 0
    for a_unique_value in unique_black_values:
        
        if pieces_number[a_unique_value][0] == 0:
            
            cpt += 1
            
    if cpt == 5:
        
        updated_game_state = set_piece_all_pawns(updated_game_state, game_dict, pieces_number, -1)
        
    # If the flag is set to call the context update function again
    if call_again == True:
        
        # Call the context update function again
        updated_game_state = context_update(updated_game_state, game_dict, pieces_number, side)

    return updated_game_state

def decreasing_recognition(row_diff, col_diff, possible_values):
    """
    Determine the possible values of a piece that moved in a decreasing way
    
    :param row_diff: the difference of row between the initial and new position
    :param col_diff: the difference of column between the initial and new position
    :param possible_values: the possible values of the piece
    
    :return possible_values: the possible values of the piece that moved
    """
    
    # If the piece has moved in a diagonal motion
    if abs(row_diff) == abs(col_diff):
        
        # If the piece has moved more than one square (i.e. not a pawn, a rook or a king), then remove values 1, 2, 5 or -1, -2, -5
        if abs(row_diff) > 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 1))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 2))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 5))
                
        # If the piece has moved only one square (i.e. not a rook), then remove value 2 or -2
        elif abs(row_diff) == 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 2))
                
    # If the piece has moved in an horizontal motion    
    if row_diff == 0:
        
        # If the piece has moved more than one square (i.e. not a pawn, a bishop or a king), then remove values 1, 4, 5 or -1, -4, -5
        if abs(col_diff) > 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 1))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 4))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 5))
                
        # If the piece has moved only one square (i.e. not a pawn or a bishop), then remove values 1, 4 or -1, -4
        elif abs(col_diff) == 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 1))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 4))
                
    # If the piece has moved in a vertical motion
    if col_diff == 0:
        
        # If the piece has moved more than one square (i.e. not a a bishop or a king), then remove values 4, 5 or -4, -5
        if abs(row_diff) > 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 4))
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 5))
                
        # If the piece has moved only one square (i.e. not a bishop), then remove value 4 or -4
        elif abs(row_diff) == 1:
            
            possible_values = np.delete(possible_values, np.where(np.abs(possible_values) == 4))
        
    return possible_values

def get_indices(initial_game_state, new_game_state):
    """
    Guess the indices of the initial and new positions of the piece that moved
    
    :param initial_game_state: the initial game state
    :param new_game_state: the new game state
    
    :return initial_indices: the indices of the initial position
    :return new_indices: the indices of the new position
    :return row_diff: the difference of row between the initial and new position
    :return col_diff: the difference of column between the initial and new position
    """
    
    # Get the indices of the piece that moved
    indices = np.where(initial_game_state != new_game_state)
    
    # Get indice of the initial and new position of the piece that moved
    if new_game_state[indices[0][0], indices[1][0]] == 0:
        
        initial_indices = (indices[0][0], indices[1][0])
        new_indices = (indices[0][1], indices[1][1])
        
    elif new_game_state[indices[0][1], indices[1][1]] == 0:
        
        initial_indices = (indices[0][1], indices[1][1])
        new_indices = (indices[0][0], indices[1][0])
        
    # Determine the difference of row and column between the initial and new position
    row_diff = new_indices[0] - initial_indices[0]
    col_diff = new_indices[1] - initial_indices[1]
    
    return initial_indices, new_indices, row_diff, col_diff

def get_piece_name(initial_indices, game_dict):
    """
    Get the name of the piece that moved in the game dictionary
    
    :param initial_indices: the indices of the initial position
    :param game_dict: the game dictionary
    
    :return piece_name: the name of the piece
    """
    
    # Get the name of the piece that moved
    for piece_name in game_dict.keys():
        
        if game_dict[piece_name][1] == initial_indices:
            
            return piece_name
        
    return None

def get_piece_possible_values(piece_name, game_dict):
    """
    Get the possible values of a piece from a single int value
    
    :param piece_name: the name of the piece
    :param game_dict: the game dictionary
    
    :return possible_values: the possible values of the piece
    """
    
    # Get the possible values as a string
    possible_string = str(game_dict[piece_name][0])
    
    # If there is a minux sign, then remove it
    if possible_string[0] == "-":
        
        possible_string = possible_string[1:]
        
        # Get the possible values as negative values
        possible_values = np.array([-int(value) for value in possible_string])
        
    else:
        
        # Get the possible values
        possible_values = np.array([int(value) for value in possible_string])
        
    return possible_values

def modify_game_state(game_state, move):
    """
    Modify the game state according to the move
    
    :param game_state: the game state
    :param move: the move
    
    :return new_game_state: the new game state
    """
    
    new_game_state = copy.copy(np.flipud(game_state)) # A1 is 0,0
    
    if " -> " in move:
        
        init_pos = move.split(" -> ")[0]
        new_pos = move.split(" -> ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif " + " in move:
        
        init_pos = move.split(" + ")[0]
        new_pos = move.split(" + ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif " x " in move:
        
        init_pos = move.split(" x ")[0]
        new_pos = move.split(" x ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif "O-O-O" in move:
        
        if move[0] == "w":
            
            new_game_state[0, 2] = new_game_state[0, 4]
            new_game_state[0, 3] = new_game_state[0, 0]
            new_game_state[0, 0] = 0
            new_game_state[0, 4] = 0
            
        else:
            
            new_game_state[7, 2] = new_game_state[7, 4]
            new_game_state[7, 3] = new_game_state[7, 0]
            new_game_state[7, 0] = 0
            new_game_state[7, 4] = 0

    elif "O-O" in move:
        
        if move[0] == "w":
            
            new_game_state[0, 6] = new_game_state[0, 4]
            new_game_state[0, 5] = new_game_state[0, 7]
            new_game_state[0, 7] = 0
            new_game_state[0, 4] = 0
        else:
            
            new_game_state[7, 6] = new_game_state[7, 4]
            new_game_state[7, 5] = new_game_state[7, 7]
            new_game_state[7, 7] = 0
            new_game_state[7, 4] = 0
        
    return np.flipud(new_game_state)

def pieces_recognition(initial_game_state, new_game_state, game_dict, pieces_number, side):
    """
    Guess the value of a piece and update the game state accordingly

    :param initial_game_state: the initial game state
    :param new_game_state: the new game state
    :param game_dict: the game dictionary
    :param pieces_number: the pieces number dictionary
    :param side: the side of the piece
    
    :return updated_game_state: the new game state
    :return multi_pieces: a flag that is true if there are multiple pieces that have moved
    """
    
    # Set the updated game state as the new game state by default and the multi_pieces flag to False
    updated_game_state = new_game_state
    multi_pieces = False
    
    # Determine the number of pieces that have moved
    nb_pieces_moved = np.sum(initial_game_state != new_game_state)
    
    # If no piece has moved, return the updated game state
    if nb_pieces_moved == 0:
        
        return updated_game_state, False
    
    # If more than one piece has moved, call the rock recognition function
    elif nb_pieces_moved > 2:
        
        updated_game_state, multi_pieces = rock_recognition(initial_game_state, new_game_state, game_dict)
        
        # If the move was a rock
        if multi_pieces == False:
            
            # Update the whole game state according to chess rules
            updated_game_state = context_update(updated_game_state, game_dict, pieces_number, side)
            
            # Return the updated game state
            return updated_game_state, False
        
        # If the move was not a rock
        elif multi_pieces == True:
            
            # Return the initial game state
            return initial_game_state, True
        
    # If only one piece has moved
    else:
        
        # Get the indices of the initial and new positions of the piece that has moved
        initial_indices, new_indices, row_diff, col_diff = get_indices(initial_game_state, new_game_state)
        
        # Get the name of the piece
        piece_name = get_piece_name(initial_indices, game_dict)
            
        # Get the value of the piece
        piece_value = game_dict[piece_name][0]
            
        # If the piece has already been recognized
        if abs(int(piece_value)) < 7:
            
            # Update the game dictionary
            game_dict[piece_name] = (piece_value, new_indices, initial_indices, side)
            
            # Return the updated game state
            return updated_game_state, False
        
        # If the piece is unknown (i.e. its value is 7 or -7)
        elif abs(int(piece_value)) == 7:
            
            # Determine the possible values using the ascending recognition function
            possible_values = ascending_recognition(row_diff, col_diff, pieces_number, side)
            
            # If there is only one possible value
            if len(possible_values) == 1:
                
                # Update the game dictionary
                game_dict[piece_name] = (possible_values[0], new_indices, initial_indices, side)
                
                # Update the game state
                updated_game_state[new_indices] = possible_values[0]
                
                # Update the whole game state according to chess rules
                updated_game_state = context_update(updated_game_state, game_dict, pieces_number, side)
            
                # Return the updated game state
                return updated_game_state, False
            
        # If there are multiple possible values
        else:
            
            # Get the possible values using the translation function
            possible_values = get_piece_possible_values(piece_name, game_dict)
            
        # Once we get multiple possible values, we need to reduce the amount of possibilities using the decreasing recognition function
        possible_values = decreasing_recognition(row_diff, col_diff, possible_values)
        
        # If there is only one possible value
        if len(possible_values) == 1:
            
            # Update the game dictionary
            game_dict[piece_name] = (possible_values[0], new_indices, initial_indices, side)
            
            # Update the game state
            updated_game_state[new_indices] = possible_values[0]
            
            # Update the whole game state according to chess rules
            updated_game_state = context_update(updated_game_state, game_dict, pieces_number, side)
            
            # Return the updated game state
            return updated_game_state, False
        
        # If there are still multiple possible values
        else:
            
            # Update the game dictionary
            set_piece_possible_values(piece_name, game_dict, possible_values, new_indices, initial_indices, side)
            
            # Return the updated game state
            return updated_game_state, False
        
def rock_recognition(initial_game_state, new_game_state, game_dict):
    """
    Determine if the move was a rock and update the game state and the game dictionary accordingly
    
    :param initial_game_state: the initial game state
    :param new_game_state: the new game state
    :param game_dict: the game dictionary
    :param pieces_number: the pieces number dictionary
    :param side: the side of the piece
    
    :return updated_game_state: the new game state
    :return multi_pieces: a flag that is true if there are multiple pieces that have moved
    """
    
    white_side = [7, 7, 7, 7]
    black_side = [0, 0, 0, 0]
    little_rock = [4, 5, 6, 7]
    big_rock = [0, 1, 2, 3, 4]
    
    # By default, set the updated game state as the new game state
    updated_game_state = new_game_state
    
    # Get the row of the pieces that moved
    rows = np.where(initial_game_state != new_game_state)[0]
    
    # Get the column of the pieces that moved
    cols = np.where(initial_game_state != new_game_state)[1]
    
    # If the pieces that moved are not on the same row
    if len(set(rows)) != 1:
        
        # Signal that there are multiple pieces that have moved and it was not a rock
        return new_game_state, True
    
    # If white side
    if (rows == white_side).all():
        
        # If O-O
        if (cols == little_rock).all():
            
            # If 0's between the king and the rook
            if initial_game_state[7, 5] == 0 and initial_game_state[7, 6] == 0:
                
                # Get the piece names
                king_name = get_piece_name((7, 4), game_dict)
                rook_name = get_piece_name((7, 7), game_dict)
                
                # Update the game dictionary
                game_dict[king_name] = (5, (7, 6), (7, 4), 1)
                game_dict[rook_name] = (2, (7, 5), (7, 7), 1)
                
                # Update the game state
                updated_game_state[7, 6] = 5
                updated_game_state[7, 5] = 2
                
            else:
                
                # Signal that there are multiple pieces that have moved and it was not a rock
                return new_game_state, True
            
        # If O-O-O
        elif (cols == big_rock).all():
            
            # If 0's between the king and the rook
            if initial_game_state[7, 1] == 0 and initial_game_state[7, 2] == 0 and initial_game_state[7, 3] == 0:
                
                # Get the piece names
                king_name = get_piece_name((7, 4), game_dict)
                rook_name = get_piece_name((7, 0), game_dict)
                
                # Update the game dictionary
                game_dict[king_name] = (5, (7, 2), (7, 4), 1)
                game_dict[rook_name] = (2, (7, 3), (7, 0), 1)
                
                # Update the game state
                updated_game_state[7, 2] = 5
                updated_game_state[7, 3] = 2
                
            # Else, return the new game state
            else:
                
                # Signal that there are multiple pieces that have moved and it was not a rock
                return new_game_state, True
    
    # Black side
    elif (rows == black_side).all():
        
        # If O-O
        if (cols == little_rock).all():
            
            # If 0's between the king and the rook
            if initial_game_state[0, 5] == 0 and initial_game_state[0, 6] == 0:
                
                # Get the piece names
                king_name = get_piece_name((0, 4), game_dict)
                rook_name = get_piece_name((0, 7), game_dict)
                
                # Update the game dictionary
                game_dict[king_name] = (-5, (0, 6), (0, 4), -1)
                game_dict[rook_name] = (-2, (0, 5), (0, 7), -1)
                
                # Update the game state
                updated_game_state[0, 6] = -5
                updated_game_state[0, 5] = -2
                
            # Else, return the new game state
            else:
                
                # Signal that there are multiple pieces that have moved and it was not a rock
                return new_game_state, True
            
        # If O-O-O
        elif (cols == big_rock).all():
            
            # If 0's between the king and the rook
            if initial_game_state[0, 1] == 0 and initial_game_state[0, 2] == 0 and initial_game_state[0, 3] == 0:
                
                # Get the piece names
                king_name = get_piece_name((0, 4), game_dict)
                rook_name = get_piece_name((0, 0), game_dict)
                
                # Update the game dictionary
                game_dict[king_name] = (-5, (0, 2), (0, 4), -1)
                game_dict[rook_name] = (-2, (0, 3), (0, 0), -1)
                    
                # Update the game state
                updated_game_state[0, 2] = -5
                updated_game_state[0, 3] = -2
                    
            # Else, return the new game state
            else:
                    
                # Signal that there are multiple pieces that have moved and it was not a rock
                return new_game_state, True
            
        else:
            
            # Signal that there are multiple pieces that have moved and it was not a rock
            return new_game_state, True
            
    return updated_game_state, False

def set_piece_all_pawns(updated_game_state, game_dict, pieces_number, side):
    """
    Set all unknown pieces of a side as pawns
    
    :param updated_game_state: the updated game state
    :param game_dict: the game dictionary
    :param pieces_number: the pieces number dictionary
    :param side: the side of the piece
    
    :return updated_game_state: the updated game state
    """
    
    # Loop through the game dictionary
    for piece_name in game_dict.keys():
        
        # Get the value of the piece
        piece_value = game_dict[piece_name][0]
        
        # If the piece is unknown
        if piece_value == (side * 7):
            
            # Get the indices of the piece
            indices = game_dict[piece_name][1]
            
            # Update the game dictionary
            game_dict[piece_name] = (side * 1, indices, indices, side)
            
            # Update the pieces number dictionary
            piece_number_list = list(pieces_number[side * 1])
            void_index = piece_number_list.index("")
            piece_number_list[void_index] = piece_name
            pieces_number[side * 1] = piece_number_list
            pieces_number[side * 1][0] -= 1
            
            # Update the game state
            updated_game_state[indices] = side * 1
            
    return updated_game_state

def set_piece_possible_values(piece_name, game_dict, possible_values, new_indices, initial_indices, side):
    """
    Set the possible values of a piece in the game dictionary as a single int value
    
    :param piece_name: the name of the piece
    :param game_dict: the game dictionary
    :param possible_values: the possible values of the piece
    
    :return None
    """
    
    # Get the possible values as a string
    possible_string = "".join([str(value) for value in possible_values])
    
    # If there are several minus signs
    if possible_string.count("-") > 1:
        
        # Get the possible values as a positive value
        possible_int = int(possible_string.replace("-", "", possible_string.count("-")))
        
        # Set the possible value as a negative value
        possible_int = -possible_int
        
    else:
        
        # Get the possible values as a positive value
        possible_int = int(possible_string)
        
    # Update the game dictionary
    game_dict[piece_name] = (possible_int, new_indices, initial_indices, side)

def test_pieces_recognition():
    """
    Test the pieces_recognition function
    """
    
    # Define a game
    game = [
    "e2 -> e4",
    "c7 -> c5",
    "Ng1 -> Nf3",
    "e7 -> e6",
    "Nb1 -> Nc3",
    "Nb8 -> Nc6",
    "d2 -> d4",
    "c5 x d4",
    "Nf3 x Nd4",
    "a7 -> a6",
    "f2 -> f4",
    "Qd8 -> Qc7",
    "Bc1 -> Be3",
    "b7 -> b5",
    "Bf1 -> Bd3",
    "Bc8 -> Bb7",
    "Qd1 -> Qf3",
    "Ng8 -> Nf6",
    "Nd4 -> Nb3",
    "Nc6 -> Nb4",
    "a2 -> a3",
    "Nb4 x Nd3",
    "c2 x d3",
    "d7 -> d6",
    "Ra1 -> Rc1",
    "Qc7 -> Qd8",
    "f4 -> f5",
    "Bf8 -> Be7",
    "f5 x e6",
    "f7 x e6",
    "Nb3 -> Nd4",
    "Qd8 -> Qd7",
    "Qf3 -> Qh3",
    "Bb7 -> Bc8",
    "wO-O",
    "bO-O",
    "Rf1 -> Rf3",
    "Be7 -> Bd8",
    "Rc1 -> Rf1",
    "Bd8 -> Bb6",
    "Qh3 -> Qg3",
    "Qd7 -> Qe7",
    "Kg1 -> Kh1",
    "Bc8 -> Bb7",
    "Nc3 -> Ne2",
    "Bb6 x Bd4",
    "Ne2 x Nd4",
    "Rf8 -> Rf7",
    "Qg3 -> Qh3",
    "Ra8 -> Re8",
    "Be3 -> Bg5",
    "Bb7 -> Bc8",
    "Bg5 x Bf6",
    "Rf7 x Rf6",
    "Rf3 x Rf6",
    "g7 x f6",
    "Qh3 + Qg3",
    "Kg8 -> Kh8",
    "Nd4 -> Nc6",
    "Qe7 -> Qc7",
    "Qg3 -> Qh4",
    "Qc7 x Qc6",
    "Qh4 x Qf6",
    "Kh8 -> Kg8",
    "Qf6 + Qg5",
    "Kg8 -> Kh8",
    "Rf1 -> Rf7"
    ]

    # Define an initial game state
    initial_game_state = np.array([
        [-7, -7, -7, -7, -7, -7, -7, -7],
        [-7, -7, -7, -7, -7, -7, -7, -7],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 7,  7,  7,  7,  7,  7,  7,  7],
        [ 7,  7,  7,  7,  7,  7,  7,  7],
    ])
    
    # Define an initial game dictionary
    game_dict = {
        "piece1": (-7, (0, 0), (-1, -1), 1),
        "piece2": (-7, (0, 1), (-1, -1), 1),
        "piece3": (-7, (0, 2), (-1, -1), 1),
        "piece4": (-7, (0, 3), (-1, -1), 1),
        "piece5": (-7, (0, 4), (-1, -1), 1),
        "piece6": (-7, (0, 5), (-1, -1), 1),
        "piece7": (-7, (0, 6), (-1, -1), 1),
        "piece8": (-7, (0, 7), (-1, -1), 1),
        "piece9": (-7, (1, 0), (-1, -1), 1),
        "piece10": (-7, (1, 1), (-1, -1), 1),
        "piece11": (-7, (1, 2), (-1, -1), 1),
        "piece12": (-7, (1, 3), (-1, -1), 1),
        "piece13": (-7, (1, 4), (-1, -1), 1),
        "piece14": (-7, (1, 5), (-1, -1), 1),
        "piece15": (-7, (1, 6), (-1, -1), 1),
        "piece16": (-7, (1, 7), (-1, -1), 1),
        "piece17": (7, (6, 0), (-1, -1), 1),
        "piece18": (7, (6, 1), (-1, -1), 1),
        "piece19": (7, (6, 2), (-1, -1), 1),
        "piece20": (7, (6, 3), (-1, -1), 1),
        "piece21": (7, (6, 4), (-1, -1), 1),
        "piece22": (7, (6, 5), (-1, -1), 1),
        "piece23": (7, (6, 6), (-1, -1), 1),
        "piece24": (7, (6, 7), (-1, -1), 1),
        "piece25": (7, (7, 0), (-1, -1), 1),
        "piece26": (7, (7, 1), (-1, -1), 1),
        "piece27": (7, (7, 2), (-1, -1), 1),
        "piece28": (7, (7, 3), (-1, -1), 1),
        "piece29": (7, (7, 4), (-1, -1), 1),
        "piece30": (7, (7, 5), (-1, -1), 1),
        "piece31": (7, (7, 6), (-1, -1), 1),
        "piece32": (7, (7, 7), (-1, -1), 1)
    }
        
    # Define an initial side for the first piece
    side = 1
    
    # Define an initial pieces number dictionary following chess rules (1 white king, 8 black panws, ...)
    pieces_number = {
        1: (8, "", "", "", "", "", "", "", ""),
        2: (2, "", ""),
        3: (2, "", ""),
        4: (2, "", ""),
        5: (1, ""),
        6: (1, ""),
        -1: (8, "", "", "", "", "", "", "", ""),
        -2: (2, "", ""),
        -3: (2, "", ""),
        -4: (2, "", ""),
        -5: (1, ""),
        -6: (1, "")
    }
    
    # For each move, update the game state and test the pieces recognition
    for move in game:
        
        new_game_state = modify_game_state(initial_game_state, move)
        
        updated_game_state, _ = pieces_recognition(initial_game_state, new_game_state, game_dict, pieces_number, side)
        
        if side == 1:
            
            side = -1
            
        elif side == -1:
            
            side = 1
        
        initial_game_state = new_game_state

    print(updated_game_state)
        
    return

#*---------------------------------------------Main----------------------------------------------*#

if __name__ == "__main__":
    
    # Test the pieces_recognition function
    test_pieces_recognition()

#*---------------------------------------------EOF-----------------------------------------------*#