#*-----------------------------------------------------------------------------------------------*#
#*----------------------------------------Recognition.py-----------------------------------------*#
#*-----------------------------------------------------------------------------------------------*#

#*
#* @project ELEN0016-2 - Computer Vision - Student Project - Task 3: Piece Recognition
#* @authors Adrien Martins, Léa Rion and Elliot Bouchy
#* @team: group 9
#* @date 04/12/2024
#* @brief Piece recognition based on its motion
#*

#*-------------------------------------------Imports---------------------------------------------*#

from datetime import datetime

import cv2
import numpy as np
import os
from calib import *

from detect_piece import *
from piece_recognition import *

# Nouvelles importations
import json
from piece_recognition import get_indices, rock_recognition

#*------------------------------------------Functions--------------------------------------------*#

def pieces_initial(frame, corners, original_frame):
    """
    Determine the initial game_state from a frame

    :param frame: the frame to analyze
    :param rose_pos: position of the rose sticker
    :param blue_pos: position of the blue sticker
    :param corners: an array with the chessboard corner coordinates
    
    :return is_initialized: a flag to signal that the initialisation is done
    :return game_state: the initial game state
    """
    
    # Initialize return variables
    print("initializing...")
    is_initialized = False
    game_state = np.zeros((8, 8), dtype=int)

    game_state = detect_pieces(frame, game_state, corners, original_frame, is_initialized)
    print(game_state)

    is_initialized = True

    return is_initialized, game_state

def process_video(video_path, first_frame_to_analyze, output_states_file):
    """
    Process the video to perform a complete piece recognition

    :param video_path: the path to the video
    :param output_states_file: the file to save the game states to
    :param board_dimension: the dimension of the board.
    """
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        
        print(f"Error: Video file does not exist: {video_path}")
        return
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        
        print(f"Error: Unable to open video {video_path}")
        return
    
    # Set loop variables
    frame_number = 0
    is_initialized = False
    process_next_frame = False

    # Initialisation pour pieces_recognition
    side = 1
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
    
    # Structure de données pour stocker les mouvements et les états du jeu
    moves_data = {"moves": [], "game_states": []}
    
    # Loop through video
    while cap.isOpened():
        
        # Read frame
        ret_cap, frame = cap.read()
        original_frame = frame.copy()
        if not ret_cap:
            
            print("End of video or reading error")
            break
        
        # Process every 100 frame after the first frame to analyze or the next frame if the flag is set
        if (frame_number >= first_frame_to_analyze) and ((((frame_number+1) % 100) == 1) or (process_next_frame == True)):
            print("frame = ", frame_number)
            # Perform a calibration to get the chessboard corners and the stickers positions
            is_calibrate, rose_pos, blue_pos, corners = corners_calibration(frame)
            print("is_calibrate = ", is_calibrate)
            # If calibration is not successful
            if (is_calibrate == False):               
                # Process the next frame
                process_next_frame = True
                frame_number += 10
                continue
            
            # If it is the first frame to analyze
            if (frame_number == first_frame_to_analyze) or (is_initialized == False):
                
                # Perform the initialization of the game state
                is_initialized, game_state = pieces_initial(frame, corners, original_frame)
                print("is_initialized = ", is_initialized)

                if (frame_number == first_frame_to_analyze):
                    print("first frame to analyze")
                    game_state = np.rot90(game_state, k=-1)
                    game_state0 = game_state 

                    dict_pieces0 = create_32_pieces(game_state0)

                    # Ajouter l'état initial au fichier JSON
                    moves_data["game_states"].append({"frame": frame_number, "gs": game_state.tolist()})

                # cv2.waitKey(0)
            
            game_state = np.rot90(game_state, k=-1)
            dict_pieces = create_32_pieces(game_state)

            #### PIECE RECOGNITION ####
            if is_initialized and (frame_number != first_frame_to_analyze):
                print('piece recognition...')
                updates_game_state, move_notation = pieces_recognition(game_state0, game_state, dict_pieces0, pieces_number, side)
                game_state0 = game_state
                dict_pieces0 = dict_pieces

                # Détecter les mouvements spéciaux
                updated_game_state, special_move_notation = detect_special_moves(game_state0, game_state, dict_pieces0)
                if special_move_notation:
                    move_notation = special_move_notation

                # Ajouter le mouvement et l'état du jeu au fichier JSON
                moves_data["moves"].append(move_notation)
                moves_data["game_states"].append({"frame": frame_number, "gs": game_state.tolist()})

                print(updates_game_state)

            # Reset the next frame flag
            process_next_frame = False
    
        # Increment the frame count
        frame_number += 1
        is_initialized = False

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break
        
        # Display the frame
        cv2.imshow('Frame', frame)
    
    # Release video and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Enregistrer les mouvements et les états du jeu dans un fichier JSON
    with open(output_states_file, 'w') as json_file:
        json.dump(moves_data, json_file, indent=4)
    
    return

def create_32_pieces(game_state):
    pieces = {}
    k = 1
    for row in range(len(game_state)):
        for col in range(len(game_state[row])):
            value = game_state[row, col]
            # print(value)
            if value != 0:  # Si une pièce est présente sur la case
                color = -1 if value < 0 else 1  # Détermine la couleur
                pieces[f"piece{k}"]=('7', (row, col), (-1,-1), color)

                k+=1
    return pieces

def detect_special_moves(initial_game_state, new_game_state, game_dict):
    """
    Détecte les mouvements spéciaux : roque, échec, échec et mat.
    
    :param initial_game_state: np.ndarray
        L'état initial du jeu sous forme de matrice 8x8.
    :param new_game_state: np.ndarray
        L'état suivant du jeu sous forme de matrice 8x8.
    :param game_dict: dict
        Dictionnaire contenant les informations sur les pièces et leur mouvement.
    
    :return: tuple(updated_game_state, str or None)
        - updated_game_state : L'état mis à jour du jeu après le roque, si détecté.
        - notation : La notation algébrique du mouvement, y compris "O-O", "O-O-O", "+", "#" ou None.
    """
    
    # Utiliser la fonction rocks_recognition pour détecter le roque
    updated_game_state, roque_notation = rock_recognition(initial_game_state, new_game_state, game_dict)
    
    if roque_notation:
        return updated_game_state, roque_notation
    
    # Identifier les indices des positions initiale et finale
    initial_indices, new_indices, row_diff, col_diff = get_indices(initial_game_state, new_game_state)
    
    # Convertir les indices en notation algébrique
    initial_square = f"{chr(97 + initial_indices[1])}{8 - initial_indices[0]}"
    new_square = f"{chr(97 + new_indices[1])}{8 - new_indices[0]}"
    
    # Vérifier si c'est une capture
    is_capture = new_game_state[new_indices] != 0 and initial_game_state[initial_indices] * new_game_state[new_indices] <= 0

    # Construire la base du mouvement
    if is_capture:
        move = f"{initial_square} x {new_square}"
    else:
        move = f"{initial_square} -> {new_square}"
    
    # Vérifier si c'est un échec ou un échec et mat
    if is_check(new_game_state, is_white=initial_game_state[initial_indices] > 0):
        if is_checkmate(new_game_state, is_white=initial_game_state[initial_indices] > 0):
            move += "#"
        else:
            move += "+"
    
    return updated_game_state, move

def is_check(game_state, is_white):
    """
    Vérifie si le roi est en échec.

    Parameters:
    - game_state : np.ndarray
        L'état actuel du jeu sous forme de matrice 8x8.
    - is_white : bool
        True si c'est le roi blanc, False si c'est le roi noir.

    Returns:
    - bool : True si le roi est en échec, sinon False.
    """
    king_value = 5 if is_white else -5
    king_position = tuple(np.argwhere(game_state == king_value)[0])  # Trouve le roi

    # Parcourir toutes les cases pour voir si une pièce ennemie peut atteindre le roi
    for i in range(8):
        for j in range(8):
            if game_state[i, j] != 0 and (is_white and game_state[i, j] < 0 or not is_white and game_state[i, j] > 0):
                if can_attack(game_state, (i, j), king_position):
                    return True
    return False

def is_checkmate(game_state, is_white):
    """
    Vérifie si le roi est en échec et mat.

    Parameters:
    - game_state : np.ndarray
        L'état actuel du jeu sous forme de matrice 8x8.
    - is_white : bool
        True si c'est le roi blanc, False si c'est le roi noir.

    Returns:
    - bool : True si le roi est en échec et mat, sinon False.
    """
    if not is_check(game_state, is_white):
        return False
    
    # Vérifier tous les mouvements possibles
    for i in range(8):
        for j in range(8):
            if (is_white and game_state[i, j] > 0) or (not is_white and game_state[i, j] < 0):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        new_pos = (i + di, j + dj)
                        if is_valid_move(game_state, (i, j), new_pos, is_white):
                            # Simuler le mouvement
                            temp_state = simulate_move(game_state, (i, j), new_pos)
                            if not is_check(temp_state, is_white):
                                return False
    return True

def can_attack(game_state, source, target):
    """
    Vérifie si une pièce à la position `source` peut attaquer la position `target`.

    Parameters:
    - game_state : np.ndarray
        L'état actuel du jeu sous forme de matrice 8x8.
    - piece_dict : dict
        Dictionnaire contenant les informations sur les pièces et leur mouvement.
    - source : tuple(int, int)
        La position de la pièce qui attaque (ligne, colonne).
    - target : tuple(int, int)
        La position de la cible (ligne, colonne).

    Returns:
    - bool : True si la pièce peut attaquer la cible, sinon False.
    """

    # Récupérer les informations sur la pièce source
    piece = game_state[source]
    piece_type = abs(piece)
    is_white = piece > 0

    # Si la cible est hors du plateau, on retourne False
    if not (0 <= target[0] < 8 and 0 <= target[1] < 8):
        return False

    # Si la case cible est occupée par une pièce alliée, on ne peut pas attaquer
    if game_state[target] != 0 and (game_state[source] * game_state[target] > 0):
        return False

    # Vérifier le type de pièce et sa capacité à attaquer la case cible
    if piece_type == 1:  # Pion
        return can_pawn_attack(game_state, source, target, is_white)
    elif piece_type == 2:  # Tour
        return can_rook_attack(game_state, source, target)
    elif piece_type == 3:  # Cavalier
        return can_knight_attack(source, target)
    elif piece_type == 4:  # Fou
        return can_bishop_attack(game_state, source, target)
    elif piece_type == 5:  # Roi
        return can_king_attack(source, target)
    elif piece_type == 6:  # Dame
        return can_queen_attack(game_state, source, target)
    
    return False

def can_pawn_attack(game_state, source, target, is_white):
    """
    Vérifie si un pion peut attaquer la position cible.

    :param game_state: L'état actuel du jeu
    :param source: Position du pion (source)
    :param target: Position de la cible
    :param is_white: True si le pion est blanc, False si noir
    :return: True si le pion peut attaquer, sinon False
    """
    row_diff = target[0] - source[0]
    col_diff = abs(target[1] - source[1])
    
    # Le pion attaque une case diagonale
    if is_white:
        return row_diff == -1 and col_diff == 1
    else:
        return row_diff == 1 and col_diff == 1

def can_rook_attack(game_state, source, target):
    """
    Vérifie si une tour peut attaquer la position cible.

    :param game_state: L'état actuel du jeu
    :param source: Position de la tour (source)
    :param target: Position de la cible
    :return: True si la tour peut attaquer, sinon False
    """
    # La tour se déplace horizontalement ou verticalement
    if source[0] == target[0]:  # Même ligne
        return all(game_state[source[0], min(source[1], target[1])+1:max(source[1], target[1])]) == 0
    elif source[1] == target[1]:  # Même colonne
        return all(game_state[min(source[0], target[0])+1:max(source[0], target[0]), source[1]]) == 0
    return False

def can_knight_attack(source, target):
    """
    Vérifie si un cavalier peut attaquer la position cible.

    :param source: Position du cavalier (source)
    :param target: Position de la cible
    :return: True si le cavalier peut attaquer, sinon False
    """
    row_diff = abs(target[0] - source[0])
    col_diff = abs(target[1] - source[1])
    return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

def can_king_attack(source, target):
    """
    Vérifie si un roi peut attaquer la position cible.

    :param source: Position du roi (source)
    :param target: Position de la cible
    :return: True si le roi peut attaquer, sinon False
    """
    row_diff = abs(target[0] - source[0])
    col_diff = abs(target[1] - source[1])
    return max(row_diff, col_diff) == 1

def can_bishop_attack(game_state, source, target):
    """
    Vérifie si un fou peut attaquer la position cible.

    :param game_state: L'état actuel du jeu
    :param source: Position du fou (source)
    :param target: Position de la cible
    :return: True si le fou peut attaquer, sinon False
    """
    row_diff = abs(target[0] - source[0])
    col_diff = abs(target[1] - source[1])
    if row_diff != col_diff:
        return False
    step_row = (target[0] - source[0]) // row_diff
    step_col = (target[1] - source[1]) // col_diff
    for step in range(1, row_diff):
        intermediate = (source[0] + step * step_row, source[1] + step * step_col)
        if game_state[intermediate] != 0:
            return False
    return True

def can_queen_attack(game_state, source, target):
    """
    Vérifie si une dame peut attaquer la position cible.

    :param game_state: L'état actuel du jeu
    :param source: Position de la dame (source)
    :param target: Position de la cible
    :return: True si la dame peut attaquer, sinon False
    """
    return can_rook_attack(game_state, source, target) or can_bishop_attack(game_state, source, target)

def simulate_move(game_state, source, target):
    """
    Simule un mouvement d'une pièce sans modifier l'état réel du jeu.
    
    Parameters:
    - game_state : np.ndarray
        L'état actuel du jeu sous forme de matrice 8x8.
    - source : tuple(int, int)
        La position de départ de la pièce (ligne, colonne).
    - target : tuple(int, int)
        La position cible de la pièce (ligne, colonne).
    
    Returns:
    - temp_state : np.ndarray
        L'état du jeu après le mouvement simulé.
    """
    # Créer une copie de l'état du jeu pour ne pas modifier l'état original
    temp_state = game_state.copy()

    # Obtenir la pièce à la position source
    piece = temp_state[source]

    # Déplacer la pièce de la position source à la position cible
    temp_state[target] = piece
    temp_state[source] = 0  # La case source devient vide

    return temp_state

def is_valid_move(game_state, source, target, is_white):
    """
    Vérifie si un mouvement est valide en simulant le mouvement et en vérifiant les règles d'échec.
    
    Parameters:
    - game_state : np.ndarray
        L'état actuel du jeu sous forme de matrice 8x8.
    - source : tuple(int, int)
        La position de départ de la pièce (ligne, colonne).
    - target : tuple(int, int)
        La position cible de la pièce (ligne, colonne).
    - is_white : bool
        True si la couleur de la pièce qui effectue le mouvement est blanche, sinon False pour noir.
    
    Returns:
    - bool : True si le mouvement est valide, sinon False.
    """
    # Vérifier si le mouvement est autorisé pour la pièce
    if not can_attack(game_state, {}, source, target):  # Si le mouvement n'est pas valide
        return False

    # Simuler le mouvement
    temp_state = simulate_move(game_state, source, target)

    # Vérifier si le roi est en échec après le mouvement
    if is_check(temp_state, is_white):
        return False  # Si le roi est en échec, le mouvement est invalide

    return True  # Le mouvement est valide avec le fichier

#*---------------------------------------------Main----------------------------------------------*#

if __name__ == "__main__":
    
    # Set the path to the video
    script_directory = os.path.dirname(os.path.abspath(__file__))
    video_moving = '../../videos/LR_move.mp4'
    video_path = os.path.join(script_directory, video_moving)
    
    # Set the first frame number to begin the analyze
    first_frame_to_analyze = 0
    
    # Set the output file name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_states_file = os.path.join(script_directory, f"states_{timestamp}.json")
    
    # Video processing
    process_video(video_path, first_frame_to_analyze, output_states_file)

#*-----------------------------------------------------------------------------------------------*#
#*---------------------------------------------EOF-----------------------------------------------*#
#*-----------------------------------------------------------------------------------------------*#