import cv2
import numpy as np
import json
from datetime import datetime
import os
from calib import corners_calibration
from detect_piece import find_chessboard_corners, pieces_initial
from detect_move import detect_special_moves
from piece_recognition import pieces_recognition

def new_state(current_frame, next_frame,game_dict,pieces_number, rose_pos, blue_pose, corners, side):
    """
    Met à jour l'état du jeu en comparant les frames actuelles et suivantes.
    
    :param current_frame: Frame actuelle contenant l'état du jeu
    :param next_frame: Prochaine frame contenant l'état du jeu modifié
    
    :return: updated_game_state : L'état du jeu mis à jour
    """
    # Charger les états du jeu pour les deux frames
    initial_game_state = pieces_initial(current_frame, rose_pos, blue_pose, corners, current_frame)
    new_game_state = pieces_initial(next_frame, rose_pos, blue_pose, corners, current_frame)
    
    
    
    
    # Reconnaissance des pièces et mise à jour de l'état
    updated_game_state, multi_pieces = pieces_recognition(
        initial_game_state, new_game_state, game_dict, pieces_number, side
    )
    
    
    return updated_game_state



def end_of_video(video_capture):
    """
    Vérifie si on a atteint la fin de la vidéo.

    
    Returns:
    - bool : True si la fin de la vidéo est atteinte, sinon False.
    """
    # Lire une frame de la vidéo
    ret, _ = video_capture.read()
    
    # Si `ret` est False, la fin de la vidéo est atteinte
    return not ret



def get_next_frame(video_capture):
    """
    Lit la prochaine frame d'une vidéo.
    
    Parameters:
    - video_capture : cv2.VideoCapture
        Objet OpenCV représentant la vidéo en cours.
    
    Returns:
    - tuple(bool, ndarray or None): 
        - bool : Indique si la lecture de la frame a réussi.
        - ndarray : La frame capturée sous forme d'image (ou `None` si la fin de la vidéo est atteinte).
    """
    # Lire la prochaine frame
    ret, frame = video_capture.read()
    
    # Retourner un booléen et la frame
    return ret, frame

def json_file(data, move, frame_number, game_state):
    data["moves"].append(move)
    data["game_states"].append({"frame": frame_number,
                                "gs" : game_state})

def json_file(move, initial_frame,frame, rose_pos, blue_pose, corners):
    """
    Fonction pour générer un fichier JSON avec les mouvements détectés et les états du jeu.
    
    :param move: Le mouvement à détecter et à enregistrer
    :param initial_frame: Le premier frame pour commencer l'analyse
    :return: data : un dictionnaire contenant les mouvements et les états du jeu
    """
    
    # Fonction fictive pour détecter les coins de l'échiquier
    real_grid = find_chessboard_corners(frame, corners)
    
    # État initial du jeu avec la représentation des pièces
    initial_game_state = pieces_initial(frame, rose_pos, blue_pose, corners, initial_frame)
    
    # Structure de données pour suivre les états du jeu
    data = {
        "moves": [],  # Liste des mouvements détectés
        "game_states": []  # Liste des états du jeu capturés
    }
    
    # Ajouter l'état initial du jeu
    data["game_states"].append({"frame": initial_frame, "gs": initial_game_state.tolist()})
    
    # État courant du jeu (copie de l'état initial)
    game_state = initial_game_state.copy()
    
    # Boucle pour analyser les mouvements
    current_frame = initial_frame
    while True:
        # Capturer le prochain état du jeu
        next_frame = get_next_frame(current_frame)
        next_game_state = pieces_initial(next_frame, rose_pos, blue_pose, corners, initial_frame)
        
        # Comparer les états pour détecter un mouvement
        if not np.array_equal(game_state, next_game_state):  # S'il y a un changement
            move_detected, move_notation = detect_special_moves(current_frame, next_frame)
            data["moves"].append({"frame": current_frame, "move": move_notation})
            
            # Mettre à jour l'état du jeu
            game_state = new_state(current_frame, next_frame).copy()
            
            # Ajouter l'état du jeu modifié
            data["game_states"].append({"frame": current_frame, "gs": game_state.tolist()})
        
        # Passer au prochain frame
        current_frame += 1
        
        # Condition d'arrêt : fin de la vidéo
        if end_of_video(next_frame): 
            break
    
    # Sauvegarder les données dans un fichier JSON
    with open("game_states.json", "w") as f:
        json.dump(data, f, indent=2)
    
    return data

