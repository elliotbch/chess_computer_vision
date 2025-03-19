
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
import json

from detect_piece import *
from piece_recognition import *
from detect_move import *

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

    #initialisation pour pieces_recignition
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

    #initialisaiton du json_file
    data = {} # data contains the list of the move and the game_state + frame_number after each move
    data["moves"] = []
    # data["moves"] = game_mov
    data["game_states"] = []
    
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
            # If calibration is not succesful
            if (is_calibrate == False):               
                # Process the next frame
                process_next_frame = True
                frame_number += 10
                continue
            
            # If it is the first frame to analyze
            if (frame_number == first_frame_to_analyze) or (is_initialized == False):
                
                # Perform the initilization of the game state
                is_initialized, game_state = pieces_initial(frame, corners, original_frame)
                print("is_initiatized = ", is_initialized)

                if (frame_number == first_frame_to_analyze):
                    print("first frame to analyze")
                    game_state = np.rot90(game_state, k=-1)
                    game_state0 = game_state 

                    dict_pieces0 = create_32_pieces(game_state0)

                cv2.waitKey(0)
            
                 
            game_state = np.rot90(game_state, k=-1)
            dict_pieces = create_32_pieces(game_state)

            #### PIECE RECOGNITION ####
            if is_initialized and (frame_number != first_frame_to_analyze):
                print('piece recognition...')
                updates_game_state, _ = pieces_recognition(game_state0, game_state, dict_pieces0, pieces_number, side)
                game_state0 = game_state
                dict_pieces0 = dict_pieces

                print(updates_game_state)

                # dans detect_move il nous faut trouver qu'elle move est detecté poyr obtenir le move a mettre dans le json
                # moves = detect_move()
                # correspond a tous les mouvements faits pdt le chagement de frame, formaté de la manière officiel en supposant que ce sont tous des pions

                data = json_file(data, moves, frame_number, updates_game_state)
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

    json.dump(data, open("game.json", "w"))
    
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

def json_file(data, moves, frame_number, game_state):
    for move in moves :
        data["moves"].append(move)
    data["game_states"].append({"frame": frame_number,
                                "gs" : game_state})
    return data
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
    output_states_file = os.path.join(script_directory, f"states_{timestamp}.txt")
    
    # Video processing
    process_video(video_path, first_frame_to_analyze, output_states_file)

#*-----------------------------------------------------------------------------------------------*#
#*---------------------------------------------EOF-----------------------------------------------*#
#*-----------------------------------------------------------------------------------------------*#