
#*-------------------------------------------Imports---------------------------------------------*#

from datetime import datetime
from detection import *
from output_json import *
from logic import *

import cv2
import json
import numpy as np
import os
import time

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
    print("Initializing...")
    is_initialized = False
    game_state = np.zeros((8, 8), dtype=int)

    game_state = detect_pieces(frame, game_state, corners, original_frame, is_initialized)
    if game_state is None: 
        is_initialized = False
    else:
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
    last_frame_checked = 0
    a_problem = 0
    rollback_frames = 0
    initial_problem_frame = 0

    #initialisation pour pieces_recignition
    first_frame_analized = False
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
    data["time"] = []
    

    t0 = time.time()
    # Loop through video
    while cap.isOpened():
        
        # Read frame
        ret_cap, frame = cap.read()
        
        if not ret_cap:
            
            print("End of video or reading error")
            break
        
        original_frame = frame.copy()
        
        # Process every 50 frame after the first frame to analyze or the next frame if the flag is set
        if (frame_number >= first_frame_to_analyze) and ((((frame_number + 1) % 50) == 1) or (process_next_frame == True)):
            print("Frame = ", frame_number)
            # Perform a calibration to get the chessboard corners and the stickers positions
            is_calibrate, rose_pos, blue_pos, corners = corners_calibration(frame)
            print("Is_calibrate = ", is_calibrate)
            # If calibration is not succesful
            if (is_calibrate == False):
#                if a_problem == 0:
 #                   initial_problem_frame = frame_number
                
                # If there is between 1 and 3 problems
  #              if (a_problem > 0) and (a_problem < 3) and (frame_number > last_frame_checked):
                    
                    # Update the number of problems
   #                 a_problem += 1
                    
                    # Set the new analyzed frame number
    #                frame_number = frame_number - rollback_frames
                    
                    # Set the cap parameter to the new frame number
     #               cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    
                    # Set the flag to process the next frame
      #              process_next_frame = True
                    
       #             is_initialized = False
                    
                    # Reset the loop
        #            continue
                
                # If there is more than 3 problems
         #       elif (a_problem >= 3):
                    
                    # Set the frame number to its initial value plus 1
          #          frame_number = initial_problem_frame + 1
                    
                    # Set the cap parameter to the new frame number
           #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    
                    # Set the flag to process the next frame
            #        process_next_frame = True
             #       a_problem = 0
                    
                    # Reset the flags
              #      is_initialized = False
               #     process_frame = False
                    
                    # Reset the loop 
                #    continue
                
                # If there is no problem
                #else:
                
                # Process the next frame
                process_next_frame = True
                frame_number += 10
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                is_initialized = False
                continue
            
            # If it is the first frame to analyze
            if (frame_number == first_frame_to_analyze) or (is_initialized == False):
                
                # Perform the initilization of the game state
                is_initialized, game_state = pieces_initial(frame, corners, original_frame)
                print("Is_initiatized = ", is_initialized)
                
                game_state = np.rot90(game_state, k=1)

                if (frame_number == first_frame_to_analyze):
                    print("First frame to analyze")

                    game_state0 = game_state

                    dict_pieces = create_32_pieces(game_state0)

                    first_frame_analized = True
                    print(game_state0)
                # cv2.waitKey(0)

            #### PIECE RECOGNITION ####
            if is_initialized and (frame_number != first_frame_to_analyze):
                if first_frame_analized : 
                    
                    #side = get_initial_side(game_state0, game_state)
                    side = 1
                    # If the side is not determined
                    #if side == None:
                        
                        # Process the next frame
                     #   process_next_frame = True
                        
                        # Increment the frame count
                      #  frame_number += 10
                        
                        # Set the cap parameter to the new frame number
                       # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                        
                        #is_initialized = False
                        #continue

                    first_frame_analized = False

                print('Piece recognition...')
                updated_game_state, multi_pieces = pieces_recognition(game_state0, game_state, dict_pieces, pieces_number, side)
                data = json_file(data, dict_pieces, frame_number, game_state0, updated_game_state, time.time()-t0)
                print(updated_game_state)
                
                # If there is a first problem with the piece recognition (no piece movement detected)
                # if multi_pieces and (a_problem == 0):
                #     print("There is a problem")
                    
                #     # Update the number of problems
                #     a_problem += 1
                    
                #     # Set the initial problem frame number
                #     initial_problem_frame = frame_number
                    
                #     # Get the difference between the current frame number and the last frame checked
                #     frame_diff = frame_number - last_frame_checked
                    
                #     # Get the quantity of frames to rollback (1/4 of the difference)
                #     rollback_frames = frame_diff // 4
                    
                #     # Set the new analyzed frame number
                #     frame_number = frame_number - rollback_frames
                    
                #     # Set the cap parameter to the new frame number
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    
                #     # Set the flag to process the next frame
                #     process_next_frame = True
                    
                #     is_initialized = False
                    
                #     # Reset the loop
                #     continue
                
                # # If there is another problem with the piece recognition (no piece movement detected), the current frame number above the last frame checked and there is no more than 4 rollbacks already made
                # elif multi_pieces and (frame_number > last_frame_checked) and (a_problem < 3):
                #     print("There is a problem")
                    
                #     # Update the number of problems
                #     a_problem += 1
                    
                #     # Set the new analyzed frame number
                #     frame_number = frame_number - rollback_frames
                    
                #     # Set the cap parameter to the new frame number
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    
                #     # Set the flag to process the next frame
                #     process_next_frame = True
                    
                #     is_initialized = False
                    
                #     # Reset the loop
                #     continue
                    
                # # If there is no problem with the piece recognition or there is more than 4 rollbacks already made
                # else :
                    
                    # If there was one or more problems with the piece recognition
                    # if a_problem > 0:
                        
                    #     # Set the frame number above the initial problem frame number
                    #     frame_number = initial_problem_frame + 1
                        
                    #     # Set the cap parameter to the new frame number
                    #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                        
                    #     # Reset the variables
                    #     last_frame_checked = frame_number
                    #     a_problem = 0
                    #     rollback_frames = 0
                    #     initial_problem_frame = 0
                        
                    #     # Reset the flags
                    #     is_initialized = False
                    #     process_frame = False
                    
                # Update the side of the next move
                if side == 1:

                    side = -1
                    
                elif side == -1:
                    
                    side = 1
                
                game_state0 = updated_game_state
                    # print(game_state0)
                
            # Reset the next frame flag
            process_next_frame = False
    
        # Increment the frame count
        frame_number += 1
        is_initialized = False

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):    
            break
        
        # Display the frame
        # cv2.imshow('Frame', frame)
    
    # Release video and close windows
    cap.release()
    cv2.destroyAllWindows()

    final_game_state = get_final_game_state(updated_game_state, dict_pieces, pieces_number)

    final_game_state = context_update(final_game_state, dict_pieces, pieces_number)
    
    data = json_file(data, dict_pieces, frame_number, updated_game_state, final_game_state, time.time()-t0)

    # changement de nom des pieces en consequence dans data.
    data = update_JSON(data)
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


#*---------------------------------------------Main----------------------------------------------*#

if __name__ == "__main__":
    
    # Set the path to the video
    script_directory = os.path.dirname(os.path.abspath(__file__))
    video_moving = '../../../videos/challenge_fix.MOV'
    video_path = os.path.join(script_directory, video_moving)
    
    # Set the first frame number to begin the analyze
    first_frame_to_analyze = 0
    
    # Set the output file name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_states_file = os.path.join(script_directory, f"states_{timestamp}.txt")
    
    # Video processing
    process_video(video_path, first_frame_to_analyze, output_states_file)

#*---------------------------------------------EOF-----------------------------------------------*#