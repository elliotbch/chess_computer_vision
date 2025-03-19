
#*-------------------------------------------Imports---------------------------------------------*#

from calibration import *
from datetime import datetime

import cv2
import numpy as np
import os

#*------------------------------------------Functions--------------------------------------------*#

def find_chessboard_corners(frame, real_corners, grid_size=(9, 9)):
    """
    Trouver tous les coins du plateau d'échecs en utilisant une matrice d'homographie préalablement calculée.

    :param frame: L'image contenant le plateau d'échecs.
    :param homography: La matrice d'homographie déjà calculée entre la grille idéale et l'image réelle.
    
    :return: La grille de coins du plateau d'échecs sous forme de matrice 9x9.
    """
    # Créer les points idéaux de la grille d'échecs (9x9)
    ideal_points = np.array([[100*j, 100*i] for i in range(grid_size[0]) for j in range(grid_size[1])], dtype=np.float32)
    
    ideal_corners = np.array([
        ideal_points[0],            # Coin en haut à gauche (0, 0)
        ideal_points[grid_size[1] - 1],  # Coin en haut à droite (8, 0)
        ideal_points[-1],  # Coin en bas à gauche (0, 8)
        ideal_points[(grid_size[0] - 1) * grid_size[1]]           # Coin en bas à droite (8, 8)
    ], dtype=np.float32)
    # print("ideal corners", ideal_corners)
    # print("real_corners", real_corners)

    real_corners.reshape(-1, 2)
    # print(real_corners)
    homography, _ = cv2.findHomography(real_corners, ideal_corners)
    # print(homography)
    height, width = frame.shape[:2]
    transformed_frame = cv2.warpPerspective(frame, homography, (800, 800))
    # cv2.imshow("transformed_perspective", transformed_frame)
    # print(width, height)

    # Transformer les points idéaux vers l'espace de l'image à l'aide de l'homographie
    ideal_points = ideal_points.reshape(-1, 1, 2)  # Format requis par OpenCV
    real_grid = cv2.perspectiveTransform(ideal_points, homography)
    
    # Redimensionner les points transformés en une matrice 9x9
    real_grid = real_grid.reshape(grid_size[0], grid_size[1], 2)
    # print(real_grid)
    
    # Dessiner les coins sur l'image pour vérification
    for row in real_grid:
        for point in row:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # Affichage de l'image avec les coins détectés
    # cv2.imshow("Chessboard Corners", frame)
    # cv2.waitKey(0)
    
    # Retourner la grille des coins
    return real_grid, transformed_frame

def detect_piece_color(img, area_roi, color_square):
    bool_piece = False

    if color_square:
        lower_saturated_color = np.array([0, 150, 30])  # Seuil HSV : saturation minimale
    else:
        lower_saturated_color = np.array([0, 25, 0])  # Seuil HSV : saturation minimale
    upper_saturated_color = np.array([180, 255, 255])  # Saturation maximale

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)

    brightness = np.mean(hsv_image[:, :, 2])
    # print("brightness =", brightness)

    if (color_square and brightness > 135) or (not color_square and brightness > 220):
        # print("TOO MUCH LIGHT !!!")
        factor = 0.75
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * factor  # Multiplier le canal V par un facteur pour réduire la luminosité
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    mask = cv2.inRange(hsv_image, lower_saturated_color, upper_saturated_color)

    # cv2.imshow("Mask of Saturated Colors", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
    # cv2.imshow('contour_img', img )
    # cv2.waitKey(0)
    for contour in contours:
        area = cv2.contourArea(contour)
        # print(area, area_roi)
        if 1300 < area :
            bool_piece = True
            # print("piece detected in square")
            break  
    # print(bool_piece)
    # cv2.waitKey(0)
    return bool_piece

def detect_pieces(frame, game_state, corners, original_frame, is_initialized, square_size=100):
    """
    Détecte les pièces sur un échiquier en analysant chaque case.
    :param frame: Image de l'échiquier.
    :param corners: Coordonnées des coins détectés de l'échiquier.
    :param grid_size: Taille de l'échiquier (par défaut 8x8).
    :return: Image annotée avec les pièces détectées.
    """
    print("detecting...")
    try:
        real_grid, frame2D = find_chessboard_corners(original_frame, corners)
    except AttributeError as e:
        return None

    color = 1
    area_roi = square_size*square_size
    # Boucle sur chaque case en utilisant les points d'intersection (real_grid)
    for row in range(8):  # Sur un échiquier 8x8, il y a 7 espaces entre les 8 points
        for col in range(8):
            if not is_initialized:
                if col <= 3 :
                    color = 1
                else:
                    color = -1
            
            # print("square {}, {}".format(row, col))

            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            
            # Extraire la ROI correspondant au carré
            roi = frame2D[y1:y2, x1:x2]
            # cv2.imshow(f"ROI: Square", roi)

            ########### On essaie de détecter les contours pour voir si il y a une pièce
            bool_piece = False
            color_square = (row%2 == 0 and col%2 == 0) or (row%2 == 1 and col%2 == 1)           #here for black square

            bool_piece = detect_piece_color(roi, area_roi, color_square)

            if bool_piece : 
                game_state[row][col] = color*7
    
            # cv2.imshow("Frame Annotée avec Zones", annotated_frame)
            # cv2.waitKey(0)

    # print(game_state)
    return game_state

def pieces_initial(frame, rose_pos, blue_pose, corners, original_frame):
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

    game_state = detect_pieces(frame, game_state, corners, original_frame)
    print(game_state)

    is_initialized = True

    return is_initialized, game_state

def process_frame(frame):
    original_frame = frame.copy()
    is_calibrate, rose_pos, blue_pos, corners = corners_calibration(frame)
    print("is_calibrate = ", is_calibrate)
    is_initialized, game_state = pieces_initial(frame, rose_pos, blue_pos, corners, original_frame)
    print("is_initiatized = ", is_initialized)

#*---------------------------------------------Main----------------------------------------------*#

if __name__ == "__main__":
    
    # Set the path to the video
    script_directory = os.path.dirname(os.path.abspath(__file__))
    frame = './frames/frame03.png'
    frame_path = os.path.join(script_directory, frame)

    frame = cv2.imread(frame_path)

    # cv2.imshow('frame', frame)
    
    # Set the output file name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_states_file = os.path.join(script_directory, f"states_{timestamp}.txt")
    
    # Video processing
    process_frame(frame)

    # cv2.imshow('frame', frame)
    
#*---------------------------------------------EOF-----------------------------------------------*#