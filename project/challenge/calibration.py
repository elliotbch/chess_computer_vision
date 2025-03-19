
#*-------------------------------------------Imports---------------------------------------------*#

import cv2
import math
import numpy as np

#*------------------------------------------Functions--------------------------------------------*#

def are_corners_within_bounds(int_corners, ext_corners):
    """
    Vérifie si tous les coins détectés sont à l'intérieur des coins externes calculés.
    
    :param detected_corners: Tableau de coins détectés.
    :param external_corners: Tableau des coins externes, supposé être un quadrilatère (4 points).
    :return: True si tous les coins détectés sont à l'intérieur des limites, False sinon.
    """
    # Conversion des coins externes en un contour (polygone)
    contour = np.array(ext_corners, dtype=np.float32).reshape((-1, 1, 2))
    
    point = int_corners[0]  # Chaque coin détecté est dans un tableau [[x, y]]
    result = cv2.pointPolygonTest(contour, (point[0], point[1]), False)
    # print(result)

    # Si un point est en dehors du contour, retourner False
    if result < 0:
        return False

    # Tous les points sont à l'intérieur du contour
    return True


def corners_calibration(frame):
    """
    Perform a camera calibration as well as a corners and stickers detection

    :param frame: the frame to analyze
    
    :return is_calibrate: a flag to signal that the calibration is done
    :return rose_pos: the position of the rose sticker
    :return blue_pos: the position of the blue sticker
    :return corners: an array with the chessboard corner coordinates
    """
    
    # Initialize the the return variables
    is_calibrate = False
    rose_pos = [(0, 0), (0, 0)]
    blue_pos = [(0, 0), (0, 0)]
    corners = np.array([])
    
    # Define the internal board dimensions
    board_dimension = (7, 7)
    
    # Define methods to find the chessboard corners
    methods = [
        lambda img: img,
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        lambda img: cv2.equalizeHist(img)
    ]
    
    # Define criteria for corners refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Get the sticker coordinates
    x_rose, y_rose, x_blue, y_blue, ret_stickers = corners_stickers(frame)
    rose_pos, blue_pos = [x_rose, y_rose], [x_blue, y_blue]
    
    # If the stickers are not found, set the flag and return
    if not ret_stickers:
        print("calibration not possible")
        is_calibrate = False
        return is_calibrate, rose_pos, blue_pos, corners
    
    # Convert to grayscale and apply a Gaussian blur filter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find chessboard corners by looping through the 3 methods
    for method in methods:
        
        processed = method(gray)
        
        # Test with the first function
        ret_corner, corners = cv2.findChessboardCorners(processed, board_dimension, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret_corner:
            
            break
        
        # Test with the second function
        ret_corner, corners = cv2.findChessboardCornersSB(processed, board_dimension, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        
        if ret_corner:
            
            break
        
    # If the corners are not found, set the flag and return
    if not ret_corner:
        return is_calibrate, rose_pos, blue_pos, corners
    
    # Refine the pixel coordinates
    # print(corners)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Determine the coordinates of the internal corners
    a1 = tuple(corners2[0][0])
    a8 = tuple(corners2[board_dimension[0]-1][0])
    h8 = tuple(corners2[-1][0])
    h1 = tuple(corners2[-(board_dimension[0])][0])
    internal_corners = np.array([a1, a8, h1, h8], dtype = np.float32)
    print('internal_corners')
    
    # Find the external corners of the chessboard and get the homography matrix
    external_corners, homography = corners_external(corners2, board_dimension)
    print('external_corners, homography')

    # Process the corners to determine the correct order
    final_corners, final_ext_corners = np.array(corners_process(internal_corners, external_corners, rose_pos, blue_pos))
    
    cv2.polylines(frame, [final_ext_corners.astype(np.int32)], isClosed = True, color = (0, 255, 0), thickness = 2)
    display_corners(final_corners, frame)

    is_calibrate = True
    return is_calibrate, rose_pos, blue_pos, final_ext_corners

def corners_external(corners, board_dimension, square_size=5):
    """
    Find the external corners of the chessboard using homography
    
    :param corners: detected internal corners of the chessboard
    :param board_dimension: the number of internal corners (width, height)
    :param square_size: the size of a square on the chessboard (in some units, e.g., cm or pixels)
    
    :return external_corners: external corners of the chessboard
    :return homography: the homography matrix
    """
    
    # Internal corners from detected chessboard (e.g., 6x6 for a 7x7 grid)
    internal_corners = corners.reshape(-1, 2)
    
    # Define the real-world coordinates of the internal corners
    # Assuming the chessboard is in the XY plane with Z=0
    obj_points = np.zeros((board_dimension[0] * board_dimension[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:(board_dimension[0]), 0:(board_dimension[1])].T.reshape(-1, 2)
    obj_points *= square_size
    
    # Define the real-world coordinates for the external corners (8x8 grid for a 7x7 chessboard)
    external_obj_points = np.array([
        [-square_size, -square_size], # Top-left external corner - a8
        [(board_dimension[0]) * square_size, -square_size], # Top-right external corner - h8
        [-square_size, (board_dimension[1]) * square_size], # Bottom-left external corner - a1
        [(board_dimension[0]) * square_size, (board_dimension[1]) * square_size] # Bottom-right external corner - h1
    ], dtype = np.float32)
    
    # Find the homography that maps the internal corners to the ideal grid internal corners
    homography, _ = cv2.findHomography(obj_points[:, :2], internal_corners)
    
    # Apply the homography to the external corners
    external_corners = cv2.perspectiveTransform(external_obj_points.reshape(-1, 1, 2), homography)
    
    return external_corners.reshape(-1, 2), homography

def corners_h1_h8 (pts, i_A, i_B):
    """
    Determine the indices of the points C and D given the indices of the points A and B
    
    :param pts: a list of 4 points
    :param i_A: the index of point A
    :param i_B: the index of point B
    
    :return i_C: the index of point C
    :return i_D: the index of point D
    """
    
    index = [0, 1, 2, 3]
    index_left = [x for x in index if x != i_A and x != i_B]
    i, j = index_left[0], index_left[1]
            
    AC = np.linalg.norm(pts[i] - pts[i_A])
    AD = np.linalg.norm(pts[j] - pts[i_A])
    AB = np.linalg.norm(pts[i_B] - pts[i_A])
    BC = np.linalg.norm(pts[i_B] - pts[i])
    
    vect_AB = np.array(pts[i_A]) - np.array(pts[i_B])
    vect_BC = np.array(pts[i_B]) - np.array(pts[i])
    
    deg_ABC = math.degrees(np.arccos(np.dot(vect_AB, vect_BC)/(AB*BC)))
    
    if AC > AD and deg_ABC > 90:
        
        i_C = i
        i_D = j
        
    elif AC < AD and deg_ABC < 90: 
        
        i_C = i
        i_D = j
        
    else:
        
        i_C = j
        i_D = i 
    
    return i_C, i_D

def corners_process(int_corners, ext_corners, rose_pos, blue_pos):
    """
    Process the corners to determine the correct order and orientation
    
    :param int_corners: internal corners of the chessboard
    :param ext_corners: external corners of the chessboard
    :param rose_pos: position of the rose sticker
    :param blue_pos: position of the blue sticker
    
    :return ordered_corners: ordered corners of the chessboard
    :return ord_ext_corners: ordered external corners of the chessboard
    """
    
    # Find the center of each corner
    center_corners = (int_corners + ext_corners) / 2
    
    # Determine the order to have the correct orientation
    # Compute the distances of the corners to the stickers
    distances_to_x_blue = [np.linalg.norm(corner - blue_pos) for corner in ext_corners]
    index_a1 = np.argmin(distances_to_x_blue)
    distances_to_x_rose = [np.linalg.norm(corner - rose_pos) for corner in ext_corners]
    index_a8 = np.argmin(distances_to_x_rose)
    
    # Get the corners in the correct order
    a1 = center_corners[index_a1]
    a8 = center_corners[index_a8]

    index_h1, index_h8 = corners_h1_h8(ext_corners, index_a8, index_a1)
    h8 = center_corners[index_h8]
    h1 = center_corners[index_h1]
    ordered_corners = [a1, a8, h8, h1]

    ord_ext_corners=[ext_corners[index_a1], 
                     ext_corners[index_a8],
                     ext_corners[index_h8],
                     ext_corners[index_h1]]
    
    bool_inversed_corners = are_corners_within_bounds(int_corners, ord_ext_corners)
    if not(bool_inversed_corners):
        ord_ext_corners[1], ord_ext_corners[2], ord_ext_corners[3] = ord_ext_corners[1], ord_ext_corners[3], ord_ext_corners[2]
        ordered_corners[1], ordered_corners[2], ordered_corners[3] = ordered_corners[1], ordered_corners[3], ordered_corners[2]

    return ordered_corners, ord_ext_corners

def corners_stickers(frame):
    """
    Find the blue and the rose stickers in the frame

    :param frame: a CV2 frame
    
    :return x_rose: the x coordinate of the rose sticker
    :return y_rose: the y coordinate of the rose sticker
    :return x_blue: the x coordinate of the blue sticker
    :return y_blue: the y coordinate of the blue sticker
    :return flag: a boolean indicating if the stickers were found
    """
    
    # Define the color ranges for rose and blue stickers in HSV
    rose_lower = np.array([160, 120, 120])
    rose_upper = np.array([179, 255, 255])
    blue_lower = np.array([90, 100, 100])
    blue_upper = np.array([130, 255, 255])
    
    # Define size thresholds for stickers
    min_area = 250
    max_area = 2500
    
    # Apply a Gaussian blur to the frame
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for rose and blue colors
    rose_mask = cv2.inRange(hsv_frame, rose_lower, rose_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    
    # Morphological operations to clean the masks
    kernel = np.ones((5, 5), np.uint8)
    rose_mask = cv2.morphologyEx(rose_mask, cv2.MORPH_OPEN, kernel)
    rose_mask = cv2.morphologyEx(rose_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for the rose sticker and draw a rectangle around it
    rose_contours, _ = cv2.findContours(rose_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in rose_contours:
        
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) >= 4:
                x_rose, y_rose, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x_rose, y_rose), (x_rose + w, y_rose + h), (0, 0, 255), 2)


    # Find contours for blue stickers
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_stickers = []
    for contour in blue_contours:
        
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            if len(approx) >= 4:
                
                x_blue, y_blue, w, h = cv2.boundingRect(contour)
                blue_stickers.append((x_blue, y_blue, w, h, area))
                
    # If any value is not a value, return False
    if 'x_rose' not in locals() or 'y_rose' not in locals() or 'x_blue' not in locals() or 'y_blue' not in locals():
        
        return 0, 0, 0, 0, False
    
    # Process the blue elements to only get the blue sticker
    if len(blue_stickers) > 1 or blue_stickers[0][1] < 100:
        
        blue_stickers = [sticker for sticker in blue_stickers if sticker[0] >= 600 and sticker[1] >= 100]

        # Sort by y (largest y is the lowest sticker in the image)
        if len(blue_stickers) <= 0:
            
            return 0, 0, 0, 0, False
        
        blue_stickers.sort(key=lambda x: x[1], reverse=True)
        chosen_blue = blue_stickers[0]
        
    else:
        
        chosen_blue = blue_stickers[0]

    # Draw the blue sticker
    x_blue, y_blue, w, h, _ = chosen_blue
    cv2.rectangle(frame, (x_blue, y_blue), (x_blue + w, y_blue + h), (255, 0, 0), 2)
    
    return x_rose, y_rose, x_blue, y_blue, True

def display_corners(corners, frame):
    """
    Display the center of the 4 corners on the frame as circles:
    - a1: dark blue
    - a8: rose
    - h8: black
    - h1: white

    :param corners: the corners of the chessboard
    :param frame: the frame to display the corners on
    """
    
    # List of colors for each corner
    colors = [(139, 0, 0), # dark blue
                (255, 0, 255), # rose
                (0, 0, 0), # black
                (255, 255, 255)] # white

    # Draw the corners with different colors
    for i, point in enumerate(corners):
        
        color = colors[i]
        center = tuple(point.astype(int))
        cv2.circle(frame, center, radius = 5, color = color, thickness = -1)
    
    # cv2.imshow("corners", frame)
    # cv2.waitKey(500)
    
#*---------------------------------------------EOF-----------------------------------------------*#