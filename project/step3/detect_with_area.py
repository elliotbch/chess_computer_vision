import cv2
import numpy as np
from datetime import datetime
import os
from calib import corners_calibration

def find_chessboard_corners(frame, real_corners, grid_size=(9, 9)):
    """
    Find all chessboard corners using a homography matrix.
    """
    ideal_points = np.array([[j, i] for i in range(grid_size[0]) for j in range(grid_size[1])], dtype=np.float32)
    
    ideal_corners = np.array([
        ideal_points[0],                                # Top-left (0, 0)
        ideal_points[grid_size[1] - 1],                # Top-right (8, 0)
        ideal_points[-1],                              # Bottom-right (8, 8)
        ideal_points[(grid_size[0] - 1) * grid_size[1]] # Bottom-left (0, 8)
    ], dtype=np.float32)

    real_corners = real_corners.reshape(-1, 2)
    homography, _ = cv2.findHomography(ideal_corners, real_corners)

    ideal_points = ideal_points.reshape(-1, 1, 2)
    real_grid = cv2.perspectiveTransform(ideal_points, homography)
    real_grid = real_grid.reshape(grid_size[0], grid_size[1], 2)
    
    # Visualize detected corners
    debug_frame = frame.copy()
    for row in real_grid:
        for point in row:
            x, y = int(point[0]), int(point[1])
            cv2.circle(debug_frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    cv2.imshow("Chessboard Corners", debug_frame)
    cv2.waitKey(0)
    
    return real_grid


def detect_piece_by_pixel_count(frame, corners, grid_size=(8, 8), white_threshold=0.1, black_threshold=0.9):
    real_grid = find_chessboard_corners(frame, corners)
    game_state = np.zeros((8, 8), dtype=int)

    for row in range(8):
        for col in range(8):
            pt1 = real_grid[row][col]
            pt2 = real_grid[row][col+1]
            pt3 = real_grid[row+1][col+1]
            pt4 = real_grid[row+1][col]

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            roi = cv2.bitwise_and(frame, frame, mask=mask)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            total_pixels = np.sum(mask == 255)
            white_pixels = np.sum(gray_roi[mask == 255] > 160)
            white_percentage = white_pixels / total_pixels
            
            print (white_pixels, total_pixels, white_percentage)
            is_white_square = (row + col) % 2 == 0
            if is_white_square:
                piece_present = white_percentage < white_threshold
            else:
                piece_present = white_percentage < black_threshold

            if piece_present:
                game_state[row][col] = 7
            cv2.imshow(f'Square {row},{col}', roi)
            cv2.waitKey(0)
    return game_state

def pieces_initial(frame, rose_pos, blue_pos, corners, original_frame):
    """
    Determine the initial game state from a frame
    """
    print("initializing...")
    is_initialized = False
    
    # Detect pieces using pixel count method
    game_state = detect_piece_by_pixel_count(frame, corners)
    
    # Check if all 32 pieces are detected
    pieces_count = np.count_nonzero(game_state == 7)
    is_initialized = pieces_count == 32
    
    print(game_state)
    return is_initialized, game_state

def process_frame(frame):
    original_frame = frame.copy()
    is_calibrate, rose_pos, blue_pos, corners = corners_calibration(frame)
    print("is_calibrate = ", is_calibrate)
    is_initialized, game_state = pieces_initial(frame, rose_pos, blue_pos, corners, original_frame)
    print("is_initialized = ", is_initialized)

def main():
    """
    Main function.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    frame_path = os.path.join(script_directory, './frames/frame05.png')
    
    frame = cv2.imread(frame_path)
    if frame is None:
        print("Error: Could not load image")
        return
    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_states_file = os.path.join(script_directory, f"states_{timestamp}.txt")
    
    process_frame(frame)

if __name__ == "__main__":
    main()