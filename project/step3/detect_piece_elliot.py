import cv2
import numpy as np
from datetime import datetime
import os
from calib import corners_calibration

class ChessboardDetector:
    def __init__(self, grid_size=(9, 9)):
        self.grid_size = grid_size
        self.piece_area_range = (400, 2000)  # Ajusté pour les post-its
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.debug_mode = True

    def enhance_image(self, img):
        """Améliore le contraste et la qualité de l'image."""
        # Conversion en LAB pour meilleure gestion des couleurs
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img)
        cl = self.clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        
        # Réduction du bruit
        denoised = cv2.fastNlMeansDenoisingColored(enhanced)
        return cv2.GaussianBlur(denoised, (3, 3), 0)

    def detect_piece_by_color(self, roi):
        """Détecte une pièce en utilisant la saturation et la teinte."""
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Seuils pour les post-its jaunes
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        
        # Seuils pour les post-its verts
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        
        # Combinaison des masques
        combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
        
        if self.debug_mode:
            cv2.imshow('Color Masks', combined_mask)
            cv2.waitKey(1)
        
        # Morphologie pour nettoyer le masque
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.piece_area_range[0] < area < self.piece_area_range[1]:
                # Vérification de la forme (approximativement carrée)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.7 < aspect_ratio < 1.3:  # Tolérance pour la forme carrée
                    return True
        return False

    def process_square(self, frame, corners, row, col, game_state):
        """Traite une case individuelle de l'échiquier."""
        try:
            pts = self.get_square_corners(corners, row, col)
            roi = self.extract_roi(frame, pts)
            
            if roi is None or roi.size == 0:
                return
            
            # Amélioration de l'image
            enhanced_roi = self.enhance_image(roi)
            
            # Détection uniquement par couleur (plus adaptée aux post-its)
            if self.detect_piece_by_color(enhanced_roi):
                game_state[row][col] = 7
                if self.debug_mode:
                    print(f"Post-it detected at position ({row}, {col})")
        
        except Exception as e:
            print(f"Error processing square ({row}, {col}): {str(e)}")

    # Le reste des méthodes reste inchangé...
    def find_chessboard_corners(self, frame, real_corners):
        """Trouve tous les coins du plateau d'échecs en utilisant une homographie."""
        ideal_points = np.array([[j, i] for i in range(self.grid_size[0]) 
                               for j in range(self.grid_size[1])], dtype=np.float32)
        
        ideal_corners = np.array([
            ideal_points[0],
            ideal_points[self.grid_size[1] - 1],
            ideal_points[-1],
            ideal_points[(self.grid_size[0] - 1) * self.grid_size[1]]
        ], dtype=np.float32)

        real_corners = real_corners.reshape(-1, 2)
        homography, _ = cv2.findHomography(ideal_corners, real_corners)

        ideal_points = ideal_points.reshape(-1, 1, 2)
        real_grid = cv2.perspectiveTransform(ideal_points, homography)
        return real_grid.reshape(self.grid_size[0], self.grid_size[1], 2)

    def get_square_corners(self, corners, row, col):
        """Obtient les coins d'une case spécifique."""
        pt1 = corners[row][col]
        pt2 = corners[row][col+1]
        pt3 = corners[row+1][col+1]
        pt4 = corners[row+1][col]
        return np.array([pt1, pt2, pt3, pt4], dtype=np.int32).reshape((-1, 1, 2))

    def extract_roi(self, frame, pts):
        """Extrait la région d'intérêt pour une case."""
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        roi = cv2.bitwise_and(frame, mask)
        
        x, y, w, h = cv2.boundingRect(pts)
        return roi[y:y+h, x:x+w]

def process_frame(frame):
    """Traite une frame pour détecter l'échiquier et les pièces."""
    try:
        detector = ChessboardDetector()
        original_frame = frame.copy()
        
        print("Starting calibration...")
        is_calibrate, rose_pos, blue_pos, corners = corners_calibration(frame)
        if not is_calibrate:
            print("Calibration failed!")
            return False, None
        
        print("Calibration successful!")
        game_state = np.zeros((8, 8), dtype=int)
        real_grid = detector.find_chessboard_corners(frame, corners)
        
        for row in range(8):
            for col in range(8):
                detector.process_square(original_frame, real_grid, row, col, game_state)
        
        return True, game_state
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return False, None

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    frame_path = os.path.join(script_directory, './frames/frame05.png')
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print("Erreur: Impossible de charger l'image")
    
    
    success, game_state = process_frame(frame)
    if success:
        print("État du jeu détecté:")
        print(game_state)