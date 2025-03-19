import cv2
import numpy as np
import os

def apply_homography(image_path, points_src, points_dst, output_path="transformed_image.jpg"):
    """
    Applique une transformation par homographie à une image.

    :param image_path: Chemin de l'image d'entrée.
    :param points_src: Points de l'image source (list ou array, 4 points minimum).
    :param points_dst: Points correspondants dans l'image destination (list ou array).
    :param output_path: Chemin pour sauvegarder l'image transformée.
    """
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : Impossible de charger l'image.")
        return
    
    # Vérifier que les points sont bien définis
    if len(points_src) != 4 or len(points_dst) != 4:
        print("Erreur : Les points source et destination doivent contenir exactement 4 points.")
        return

    # Calculer la matrice d'homographie
    points_src = np.array(points_src, dtype=np.float32)
    points_dst = np.array(points_dst, dtype=np.float32)
    H, _ = cv2.findHomography(points_src, points_dst)

    # Dimensions de l'image
    height, width = image.shape[:2]

    # Appliquer la transformation
    transformed_image = cv2.warpPerspective(image, H, (800, 800))

    # Sauvegarder l'image transformée
    cv2.imwrite(output_path, transformed_image)
    print(f"Image transformée sauvegardée dans : {output_path}")

    # Afficher l'image originale et transformée
    cv2.imshow("Image Originale", image)
    cv2.imshow("Image Transformée", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Obtenir le chemin du script actuel
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Chemin relatif de l'image
    frame_relative_path = './frames/frame00.png'
    frame_path = os.path.join(script_directory, frame_relative_path)

    # Vérifier si le fichier existe
    if not os.path.exists(frame_path):
        print(f"Erreur : Le fichier '{frame_relative_path}' est introuvable.")
    else:
        # Points réels détectés dans l'image (exemple arbitraire)
        real_corners = [
            [1015.76685, 747.889],
            [265.77423, 584.8124],
            [547.2637, 172.11633],
            [1105.6449, 268.26883]
        ]

        # Points idéaux dans le plan cible
        ideal_corners = [
            [0, 0],
            [800, 0],
            [800, 800],
            [0, 800]
        ]

        # Appliquer l'homographie
        apply_homography(frame_path, real_corners, ideal_corners, "transformed_image.jpg")
