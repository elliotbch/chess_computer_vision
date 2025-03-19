#*--------------------------------------------Imports--------------------------------------------*#

from datetime import datetime
from logic import rock_recognition
import numpy as np


def json_file(data, new_dict, current_frame, game_state0, game_state, time):
    """
    Fonction pour générer un fichier JSON avec les mouvements détectés et les états du jeu.
    """
    _, rock = rock_recognition(game_state0, game_state, new_dict)
    
    # Conversion de `ndarray` en liste pour éviter les problèmes de sérialisation
    current_frame = current_frame.tolist() if isinstance(current_frame, np.ndarray) else current_frame
    game_state = game_state.tolist() if isinstance(game_state, np.ndarray) else game_state
    
    game_state_entry = {
        "frame": current_frame,
        "gs": game_state
    }
    
    # Convertir `time` en format ISO 8601 si ce n'est pas déjà le cas
    if isinstance(time, datetime):
        time_iso = time.isoformat()
    elif isinstance(time, str):
        try:
            # Vérifie si la chaîne est au format ISO 8601
            datetime.fromisoformat(time)
            time_iso = time
        except ValueError:
            # Si ce n'est pas le cas, convertir en datetime puis en ISO 8601
            time_iso = datetime.fromtimestamp(float(time)).isoformat()
    else:
        # Si `time` est un nombre, le convertir en datetime puis en ISO 8601
        time_iso = datetime.fromtimestamp(float(time)).isoformat()
    
    # Créer un dictionnaire des positions des pièces avant le mouvement
    positions_before = {piece[1]: key for key, piece in new_dict.items()}
    
    for key, value in new_dict.items():
        status, new_coords, old_coords, _ = value
        if old_coords == (-1, -1):
            continue  # Passer à la pièce suivante si les anciennes coordonnées sont (-1, -1)
        print(key, old_coords, new_coords)
        if old_coords != new_coords:
            if new_coords in positions_before and positions_before[new_coords] != key:
                move_str = f"{status}: ({old_coords})x({new_coords})"
            elif not rock and abs(new_coords[0] - old_coords[0]) == 2 and abs(new_coords[1] - old_coords[1]) == 0:
                move_str = f"{status}: ({old_coords})O-O({new_coords})"
            elif not rock and abs(new_coords[0] - old_coords[0]) == 3 and abs(new_coords[1] - old_coords[1]) == 0:
                move_str = f"{status}: ({old_coords})O-O-O({new_coords})"
            else:
                move_str = f"{status}: ({old_coords})->({new_coords})"
            
            # Vérification des doublons avant d'ajouter le mouvement
            if move_str not in data["moves"]:
                data["moves"].append(move_str)
                data["game_states"].append(game_state_entry)
                data["time"].append(time_iso)
                print('data["moves"]', data["moves"])
            break

    return data


def chess_notation_converter(position):
    """
    Convertit une position sous forme de coordonnées (ligne, colonne) en notation échiquier (e.g., 'g1').
    """
    columns = "abcdefgh"
    row, col = position
    return f"{columns[col]}{8 - row}"

def update_JSON(data):
    """
    Met à jour le dictionnaire pour convertir les mouvements et restructurer les données selon le format demandé.
    """
    piece_names = {
        1: "P", 2: "R", 3: "N", 4: "B", 5: "K", 6: "Q",
        -1: "p", -2: "r", -3: "n", -4: "b", -5: "k", -6: "q",
        7: "unknown", -7: "unknown"
    }

    transformed_moves = []
    for move in data.get("moves", []):
        parts = move.split(": ")
        if len(parts) == 2:
            status = int(parts[0])
            move_str = parts[1]
            piece_name = piece_names.get(status, "unknown")
            color = "W" if status > 0 else "B"
            if "x" in move_str:
                start, target = move_str.split("x")
                start = eval(start)
                target = eval(target)
                if piece_name.lower() == "p":
                    transformed_moves.append(f"{chess_notation_converter(start)}x{chess_notation_converter(target)}")
                else:
                    transformed_moves.append(f"{piece_name}{chess_notation_converter(start)}x{piece_name}{chess_notation_converter(target)}")
            elif "O-O-O" in move_str:
                start, target = move_str.split("O-O-O")
                start = eval(start)
                target = eval(target)
                transformed_moves.append(f"{color}O-O-O")
            elif "O-O" in move_str:
                start, target = move_str.split("O-O")
                start = eval(start)
                target = eval(target)
                transformed_moves.append(f"{color}O-O")
            else:
                start, target = move_str.split("->")
                start = eval(start)
                target = eval(target)
                if piece_name.lower() == "p":
                    transformed_moves.append(f"{chess_notation_converter(start)}->{chess_notation_converter(target)}")
                else:
                    transformed_moves.append(f"{piece_name}{chess_notation_converter(start)}->{piece_name}{chess_notation_converter(target)}")
        else:
            transformed_moves.append(move)

    data["moves"] = transformed_moves
    return data

def avg_time(data):
    """
    Calcule le temps moyen de détection d'un mouvement.
    """
    times = data.get("time", [])
    if len(times) < 2:
        raise ValueError("Il y a moins de deux horodatages. Impossible de calculer le temps moyen.")

    time_objects = []
    for time in times:
        try:
            time_objects.append(datetime.fromisoformat(time))
        except ValueError:
            print(f"Invalid time format: {time}")
            continue

    if len(time_objects) < 2:
        raise ValueError("Il y a moins de deux horodatages valides. Impossible de calculer le temps moyen.")

    time_diffs = [time_objects[i] - time_objects[i - 1] for i in range(1, len(time_objects))]
    avg_time = sum(diff.total_seconds() for diff in time_diffs) / len(time_diffs)

    print(f"The average time between each move is {avg_time:.2f} seconds.")
    
def avg_frame(data):
    """
    Calcule le nombre moyen de frames entre chaque mouvement.
    """
    frames = [entry["frame"] for entry in data.get("game_states", [])]
    if len(frames) < 2:
        raise ValueError("Il y a moins de deux frames. Impossible de calculer le nombre moyen de frames.")

    frame_diffs = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
    avg_frame = sum(frame_diffs) / len(frame_diffs)

    print(f"The average number of frames between each move is {avg_frame:.2f}.")
