from datetime import datetime
import json
import numpy as np
from logic import rock_recognition  # Assurez-vous que cette fonction existe

#--------------------------------------------Functions--------------------------------------------#

def json_file(data, new_dict, current_frame, game_state0, game_state, time):
    """
    Fonction pour générer un fichier JSON avec les mouvements détectés et les états du jeu.
    """
    _, rock = rock_recognition(game_state0, game_state, new_dict)
    print(_, rock)
    
    # Conversion de `ndarray` en liste pour éviter les problèmes de sérialisation
    current_frame = current_frame.tolist() if isinstance(current_frame, np.ndarray) else current_frame
    game_state = game_state.tolist() if isinstance(game_state, np.ndarray) else game_state
    
    game_state_entry = {
        "frame": current_frame,
        "gs": game_state
    }
    
    for key, value in new_dict.items():
        status, old_coords, new_coords, _ = value
        if new_coords != old_coords:
            if new_coords not in [piece[1] for piece in new_dict.values()]:
                data["moves"].append(f"{old_coords}x{new_coords}")
                data["game_states"].append(game_state_entry)
                data["time"].append(time)

            if not rock and abs(new_coords[1] - old_coords[1]) == 2:
                data["moves"].append(f"{old_coords}O-O{new_coords}")
                data["game_states"].append(game_state_entry)
                data["time"].append(time)

            elif not rock and abs(new_coords[1] - old_coords[1]) == 3:
                data["moves"].append(f"{old_coords}O-O-O{new_coords}")
                data["game_states"].append(game_state_entry)
                data["time"].append(time)

            else:
                data["moves"].append(f"{old_coords}->{new_coords}")
                data["game_states"].append(game_state_entry)
                data["time"].append(time)

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
    Met à jour le fichier JSON pour convertir les mouvements et restructurer les données selon le format demandé.
    """
    with open(data, 'r') as file:
        json_data = json.load(file)

    piece_names = {
        1: "P", 2: "R", 3: "N", 4: "B", 5: "K", 6: "Q",
        -1: "p", -2: "r", -3: "n", -4: "b", -5: "k", -6: "q",
        7: "unknown", -7: "unknown"
    }

    transformed_moves = []
    for move in json_data.get("moves", []):
        for key, value in move.items():
            piece_id = int(key)
            piece_name = piece_names.get(piece_id, "")
            if "x" in value:
                start, target = value.split("x")
                start = eval(start)
                target = eval(target)
                transformed_moves.append(f"{piece_name}{chess_notation_converter(start)} x {chess_notation_converter(target)}")
            elif "->" in value:
                start, target = value.split("->")
                start = eval(start)
                target = eval(target)
                transformed_moves.append(f"{piece_name}{chess_notation_converter(start)} -> {chess_notation_converter(target)}")
            else:
                transformed_moves.append(value)

    json_data["moves"] = transformed_moves

    with open(data, 'w') as file:
        json.dump(json_data, file, indent=4)


def avg_time(data):
    """
    Calcule le temps moyen de détection d'un mouvement.
    """
    times = data.get("time", [])
    if len(times) < 2:
        raise ValueError("Il y a moins de deux horodatages. Impossible de calculer le temps moyen.")

    time_objects = [datetime.fromisoformat(time) for time in times]

    time_diffs = [time_objects[i] - time_objects[i - 1] for i in range(1, len(time_objects))]
    avg_time = sum(diff.total_seconds() for diff in time_diffs) / len(time_diffs)

    print(f"The average time between each move is {avg_time:.2f} seconds.")

# Fonction principale pour tester
def test_chess_game():
    # Charger les données JSON
    with open('D:\school\computer_vision\projectCV_chess\project\challenge\data.json', 'r') as f:
        data = json.load(f)
    
    # Initialiser les données
    current_frame = 0
    game_state0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    game_state = game_state0.copy()
    
    # Appeler la fonction de mise à jour du JSON avec les mouvements détectés
    data = json_file(data, data["game_states"], current_frame, game_state0, game_state, datetime.now().isoformat())
    
    # Mettre à jour le fichier JSON avec les nouveaux mouvements
    update_JSON('data.json')
    
    # Calculer le temps moyen
    avg_time(data)

# Lancer le test
test_chess_game()
