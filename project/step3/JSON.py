import numpy as np
from project.challenge.logic import rock_recognition

def json_file(current_frame, actual_dict,new_dict,game_state0,game_state,data,time):
    """
    Fonction pour générer un fichier JSON avec les mouvements détectés et les états du jeu.
    
    Parameters:
    - current_frame : ndarray
        Frame actuelle de la vidéo.
    - actual_dict : dict
        Dictionnaire représentant l'état actuel du jeu.
    - new_dict : dict   
        Dictionnaire représentant l'état du jeu après le mouvement.
    - game_state0 : 
        Etat du jeu initial.
    - game_state :
        Etat du jeu après le mouvement.
    - data 
        Dictionnaire contenant les mouvements et les états du jeu.
    - time : datetime
        durée de la détection du mouvement
    
    Returns:    
    - data : dict
        Dictionnaire contenant les mouvements et les états du jeu   
    """
    
    detect_rock,rock = rock_recognition(game_state0, game_state, new_dict)
    # Compare les deux dictionnaires et rend le bon mouvement(->) pour la piece
    for key in new_dict.keys():
       
        if new_dict[key] != actual_dict[key]:     
        # detecte si il y eu une capture de piece
            if key not in actual_dict.keys():
                data["moves"].append({"piece": key[0] + "x" + key[1]})
                data["game_states"].append(game_state)
                data["time"].append(time)
    
    
            # detecte petit rock en calculant la differences de la distance en abs entre la tour 
    
            if rock and abs(new_dict['R'][1] - actual_dict['R'][1]) == 2:
                data["moves"].append({"piece": "O-O"})
                data["game_states"].append(game_state)
                data["time"].append(time)
            
            # detecte grand rock en calculant la differences de la distance en abs entre le roi et la tour
            
            elif rock and abs(new_dict['R'][1] - actual_dict['R'][1]) == 3:
                data["moves"].append({"piece": "O-O-O"})
                data["game_states"].append(game_state)
                data["time"].append(time)   
            
            # detecte si il y a eu un deplacement de piece
            
            else :
                data["moves"].append({"piece": key[0] + "->" + key[1]})
                data["game_states"].append(game_state)
                data["time"].append(time)
    
    # Sauvegarder les données dans un fichier JSON
    with open("game_states.json", "w") as f:
        json.dump(data, f, indent=2)
        
    return data
