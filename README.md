# ♟️ Chess Vision AR: Reconnaissance de pièces et Réalité Augmentée

Ce projet de vision par ordinateur vise à développer une solution en temps réel capable de reconnaître, suivre et annoter les pièces d’un échiquier, en les remplaçant par leurs images correspondantes dans un environnement en réalité augmentée. 🎥✨

---

## 📋 Table des matières

1. [Introduction](#introduction)
2. [Acquisition des données](#acquisition-des-données)
3. [Calibration de Caméra](#calibration-de-caméra)
4. [Détection des pièces](#détection-des-pièces)
5. [Reconnaissance & Mise à jour de l’état de jeu](#reconnaissance--mise-à-jour-de-letat-de-jeu)
6. [Résultats](#résultats)
7. [Contributions](#contributions)
8. [Références](#références)

---

## 📖 Introduction

Dans l'univers des échecs, la précision et la réactivité sont essentielles. Ce projet a pour objectif de proposer une solution innovante pour :  
- **Reconnaître les pièces d’échecs** en temps réel  
- **Annoter et remplacer dynamiquement** ces pièces par des images en réalité augmentée

Les principaux défis abordés incluent l’acquisition de vidéos de qualité, la calibration précise de la caméra, la détection robuste des pièces même dans des conditions variées, et l’analyse des mouvements pour mettre à jour l’état du jeu.

---

## 📹 Acquisition des données

Pour garantir une diversité de perspectives et de conditions de prise de vue, chaque membre de l’équipe a enregistré deux parties :
- **Caméra fixe** 📌 : Fournissant des vues stables et uniformes.
- **Caméra mobile** 🚀 : Introduisant des variations d’angle et de déplacement pour tester la robustesse de la solution.

Les vidéos ont été annotées avec des repères clairs pour les pièces blanches et noires et converties en séquences d’images pour une analyse approfondie.

---

## 🔧 Calibration de Caméra

Afin d’aligner le modèle idéal d’échiquier avec l’image capturée, nous avons réalisé plusieurs étapes clés :

- **Détection des coins internes**  
  Identification des coins critiques (a1, a8, h8, h1) à l’aide de techniques de détection de contours et de raffinement (ex. Harris ou Shi-Tomasi) 🔍.

- **Mapping des coordonnées**  
  Transformation des positions réelles des coins en coordonnées normalisées de l’échiquier idéal  
  (ex. (0,0) pour a1, (7,0) pour a8, etc.).

- **Calcul de l’homographie**  
  Génération d’une matrice d’homographie pour projeter la grille de l’échiquier idéal sur le plan image.  
  Cette transformation garantit une précision sub-pixel, même en cas d’angles de vue variables ou de légers déplacements de la caméra.

---

## 📷 Détection des pièces

La détection des pièces se fait en deux étapes complémentaires :

### 1. Alignement de la grille de l'échiquier
- **Coin Detection & ROI Extraction**  
  Une fois l’homographie appliquée, chaque case de l’échiquier est délimitée en tant que Region Of Interest (ROI) pour une analyse locale.

### 2. Détection et segmentation des objets
- **Segmentation en HSV**  
  Utilisation d’algorithmes de segmentation par hue, saturation et value pour isoler les objets (pièces) du fond de l’échiquier. 🎯
  
- **Détection de contours**  
  Application de méthodes comme `findContours` d’OpenCV pour extraire les contours des objets dans chaque ROI.  
  Seuls les contours respectant des critères de taille et de forme (pour éliminer les bruits) sont retenus.

---

## 🤖 Reconnaissance & Mise à jour de l’état de jeu

La reconnaissance des pièces et la gestion dynamique du plateau s’effectuent grâce au script `piece_recognition.py` qui comprend :

1. **Initialisation de l’état du jeu**  
   Attribution de valeurs par défaut (-7 pour les pièces noires et 7 pour les pièces blanches) pour les pièces indéterminées.

2. **Analyse du mouvement**  
   Comparaison des positions entre images consécutives pour déduire les déplacements sur le plateau.  
   Fonctions telles que `ascending_recognition()` et `decreasing_recognition()` ajustent l’identité des pièces en fonction des mouvements observés.

3. **Actualisation contextuelle**  
   Une fonction récursive, `context_update()`, révise l’état du plateau en appliquant les règles du jeu d’échecs.  
   Cela permet, par exemple, d’exclure une pièce déjà identifiée lors de détections ultérieures.

---

## 🏆 Résultats

- **Calibration précise**  
  La grille de l’échiquier est correctement alignée, garantissant une localisation fiable de chaque case, même en conditions de faible luminosité.
  
- **Détection robuste des pièces**  
  La segmentation HSV et la détection de contours fournissent une identification fiable dans la plupart des cas, avec quelques erreurs sur des cases éloignées ou mal éclairées.

- **Reconnaissance dynamique des mouvements**  
  La mise à jour de l’état du jeu est globalement efficace, bien que l’intégration complète, notamment l’exportation en fichier .json, reste à parfaire.

> **Note** : Chaque composant a été testé individuellement. L’intégration complète des modules fonctionne dans l’ensemble, mais des ajustements sont encore nécessaires pour améliorer certains liens, notamment la génération du fichier .json.

---

## 👥 Contributions

- **Bouchy Elliot**  
  – Organisation globale du script  
  – Calibration de base  et avancé et détection des pièces/détermination des mouvements

- **Martins Adrien**  
  – Organisation globale et élaboration de la logique de reconnaissance des pièces

- **Rion Léa**  
  – Organisation globale, calibrage avancé, transformation des frames et détection des pièces

---
