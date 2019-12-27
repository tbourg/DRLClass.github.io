# Apprentisssage profond par renforcement
### Thomas Boug & Théo Buttez - M2IA 2019-2020
### Université Lyon 1

------------------------------------------
## Introduction
### Dossier `cartpole`

Ici, l'environnement se compose d'un chariot qui se déplace avec un bâton posé verticalement à l'intérieur de celui-ci. L'objectif est de garder le bâton en équilibre le plus longtemps possible. 

A chaque étape, l'agent, qui controle le chariot, connait la position du chariot et du bâton, ainsi que leur variation. Il faut alors choisir l'action permettant de garder le bâton en équilibre. 

Grâce à un réseau de neurones, il obtient une valeur associée à chaque action (`DROITE` et `GAUCHE`). Puis il exécute l'action correspondant à la valeur maximale. 



--------------------------------------------------
## Environnement avancé
### Dossier `atari`
