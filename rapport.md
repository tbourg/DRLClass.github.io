# Apprentisssage profond par renforcement
### Thomas Boug & Théo Buttez - M2IA 2019-2020
### Université Lyon 1

------------------------------------------
## Introduction
### Dossier `cartpole`

Ici, l'environnement se compose d'un chariot qui se déplace avec un bâton posé verticalement à l'intérieur de celui-ci. L'objectif est de garder le bâton en équilibre le plus longtemps possible. 

A chaque étape, l'agent, qui controle le chariot, connait la position du chariot et du bâton, ainsi que leur variation. Il faut alors choisir l'action permettant de garder le bâton en équilibre. 

Grâce à un réseau de neurones, il obtient une valeur associée à chaque action (`DROITE` et `GAUCHE`). Puis il exécute l'action correspondant à la valeur maximale. 

Nous avons choisi d'implémenter la stratégie d'exploration &epsilon;-greedy (l'action est choisie aléatoirement avec une probabilité de &epsilon; sinon c'est celle qui maximise la récompense). De plus, le second réseau réseau (celui calculant les q-valeurs de l'état uivant pour appliquer l'équation de Bellman) et mis-à-jour à intervalles régulier par copie profonde du premier (celui calculant les q-valeurs de l'état courant et mis-à-jour via une descente de gradient).

Avec notre implémentation, notre agent apprend au bout d'une moyenne de 80 épiodes. Après apprentissage, les épisodes dure en moyenne 300 étapes, le score maximal est parfois atteint mais l'apprentissage est alors assez instable.



--------------------------------------------------
## Environnement avancé
### Dossier `atari`
