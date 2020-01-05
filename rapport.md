# Apprentisssage profond par renforcement
### Thomas Boug & Théo Buttez - M2IA 2019-2020
### Université Lyon 1

------------------------------------------
## Introduction
### Dossier `cartpole`

Ici, l'environnement se compose d'un chariot qui se déplace avec un bâton posé verticalement à l'intérieur de celui-ci. L'objectif est de garder le bâton en équilibre le plus longtemps possible. 

A chaque étape, l'agent, qui contrôle le chariot, connaît la position du chariot et du bâton, ainsi que leur variation. Il faut alors choisir l'action permettant de garder le bâton en équilibre. 

Grâce à un réseau de neurones, il obtient une valeur associée à chaque action (`DROITE` et `GAUCHE`). Puis il exécute l'action correspondant à la valeur maximale. 

Nous avons choisi d'implémenter la stratégie d'exploration &epsilon;-greedy (l'action est choisie aléatoirement avec une probabilité de &epsilon;, sinon c'est l'action qui maximise la récompense qui est sélectionnée). De plus, le second réseau de neurones (celui calculant les q-valeurs de l'état suivant pour appliquer l'équation de Bellman) est mis-à-jour à intervalles réguliers par copie profonde du premier (celui calculant les q-valeurs de l'état courant et mis-à-jour via une descente de gradient).

Avec notre implémentation, notre agent apprend au bout d'une moyenne de 80 épisodes. Après apprentissage, les épisodes durent en moyenne 300 étapes, le score maximal est parfois atteint mais l'apprentissage est alors assez instable.



--------------------------------------------------
## Environnement avancé
### Dossier `atari`

Dans cette partie, on utilise un environnement plus complexe, le BreakoutNoFrameSkip_v4. Il s'agit d'un jeu Atari constitué d'une plateforme contrôlée par l'agent qui permet de faire rebondir une balle lancée dans le jeu. La balle part alors contre des murs cassables et rebondit sur ceux-ci en les brisant. Elle revient vers la plateforme qui doit la rattraper et ainsi de suite.

A chaque étape, l'agent connaît les 4 dernières frames renvoyées par le jeu. Il doit alors étudier ces frames pour placer la plateforme au bon endroit et réceptionner la balle.

Pour cela, nous utilisons les wrappers de gym : FrameStack, qui renvoie les 4 dernières frames, et AtariPreprocessing, qui grise l'image et la redimensionne en 84x84 pixels.

Avec le réseau de neurones, l'agent récupère de la même manière une valeur associée à chaque action, puis exécute l'action correspondant à la valeur maximale.
