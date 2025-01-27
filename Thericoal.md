# Qu'est-ce qu'un transformeur ?

Un transformeur est un type de modèle de réseau de neurones introduit dans l'article "Attention is All You Need" par Vaswani et al. en 2017. Les transformeurs sont particulièrement efficaces pour les tâches de traitement du langage naturel (NLP) comme la traduction automatique, la génération de texte, et bien d'autres.

## Principes de base

Les transformeurs utilisent un mécanisme appelé "attention" pour pondérer l'importance de différentes parties de l'entrée lors de la génération de la sortie. Contrairement aux réseaux de neurones récurrents (RNN), les transformeurs n'ont pas besoin de traiter les données séquentiellement, ce qui permet un parallélisme plus efficace et des temps d'entraînement plus rapides.

### Mecanisme d'attention

#### Principe de base

L'attention permet au modèle de pondérer l'importance de chaque mot de l'entrée en fonction du mot qu'il est en train de traiter. Cela signifie que le modèle peut "attendre" plus certains mots que d'autres, en fonction de leur pertinence pour la tâche en cours.

#### Attention multi-têtes

L'attention multi-têtes consiste à appliquer plusieurs mécanismes d'attention en parallèle, chacun avec des poids différents. Les résultats sont ensuite combinés pour produire la sortie finale. Cela permet au modèle de capturer différentes relations entre les mots.

##### Calcul de l'attention

Le calcul de l'attention se fait en trois étapes principales :

Calcul des scores d'attention : Pour chaque mot de l'entrée, on calcule un score d'attention par rapport à tous les autres mots. Cela se fait en utilisant des vecteurs de requête (query), de clé (key) et de valeur (value).

Application d'une fonction softmax : Les scores d'attention sont normalisés à l'aide d'une fonction softmax pour obtenir des poids d'attention.

Pondération des valeurs : Les valeurs sont pondérées par les poids d'attention pour produire la sortie.

#### Formule

La formule de l'attention est la suivante :

[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V ]

où :

( Q ) est le vecteur de requête.
( K ) est le vecteur de clé.
( V ) est le vecteur de valeur.
( d_k ) est la dimension des vecteurs de clé.

##### Exemple

une projection c'est le passage d'un dimension à une autre, la plupart du temps c'est d'une dimension plus grande à une dimension plus petite.

une embeding c'est une projection mais qui evite le probleme de celle-ci qui est la perte d'information.
Donc les elements sont projeter en restant différents.

Donc pour le mot "chat" que l'on ne connait pas. On va calculer la similarité entre le mot "chat" et les mots du dictionnaire. On va donc calculer le produit scalaire entre le mot "chat" et les mots du dictionnaire.

dans ce cas la querry est le mot "chat" et les keys sont les mots du dictionnaire ansi que leur valeurs dans l'espace de projection.

## Architecture

L'architecture d'un transformeur est composée de deux parties principales :

**L'encodeur** : Il traite l'entrée et génère une représentation contextuelle.
**Le décodeur** : Il utilise cette représentation pour générer la sortie.
Chaque partie est composée de plusieurs couches de blocs d'attention et de couches feed-forward.

## Image

![transformer](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## Sources

https://arxiv.org/abs/1706.03762
https://jalammar.github.io/illustrated-transformer/
