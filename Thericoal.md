# Qu'est-ce qu'un transformeur ?

Un transformeur est un type de modèle de réseau de neurones introduit dans l'article "Attention is All You Need" par Vaswani et al. en 2017. Les transformeurs sont particulièrement efficaces pour les tâches de traitement du langage naturel (NLP) comme la traduction automatique, la génération de texte, et bien d'autres.

## Principes de base

Les transformeurs utilisent un mécanisme appelé "attention" pour pondérer l'importance de différentes parties de l'entrée lors de la génération de la sortie. Contrairement aux réseaux de neurones récurrents (RNN), les transformeurs n'ont pas besoin de traiter les données séquentiellement, ce qui permet un parallélisme plus efficace et des temps d'entraînement plus rapides.

### Mecanisme d'attention

#### Principe de base

L'attention permet au modèle de pondérer l'importance de chaque mot de l'entrée en fonction du mot qu'il est en train de traiter. Cela signifie que le modèle peut "attendre" plus certains mots que d'autres, en fonction de leur pertinence pour la tâche en cours.

#### Attention multi-têtes

L'attention multi-têtes consiste à appliquer plusieurs mécanismes d'attention en parallèle, chacun avec des poids différents. Les résultats sont ensuite combinés pour produire la sortie finale. Cela permet au modèle de capturer différentes relations entre les mots. En utilisant plusieurs têtes, on s'approche d'une représentation plus riche et plus expressive des données en bref on s'approche de la meme difinition (l'espérance).

#### Fit Forward

Le feed-forward est une couche de réseau de neurones qui applique une transformation linéaire suivie d'une fonction d'activation non linéaire. Cette couche est utilisée pour introduire de la non-linéarité dans le modèle et permettre l'apprentissage de fonctions plus complexes.

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

# BERT (Bidirectional Encoder Representations from Transformers)

Il s'agit d'un modèle de traitement du langage naturel (NLP) qui permet de comprendre le contexte des mots dans une phrase ainsi que le sens que ce mot a avec les mots du reste de la phrase. Il utilise une architecture type transformeur et est bidirectionnel ie au lieu de parcours la phrase de gauche a droite il va egalement la parcourir de droite a gauche pour mieux saisir le sens de la phrase. 

Il s'entraine sur 2 taches : MLM (Masked Language Modeling) et NSP (Next Sentence Prediction). Le premier cache une partie des mots de la phrase, qu'il remplace par un token [MASK] et le modèle doit deviner le mot. La deuxieme tache permet de mettre en lien des phrases entre elles et voir si le sens de la phrase qui suit permet de donner un sens different a celle actuellement traite ("Je vais au magasin. Je vais acheter des fruits", il peut alors comprendre que le magasin en question est probablement un supermarché). (si vous voulez)

# Vision Transformer (ViT) for images

L'objectif est d'utiliser les NLP comme BERT pour faire de la reconnaissance d'image. Google Research les a introduit il y a quelques années et d'un point de vue performance, les Transformers rivalisent voir peuvent surpasser les reseaux de neurones convolutionnels classique tel que ResNet.En effet, l'un des problemes lie aux CNN est qu'ils ne sont pas tres efficaces dans la comprehension globale de l'image. Ils vont reussir a detecter les textures et les contours, trouver les formes mais pas comprendre les relations entre les regions d'une image. Les Transformers peuvent se servir de leur mecanisme d'attention pour trouver ces relations. 

Fonctionnement d'un ViT
1. Decoupage de l'image : on decoupe l'image en patch qu'on transforme en vecteur de caracteristiques pour stocker les informations du patch (comme ce qu'on fait avec une phrase, on prend mot par mot sauf qu'ici on fait patch par patch)
2. Encodage de la position : La notion de position n'existe pas, on ajoute des vecteurs d'encodage positionnel pour permettre de differencier les patch et savoir quel patch est lie a quel autre patch
3. Encodeur Transformer : On considere chaque patch comme un token. On lui applique un mecanisme d'attention multi-tete pour voir sa relation avec les autres patchs pour comprendre ses relations avec les patchs qui lui sont directement lies et son sens global dans l'image.
4. Classification : On retourne les informations globales de l'image pour predire la classe de l'image


L'un des problemes lies a cette methode est que les ViT ont besoin d'une quantite de donnees assez enorme pour pouvoir surpasser un CNN. Entraines sur des petits ensembles de donnees, ils sont en sous performance par rapport aux CNN.

## Image

![transformer](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## Sources

https://arxiv.org/abs/1706.03762
https://jalammar.github.io/illustrated-transformer/
