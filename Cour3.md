# Cour 3

pour faire une generation de phrase il suffit de traduire la phrase en vecteur.
pour cela on utilise un modele de langage qui est un modele de reseau de neurone.

Si on fesait une analyse de frequence, on pert l'ordre des mots, et donc on perd le sens de la phrase.
En bref on peu faire plusieur phrase avec un ensemble de mot.

Cette analyse se nomme TF/IDF.

# RNN (Recurrent Neural Network)

C'est un reseau de neurone qui a une memoire, il se souvient des mots precedents.
Il est donc capable de generer des phrases.
pour faire ça on doit mettre les mots en vecteur, et les mots sont mis dans un espace vectoriel.
avec les tecnique de world to vecteur

Mais il faut aussi comprendre le mot dans le contexte des autres mots.

# Comprendre les relations entre les mots avec le mécanisme d'attention

Le mécanisme d'attention est une technique utilisée dans les réseaux de neurones, en particulier dans les modèles de traitement du langage naturel (NLP), pour mieux comprendre les relations entre les mots dans une phrase. Voici comment cela fonctionne :

1. **Focus sur les mots importants :** Le mécanisme d'attention permet au modèle de se concentrer sur les mots les plus pertinents dans une phrase lorsqu'il génère une réponse ou effectue une tâche. Par exemple, lorsqu'on traduit une phrase, certains mots de la phrase source sont plus importants que d'autres pour traduire un mot spécifique de la phrase cible.

2. **Calcul des poids d'attention :** Pour chaque mot de la phrase, le modèle calcule un poids d'attention qui indique l'importance de chaque mot par rapport aux autres mots. Ces poids sont utilisés pour créer une représentation pondérée de la phrase, où les mots importants ont plus d'influence.

3. **Contextualisation des mots :** En utilisant les poids d'attention, le modèle peut créer des représentations contextuelles des mots, ce qui signifie qu'il comprend chaque mot en fonction des mots qui l'entourent. Cela permet de capturer les relations complexes entre les mots et de mieux comprendre le sens global de la phrase.

4. **Applications :** Le mécanisme d'attention est utilisé dans de nombreux modèles avancés de NLP, comme les Transformers, BERT et GPT. Il améliore les performances des tâches telles que la traduction automatique, la génération de texte, la réponse aux questions et bien d'autres.

avec la phrase: "on est mardi"

**La formule pour comprendre mardi**

comprendre(mardi) = somme( poids(mardi, mot) \* vecteur(mot) ) pour tous les mots de la phrase

Le poids se calcule simplement avec un produit scalaire entre le vecteur du mot et un vecteur de poids.

**Exemple complet :**

1. **Vecteurs des mots :**

   - vecteur(on) = [0.1, 0.2, 0.3]
   - vecteur(est) = [0.4, 0.5, 0.6]
   - vecteur(mardi) = [0.7, 0.8, 0.9]

2. **Vecteur de poids pour "mardi" :**

   - vecteur_poids(mardi) = [0.2, 0.3, 0.4]

3. **Calcul des poids d'attention :**

   - poids(mardi, on) = produit*scalaire(vecteur(on), vecteur_poids(mardi))
     = (0.1 * 0.2) + (0.2 \_ 0.3) + (0.3 \* 0.4)
     = 0.02 + 0.06 + 0.12
     = 0.20

   - poids(mardi, est) = produit*scalaire(vecteur(est), vecteur_poids(mardi))
     = (0.4 * 0.2) + (0.5 \_ 0.3) + (0.6 \* 0.4)
     = 0.08 + 0.15 + 0.24
     = 0.47

   - poids(mardi, mardi) = produit*scalaire(vecteur(mardi), vecteur_poids(mardi))
     = (0.7 * 0.2) + (0.8 \_ 0.3) + (0.9 \* 0.4)
     = 0.14 + 0.24 + 0.36
     = 0.74

4. **Calcul de la représentation contextuelle de "mardi" :**
   - comprendre(mardi) = (poids(mardi, on) _ vecteur(on)) + (poids(mardi, est) _ vecteur(est)) + (poids(mardi, mardi) \* vecteur(mardi))
   - comprendre(mardi) = (0.20 _ [0.1, 0.2, 0.3]) + (0.47 _ [0.4, 0.5, 0.6]) + (0.74 \* [0.7, 0.8, 0.9])
   - comprendre(mardi) = [0.02, 0.04, 0.06] + [0.188, 0.235, 0.282] + [0.518, 0.592, 0.666]
   - comprendre(mardi) = [0.726, 0.867, 1.008]

Ainsi, la représentation contextuelle de "mardi" dans la phrase "on est mardi" est le vecteur [0.726, 0.867, 1.008], qui capture les relations entre "mardi" et les autres mots de la phrase.

## "il ne pleut pas"

### **La formule pour comprendre pleut**

comprendre(pleut) = [il, pleut]*pae(il) + [ne, pleut]*pae(ne) + [pas, pleut]*pae(pas) + [pleut, pleut]*pae(pleut)
comprendre(pas) = [il, pas]*pae(il) + [ne, pas]*pae(ne) + [pas, pas]*pae(pas) + [pleut, pas]*pae(pleut)