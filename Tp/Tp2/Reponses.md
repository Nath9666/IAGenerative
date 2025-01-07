# TP1 Advanced Generative Adversarial Networks (GANs) 

## Part 1:  CNN-based GAN

### Question 1: What is Transpose Convolution and why do we use it Generator?

La convolution transposée, également connue sous le nom de déconvolution ou de convolution à pas fractionnaire, est une opération qui effectue l'inverse d'une convolution régulière. Elle est utilisée dans le générateur d'un GAN pour suréchantillonner le vecteur de bruit d'entrée afin de générer des images de plus haute résolution. L'opération de convolution transposée applique un noyau apprenable à la carte de caractéristiques d'entrée, ce qui augmente effectivement les dimensions spatiales des données. Cela permet au générateur de transformer des cartes de caractéristiques de basse résolution en images de haute résolution en apprenant le processus de suréchantillonnage de l'espace latent à l'espace de sortie.

### Question 2: What are LeakyReLU and sigmoid and why do we use them?

LeakyReLU est une fonction d'activation qui permet un petit gradient négatif lorsque l'entrée est négative, contrairement à ReLU qui a un gradient nul pour les valeurs négatives. Cela aide à prévenir le problème de la disparition du gradient et à accélérer la convergence du modèle. LeakyReLU est couramment utilisé dans les réseaux de neurones convolutifs pour introduire une non-linéarité et favoriser la stabilité de l'entraînement.