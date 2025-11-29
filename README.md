# ML-Fundamentals-Linear-Regression-From-Scratch
Fundamental Machine Learning project implementing Linear Regression (Ordinary Least Squares - OLS) from scratch in Python using NumPy. This project demonstrates a deep understanding of the mathematical theory (calculating Alpha and Beta parameters) and the ability to translate core ML concepts into code.

# 📉 Implémentation Fondamentale : Régression Linéaire (From Scratch)

## 🚀 Vue d'Ensemble du Projet

Ce dépôt contient une implémentation de la **Régression Linéaire Simple** codée entièrement à partir de zéro, sans l'utilisation de librairies de haut niveau comme Scikit-learn pour l'entraînement.

L'objectif est de démontrer une **compréhension solide des principes mathématiques** (méthode des Moindres Carrés Ordinaires - OLS) et la capacité à traduire des formules théoriques en code Python performant et vectorisé, principalement via **NumPy**.

## 🔬 Fondation Mathématique : Moindres Carrés Ordinaires (MCO)

Le modèle cherche les paramètres optimaux $\alpha$ (ordonnée à l'origine) et $\beta$ (pente) qui minimisent l'erreur quadratique de la prédiction : $\hat{y} = \alpha + \beta x$.

### 1. Calcul du Coefficient $\beta$ (Pente)

La pente est calculée comme le ratio de la covariance de $x$ et $y$ sur la variance de $x$ :

$$\beta = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

### 2. Calcul du Coefficient $\alpha$ (Ordonnée à l'Origine)

L'ordonnée à l'origine est calculée en utilisant la pente et les moyennes ($\bar{x}$ et $\bar{y}$) :

$$\alpha = \bar{y} - \beta \bar{x}$$

### 3. Implémentation Python

Les formules ci-dessus sont implémentées en utilisant des opérations **vectorielles de NumPy** (comme `np.mean()` et `np.sum()`) pour garantir la performance et montrer une bonne pratique de codage Python pour le calcul scientifique.

## 🛠️ Outils et Démonstration des Compétences

* **Langage :** Python
* **Calcul Scientifique :** **NumPy** (Utilisé pour l'implémentation vectorisée des formules OLS).
* **Gestion de Projets :** Git / GitHub (Versionnage et historique des commits).
* **Prochaine Étape MLOps :** Ce projet simple est la base idéale pour être **Dockerisé** (conteneurisé) et déployé, servant de démo "Hello World" pour les pipelines **CI/CD** et de déploiement en ML Engineering.
