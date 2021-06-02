% Analyse en composantes principales
function [coeff, score, latent] = PCA(X)
% calcul de la matrice de variance/covariance
n = size(X, 1);
Xc = X - mean(X, 1);
Sigma = (1 / n) * (Xc') * Xc;

% calcul des vecteurs/valeurs propres de la matrice Sigma
Sigma = (Sigma + Sigma.') / 2;
[V, D] = eig(Sigma);
[D, indices_tri] = sort(diag(D), 'descend');
V = V(:, indices_tri);

% coeff : les composantes principales ie vecteurs propres de la matrice de covariance de x disposées par ordre décroissant.
coeff = V;

% score : X dans la nouvelle base de composants principaux.
score = Xc * V;

% latent : les valeurs propres de la matrice de covariance de x arrangées par ordre décroissant.
latent = D;
end
