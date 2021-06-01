% SIMPLS (Statistical Inspired Modification of Partial Least Squares),
% proposé par S. de Jong. 1993

% Motivation : dériver les facteurs PLS T directement comme
% combinaisons linéaires de l'original (centré) X

% l'algorithme SIMPLS est équivalent à NIPALS lorsque Y se limmte a une
% seule variable

function BetaPLS = SIMPLS(Y, X, k)
    % On centre X et Y
    X = X - mean(X);
    Y = Y - mean(Y);

    [n, p]  = size(X);
    [~, dy] = size(Y);

    % X = TP ou T est la matrice des scores 
    %     et P est la matrice des loadings
    T = zeros(n, k);
    P = zeros(p, k);
    
    % Y = UQ ou U est la matrice des scores
    %     et Q est la matrice des loadings
    U = zeros(n, k);
    Q = zeros(dy,k);
    
    W = zeros(p, k);
    
    % pour la déflation (pour éliminer la solution déjà déterminée, tout en
    % laissant les solutions restantes inchangées) On utilise V une base
    % othonormale de P (X-loading)
    V = zeros(p, k);
    
    % On calcule la matrice de covariance 
    Cov = (X)' * Y;
    
    for i = 1:k
        % On cherche ti=X*ai et ui=Y*bi qui maximise la covariance 
        % cov(X*ai, Y*bi) = ai'*X'*Y*bi sous la contrainte d'orthonormalité 
        % ti'*tj=0 pour j=1:(i-1) et ||ti|| = 1
        
        % Décomposition en valeurs singulières
        % C'est un peu de gaspillage car nous n'avons besoin que du premier
        % vp mais la SVD est rapide donc c'est pas grave surtout avec le
        % parametre econ qui supprime les zéros dans la matrice diagonale
        % des valeurs singulières ce qui améliore le temps d'exécution et
        % réduit le stockage sans compromettre la précision.
        [A, D, B] = svd(Cov, 'econ');
        % A = les vecteurs propres de (Cov) * (Cov)'
        % B = les vecteurs propres de (Cov)' * (Cov)
        % D = matrice diagonale des valeurs propres
        
        % On calcule les loadings de X et Y (basé sur la SVD)
        ai = A(:, 1);
        d  = D(1);
        bi = B(:, 1);
        
        % On calcule le X-score
        t       = X * ai;
        tnorm   = norm(t);
        t       = t ./ tnorm; % on normalise t
        T(:, i) = t;
        
        % On calcule le X-loading
        P(:, i) = (X') * t;
        
        % On calcule le Y-loading
        q       = d*bi / tnorm;
        Q(:, i) = q;
        
        % On calcule le Y-score
        U(:, i) = Y * q;
        
        % On calcule le X-weight
        W(:, i) = ai ./ tnorm;
        
        % Déflation de la matrice Cov ie projection sur le Complément 
        % orthogonal de P (matrice des loadings de X)
        [newV, newCov] = deflate(i, V, Cov, P);
        V              = newV;
        Cov            = newCov;
        
    end
    BetaPLS = W * Q';
        
end

function [V, Cov] = deflate(i, oldV, oldCov, P) 

    V   = oldV;
    Cov = oldCov;
    vi  = P(:, i);
    
    % la base V doit etre orthogonal donc on applique Gram Schmidt modifié
    % (plus stable numériquement)
    for j = 1:i-1
        vj = V(:, j);
        vi = vi - (vi' * vj) * vj; 
    end
    vi      = vi ./ norm(vi);
    V(:, i) = vi;
    
    % supprimer les projections au long du vecteur de base actuel
    Cov = Cov - vi * (vi' * Cov); 
end