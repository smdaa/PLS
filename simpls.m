% SIMPLS (Statistical Inspired Modification of Partial Least Squares),
% proposé par S. de Jong. 1993

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
    % laissant les solutions restantes inchangées)
    V = zeros(p, k);
    
    % On calcule la matrice de covariance 
    Cov = (X)' * Y;
    
    for i = 1:k
        % On cherche ti=X*ai et ui=Y*bi qui maximise la covariance 
        % cov(X*ai, Y*bi) = ai'*X'*Y*bi sous la contrainte d'orthonormalité 
        % ti'*tj=0 pour j=1:(i-1) et ||ti|| = 1
        
        % Décomposition en valeurs singulières
        [A, D, B] = svd(Cov, 'econ');
        % A = les vecteurs propres de (Cov) * (Cov)'
        % B = les vecteurs propres de (Cov)' * (Cov)
        % D = matrice diagonale des valeurs propres
        
        % On calcule les loadings de X et Y (basé sur la SVD)
        ai = A(:, 1);
        d = D(1);
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
        
        % deflation
        [dV, dCov] = deflate(i, V, Cov, P);
        V          = dV;
        Cov        = dCov;
        
    end
    
    BetaPLS = W * Q';
        
end

function [V, Cov] = deflate(i, oldV, oldCov, P) 

    V   = oldV;
    Cov = oldCov;
    vi  = P(:, i);
    
    for repeat = 1:2
       for j = 1:i-1
          vj = V(:, j);
          vi = vi - (vi' * vj) * vj; 
       end
    end
    
    vi               = vi ./ norm(vi);
    V(:, i) = vi;
    
    Cov = Cov - vi * (vi' * Cov); 
    Vi  = V(:, 1:i);
    Cov = Cov - Vi * (Vi' * Cov);
end