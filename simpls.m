function BetaPLS = simpls(Y, X, k)
%           X = TP
%           Y = UQ
%           k = nComponents

    % On centre X et Y
    X = X - mean(X);
    Y = Y - mean(Y);

    [nSamples, nFeatures] = size(X);
    [~, nDimY]            = size(Y);

    T = zeros(nSamples,  k);
    U = zeros(nSamples,  k);
    P = zeros(nFeatures, k);
    Q = zeros(nDimY,     k);
    W = zeros(nFeatures, k);
    
    % pour la déflation (pour éliminer la solution déjà déterminée, tout en
    % laissant les solutions restantes inchangées)
    V = zeros(nFeatures, k);
    
    % On calcule la matrice de covariance 
    Cov = (X)' * Y;
    
    for i = 1:k
        % singular value decomposition
        [A, B, C] = svd(Cov, 'econ');
        %A = les vecteurs propres de (Cov) * (Cov)'
        %C = les vecteurs propres de (Cov)' * (Cov)
        %B = matrice diagonale des valeurs propres
        
        % On calcule les loadings de X et Y (basé sur la SVD)
        a = A(:, 1);
        b = B(1);
        c = C(:, 1);
        
        % On calcule les X-scores
        t       = X * a;
        tnorm   = norm(t);
        t       = t ./ tnorm; % on normalise t
        T(:, i) = t;
        
        % On calcule les X-loadings
        P(:, i) = (X') * t;
        
        % On calcule les Y-loadings
        q       = b*c / tnorm;
        Q(:, i) = q;
        
        % On calcule les Y-scores
        U(:, i) = Y * q;
        
        % On calcule les X-weights
        W(:, i) = a ./ tnorm;
        
        % deflation
        [dV, dCov] = deflateXtY(i, V, Cov, P);
        V          = dV;
        Cov        = dCov;
        
    end
    
    BetaPLS = W * Q';
        
end

function [V, Cov] = deflateXtY(i, oldV, oldCov, P) 

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