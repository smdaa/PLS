function [BetaPLS, Y_fitted] = PLS(Y, X, k)
    
    % On centre X et Y
    X0 = X - mean(X, 1);
    Y0 = Y - mean(Y, 1);
    
    % On calcule la matrice de covariance 
    Cov = (X0)' * Y0;
    
    % Predictor loadings
    XL = zeros(size(X, 2), k);
    % Response loadings
    YL = zeros(size(Y, 2), k);
    % PLS weights
    W = zeros(size(X, 2), k);
    
    % An orthonormal basis for the span of the X loadings, to make the successive
    % deflation X0'*Y0 simple - each new basis vector can be removed from Cov
    % separately.
    V = zeros(size(X, 2), k);
    
    for i = 1:k        
        % On cherche ti=X0*ri et ui=Y0*ci qui maximise la covariance
        % ri'*X0'*Y0*ci en utilisant la décomposition svd
        [ri,si,ci] = svd(Cov, 'econ');
        ri = ri(:,1);
        ci = ci(:,1);
        si = si(1);
        
        ti = X0 * ri;
        normti = norm(ti);
        % ti'*ti == 1
        ti = ti ./ normti; 
        XL(:,i) = X0' * ti;
        
        qi = si * ci / normti; % = Y0'*ti
        YL(:,i) = qi;
        
        W(:,i) = ri ./ normti;
        
        
        % Update the orthonormal basis with modified Gram Schmidt (more stable),
        % repeated twice (ditto).
        vi = XL(:,i);
        for repeat = 1:2
           for j = 1:i-1
              vj = V(:,j);
              vi = vi - (vj'*vi)*vj;
           end
        end
        vi = vi ./ norm(vi);
        V(:,i) = vi;
        
        % Deflate Cov, i.e. project onto the ortho-complement of the X loadings.
        % First remove projections along the current basis vector, then remove any
        % component along previous basis vectors that's crept in as noise from
        % previous deflations.
        Cov = Cov - vi * (vi' * Cov);
        Vi = V(:,1:i);
        Cov = Cov - Vi*(Vi'*Cov);
           
    end
    
    BetaPLS = W * YL';
    BetaPLS = [mean(Y, 1) - mean(X, 1) * BetaPLS; BetaPLS];
    
    Y_fitted = [ones(size(X, 1),1) X] * BetaPLS;
    
end
