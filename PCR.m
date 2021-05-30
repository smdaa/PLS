function [BetaPCR, Y_fitted] = PCR(Y, X, k)

    % On applique l'analyse des principales composantes sur X
    [coeff, score, ~] = PCA(X);
    
    % On applique la regression linéaire sur un petit nombre de score avec
    % Y0 centré
    Y0 = Y - mean(Y, 1);
    
    Beta = score(:, 1:k) \ Y0;
    BetaPCR = coeff(:, 1:k) * Beta;  
    % prendre en compte l'offset
    BetaPCR = [mean(Y) - mean(X)*BetaPCR; BetaPCR];
    
    % 0n ajoute l'intercept
    Y_fitted =  [ones(size(X, 1),1) X] * BetaPCR; 
    
end

