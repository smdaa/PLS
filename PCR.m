function [BetaPCR, Y_fitted] = PCR(Y, X, k)

    % On applique l'analyse des principales composantes sur X
    [coeff, score, ~] = PCA(X);
    
    % On applique la regression linéaire sur un petit nombre de score
    Beta = score(:, 1:k) \ (Y - mean(Y));
    BetaPCR = coeff(:, 1:k) * Beta;  
    BetaPCR = [mean(Y) - mean(X)*BetaPCR; BetaPCR];
    
    Y_fitted =  [ones(size(X, 1),1) X] * BetaPCR; 
    
end

