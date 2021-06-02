%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Présentation Algèbre linéaire pour le data mining      %
%                                                          %
%   Sujet Tp : Une comparaison entre PLS et PCR sur        %
%              un jeu de donnée de Spectroscopie           %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialisation

clc;
clear;
close all;

% On charge un jeu de donnée comprenant les intensités spectrales de 60
% échantillons d'essence à 401 longueurs d'onde
load spectra

% On dispose de 60 type de gasoline (individus)
% NIR    : Spectroscopie dans l'infrarouge proche de 900 nm à 1700 nm
%          (60 individus 401 attributs) ce sont les prédicteurs
%
% octane : Indice d'octane (mesure la résistance à l'auto-allumage du carburant)
%          c'est le prédictand
Y = octane;
X = NIR;

partition = 50;

Y_train = Y(1 : partition);
X_train = X(1 : partition, :);

Y_test = Y(partition+1 : end);
X_test = X(partition+1 : end, :);

%% Visualisation des données

figure(1)
plot3(repmat(1:401, 60, 1)', repmat((1:size(octane, 1))', 1, 401)', NIR');
xlabel("indice de longueur d'onde");
ylabel("gasoline");
title('Visualisation des données');
% On a un grand nombre de variables prédictives fortement corrélés

%% Régression linéaire multiple

Beta_MLR = X_train \ Y_train;
Y_fitted_MLR = X_train * Beta_MLR;
R_2 = R_squared(Y_train, Y_fitted_MLR);
fprintf('MLR : R^2 = %.6f\n', R_2);
% On obtient un R^2 = 1.0 ce qui est un mauvais signe en effet p > n
% donc le problème de moindre carré n'admet pas une unique solution

% Conclusion : si nous avons trop de caractéristiques le modèle s'adaptera
%              très bien aux données d'apprentissage mais ne parviendra pas
%              à généraliser.

%% Régression sur composantes principales

% On réduit la dimenssion en utilisant l'analyse en composantes principales
% Par Construction les Composantes principales sont non-corrélées
k = 2;

% Pourcentage de variance expliqué en X
figure(2);
[~, ~, latent] = PCA(X_train);
latent = latent(1:10, 1);
plot(1:length(latent), sort(latent./sum(latent), 'descend'), '-o');
title('Pourcentage de variance expliqué en X');
xlabel('num de la comp. ppale');
ylabel('pourcentage de variance');

[Beta_PCR, Y_fitted_PCR] = PCR(Y_train, X_train, k);
R_2 = R_squared(Y_train, Y_fitted_PCR);
fprintf('PCR : R^2 = %.6f\n', R_2);

% On obtient une faible valeur prédictive de 0.21
% On remarque que nos premiers PC capturent 85% de la variance, les 15% restants apparaissent être important. pour prédire Y

%% Régression des moindres carrés partiels

[Beta_PLS, Y_fitted_PLS] = PLS(Y_train, X_train, k);
R_2 = R_squared(Y_train, Y_fitted_PLS);
fprintf('PLS : R^2 = %.6f\n', R_2);
fprintf('---------------------------------------\n');

%% PLS vs PCR

% Comparaison sur les données d'apprentissage
figure(3);
plot(Y_train, Y_fitted_PLS, 'bo', Y_train, Y_fitted_PCR, 'r^', Y_train, Y_fitted_MLR, 'c*');
xlabel('Observed Response train');
ylabel('Fitted Response train');
title("Comparaison sur les données d'apprentissage")
legend({'PLS1 avec 2 PC', 'PCR with 2 PC', 'MLR'})

% Comparaison sur les données de test
Y_fitted_MLR_test = X_test * Beta_MLR;
Y_fitted_PCR_test = [ones(size(X_test, 1), 1), X_test] * Beta_PCR;
Y_fitted_PLS_test = [ones(size(X_test, 1), 1), X_test] * Beta_PLS;

RMSE_MLR = RMSE(Y_test, Y_fitted_MLR_test);
RMSE_PCR = RMSE(Y_test, Y_fitted_PCR_test);
RMSE_PLS = RMSE(Y_test, Y_fitted_PLS_test);

fprintf('MLR : RMSE = %.6f\n', RMSE_MLR);
fprintf('PCR : RMSE = %.6f\n', RMSE_PCR);
fprintf('PLS : RMSE = %.6f\n', RMSE_PLS);

figure(4);
plot(Y_test, Y_fitted_PLS_test, 'bo', Y_test, Y_fitted_PCR_test, 'r^', Y_test, Y_fitted_MLR_test, 'c*');
xlabel('Observed Response test');
ylabel('Fitted Response test');
title("Comparaison sur les données de test");
legend({'PLS1 avec 2 PC', 'PCR with 2 PC', 'MLR'});
