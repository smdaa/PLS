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

%% Visualisation des données

figure(1)
plot3(repmat(1:401, 60, 1)', repmat((1:size(octane, 1))', 1,401)', NIR');
xlabel("indice de longueur d'onde");
ylabel("gasoline");
title('Visualisation des données');
% On a un grand nombre de variables prédictives fortement corrélés

%% Régression linéaire multiple

Beta_MLR = X \ Y;
Y_fitted_MLR = X * Beta_MLR;
R_2 = R_squared(Y, Y_fitted_MLR);
fprintf('MLR : R^2 = %.6f\n',R_2);
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
[~, ~, latent] = PCA(X);
latent = latent(1:10,1);
plot(1:length(latent),sort(latent ./ sum(latent),'descend'), '-o');
title('Pourcentage de variance expliqué en X');
xlabel('num de la comp. ppale');
ylabel('pourcentage de variance');

[BetaPCR, Y_fitted_PCR] = PCR(Y, X, k);
R_2 = R_squared(Y, Y_fitted_PCR);
fprintf('PCR : R^2 = %.6f\n',R_2);

% On obtient une faible valeur prédictive de 0.19
% On remarque que nos premiers PC capturent 85% de la variance, les 15% restants apparaissent être important. pour prédire Y

%% Régression des moindres carrés partiels

[BetaPLS, Y_fitted_PLS] = PLS(Y, X, k);
R_2 = R_squared(Y, Y_fitted_PLS);
fprintf('PLS : R^2 = %.6f\n',R_2);

%% PLS vs PCR
figure(3);
plot(Y, Y_fitted_PLS,'bo', Y, Y_fitted_PCR,'r^');
%plot(Y, Y_fitted_PLS,'bo');
xlabel('Observed Response');
ylabel('Fitted Response');
legend({'PLS1 avec 2 PC' 'PCR with 2 PC'})

