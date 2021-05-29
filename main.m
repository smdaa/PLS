%% Initialisation
clc;
clear;
close all;

% On charge une data set comprenant les intensités spectrales de 60 échantillons d'essence à 401 longueurs d'onde
load spectra

% On dispose de 60 type de gasoline (individus)
% NIR    : Spectroscopie dans l'infrarouge proche de 900 nm à 1700 nm (401 variables pour chaque octane aka type de gasoile)
% octane : Indice d'octane (mesure la résistance à l'auto-allumage du carburant)

%% Visualisation des données
figure(1)
plot3(repmat(1:401, 60, 1)', repmat((1:size(octane, 1))', 1,401)', NIR');
xlabel("indice de longueur d'onde");
ylabel("gasoline");
title('Visualisation des données');
% On remarque que les individus sont fortement corrélés   

%% PCR
Y = octane;
X = NIR;
k = 2;

[BetaPCR, Y_fitted_PCR] = PCR(Y, X, k);
R_2 = R_squared(Y, Y_fitted_PCR);
fprintf('multiple linear regression : R^2 = %.6f\n',R_2);
% On obtient une faible valeur prédictive de 0.19

figure(2)
plot(Y, Y_fitted_PCR,'bo')



