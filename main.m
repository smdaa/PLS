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
% On réduit la dimenssion donc plus d'équations que d'inconnues
% Par Construction les Composantes principales sont non-corrélées

Y = octane;
X = NIR;
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
% en effet la variabilité dans X qui permette de prédire y peut etre trés faible et ne contribue donc pas beaucoup aux premiers
% conposantes principales

% Combien de composants font nous avons besoin ->  Cross validation
%% PLS
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

