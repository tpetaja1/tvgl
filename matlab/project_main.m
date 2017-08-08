%% Project - Time-Varying Graphical Lasso

%% Generate real statistics

dimension = 6;
timestamps1 = 15;
timestamps2 = 10;
total_stamps = timestamps1 + timestamps2;
observations = 100;
inv_sigma1 = [1,0.5,0,0,0,0;0.5,1,0.5,0.25,0,0;0,0.5,1,0,0.25,0;0,0.25,0,1,0.5,0;0,0,0.25,0.5,1,0.25;0,0,0,0,0.25,1];
inv_sigma2 = [1,0,0,0.5,0,0;0,1,0,0,0.5,0;0,0,1,0.5,0.25,0.5;0.5,0,0.5,1,0,0;0,0.5,0.25,0,1,0;0,0,0.5,0,0,1];
sigma1 = inv(inv_sigma1);
sigma2 = inv(inv_sigma2);

datacell1 = cell(timestamps1,1);
datacell2 = cell(timestamps2,1);
for i = 1:timestamps1
    datacell1{i} = mvnrnd(zeros(dimension,1), sigma1, observations);
end
for i = 1:timestamps2
    datacell2{i} = mvnrnd(zeros(dimension,1), sigma2, observations);
end
datacells = [datacell1;datacell2];

%% Empirical covariance matrices

empirical_covariance_matrices = cell(total_stamps,1);

for i = 1:total_stamps
    empirical_covariance_matrices{i} = datacells{i}'*datacells{i}/observations;
end

%% Initialize algorithm

max_iter = 1000;
rho = 50;
lambda = 0.9;
beta = 0.9;
nju = observations/(3*rho);

Theta = init_cells(dimension,total_stamps,'ones');

Z0 = init_cells(dimension,total_stamps,'ones');
Z1 = init_cells(dimension,total_stamps,'ones');
Z2 = init_cells(dimension,total_stamps,'ones');

U0 = init_cells(dimension,total_stamps,'zeros');
U1 = init_cells(dimension,total_stamps,'zeros');
U2 = init_cells(dimension,total_stamps,'zeros');

%% Algorithm

Theta = admm_algorithm(Theta,Z0,Z1,Z2,U0,U1,U2,rho,lambda,beta,...
    nju,empirical_covariance_matrices);

%% Plot temporal deviation

deviations = zeros(total_stamps-1,1);
for i = 1:total_stamps-1
    dif = Theta{i+1} - Theta{i};
    deviations(i) = norm(dif,'fro');
end
deviations = deviations/max(deviations);

figure(1);
plot(1:total_stamps-1,deviations,'O-');
%semilogy(deviations);

%% Create and plot network

figure(2);
[G1, p1] = create_plot_graph(inv_sigma1);
figure(3);
[G2, p2] = create_plot_graph(inv_sigma2);
figure(4);
[G3, p3] = create_plot_graph(Theta{1});

