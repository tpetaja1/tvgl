%% Project - Time-Varying Graphical Lasso
%%
%% Generate real statistics
dimension = 6;
timestamps1 = 10;
timestamps2 = 10;
total_stamps = timestamps1 + timestamps2;
observations = 10;
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
    empirical_covariance_matrices{i} = datacells{i}'*datacells{i};
end

%% Create and plot graph

figure(1);
[G1, p1] = create_plot_graph(inv_sigma1);
figure(2);
[G2, p2] = create_plot_graph(inv_sigma2);

