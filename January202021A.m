%% Load and pre-process the data
clc;clear all;
load GeneratedData


% Combine data in a matrix first
num_normal=size(HR_data_N,1);
num_abnormal=size(HR_data_P,1);

X=[ [HR_data_N;HR_data_P] [RR_data_N;RR_data_P] [SpO2_data_N;SpO2_data_P]];
y=[zeros(num_normal,1);ones(num_abnormal,1)];

%% Data Visualisation
figure,

% Heart Rate vs Respiratory Rate
subplot(3,1,1),hold on
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 10);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 6);
legend('Heart Rate','Respiratory Rate')
xlabel('Heart rate'),ylabel('Respiratory Rate')
set(gca(),'FontSize',16)

% Heart Rate vs SpO2
subplot(3,1,2),hold on
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 3), 'k+','LineWidth', 2, 'MarkerSize', 10);
plot(X(neg, 1), X(neg, 3), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 6);
legend('Heart Rate','SpO_2')
xlabel('Heart rate'),ylabel('SpO_2')
set(gca(),'FontSize',16)

% Respiratory Rate vs SpO2
subplot(3,1,3),hold on
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 2), X(pos, 3), 'k+','LineWidth', 2, 'MarkerSize', 10);
plot(X(neg, 2), X(neg, 3), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 6);
legend('Respiratory Rate','SpO_2')
xlabel('Respiratory Rate'),ylabel('SpO_2')
set(gca(),'FontSize',16)

%% Set up data appropriately now
%  Setup the data matrix appropriately
[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Initialize the fitting parameters
initial_theta = zeros(n + 1, 1);

%Compute the gradient
% Compute and display the initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
disp('Gradient at initial theta (zeros):'); 
disp(grad);

%  Set options for fminunc
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

