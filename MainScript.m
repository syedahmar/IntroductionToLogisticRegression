%let us start by illustrating a lienar decision boundary
%% Load and pre-process the data
clc;clear all;
load GeneratedData_final


% Combine data in a matrix first
num_normal=size(data_normal.heartRate,1);
num_abnormal=size(data_abnormal_2.heartRate,1);

X=[ [data_normal.heartRate;data_abnormal_2.heartRate] [data_normal.breathingRate;data_abnormal_2.breathingRate]];
y=[zeros(num_normal,1);ones(num_abnormal,1)];

%% Data Visualisation
% Heart Rate vs Respiratory Rate
figure,hold on
pos = find(y==1); neg = find(y == 0);
plot(X(neg, 1), X(neg, 2), 'k+','LineWidth', 2, 'MarkerSize', 10);
plot(X(pos, 1), X(pos, 2), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 6);
legend('Normal','Abnormal')
xlabel('Heart rate'),ylabel('Breathing Rate')
set(gca(),'FontSize',16)
xlim([76 86])
ylim([14 24])

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

%% plot decision boundary now
x_points=[min(X(:,2))-1 max(X(:,2))+1];
y_points=(-1./theta(3)).*(theta(1)+(theta(2).*x_points));
hold on,plot(x_points,y_points,'-b','LineWidth',3)
legend('Normal','Abnormal','Decision Boundary')
xlim([76 86])
ylim([14 24])
