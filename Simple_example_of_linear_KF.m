clear; clc; close all;
noisemax=0.5;
bias=0.2;
[err1,sigma_pos_avg1,~]=simulation(0,0);
[err2,sigma_pos_avg2,~]=simulation(noisemax,0);
[err3,sigma_pos_avg3,~]=simulation(noisemax,bias);
[err4,sigma_pos_avg4,N]=simulation(noisemax,1);
%% Plot: |error|  vs  1-sigma
t1 = 1:N;
figure('Position', [100, 100, 700, 500]);  % [left bottom width height]

% Use tiledlayout for modern style
t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

% First subplot
nexttile;
%plot(t, err1, 'b',  t, sigma_pos_avg, 'r--', 'LineWidth',1.2);
hold on
plot(t1, err1, 'LineWidth',1.2);
plot(t1, sigma_pos_avg1, 'LineWidth',1.2)
set(gca,'FontSize',12, 'YScale', 'log')
xlabel('Time step','FontSize',12); 
%ylabel('Position (m)');
ylabel('Estimation error','FontSize',12);
legend('Actual error','Estimated error', 'Location','northeast','FontSize',12);
grid on; 
title('(a) $\gamma = 0, \quad \beta=0$', 'Interpreter','latex','FontSize',14,'FontWeight','normal');

% Second subplot
nexttile;
%plot(t, err1, 'b',  t, sigma_pos_avg, 'r--', 'LineWidth',1.2);
hold on
plot(t1, err2, 'LineWidth',1.2);
plot(t1, sigma_pos_avg2, 'LineWidth',1.2)
set(gca,'FontSize',12, 'YScale', 'log')
xlabel('Time step','FontSize',12); 
%ylabel('Position (m)');
ylabel('Estimation error','FontSize',12);
legend('Actual error','Estimated error', 'Location','northeast','FontSize',12);
grid on; 
title('(b) $\gamma = 0.5, \quad \beta=0$', 'Interpreter','latex','FontSize',14,'FontWeight','normal');

% Third subplot
nexttile;
%plot(t, err1, 'b',  t, sigma_pos_avg, 'r--', 'LineWidth',1.2);
hold on
plot(t1, err3, 'LineWidth',1.2);
plot(t1, sigma_pos_avg3, 'LineWidth',1.2)
set(gca,'FontSize',12, 'YScale', 'log')
xlabel('Time step','FontSize',12); 
%ylabel('Position (m)');
ylabel('Estimation error','FontSize',12);
legend('Actual error','Estimated error', 'Location','northeast','FontSize',12);
grid on; 
title('(c) $\gamma = 0.5, \quad \beta=0.2$', 'Interpreter','latex','FontSize',14,'FontWeight','normal');

% Fourth subplot
nexttile;
%plot(t, err1, 'b',  t, sigma_pos_avg, 'r--', 'LineWidth',1.2);
hold on
plot(t1, err4, 'LineWidth',1.2);
plot(t1, sigma_pos_avg4, 'LineWidth',1.2)
set(gca,'FontSize',12, 'YScale', 'log')
xlabel('Time step','FontSize',12); 
%ylabel('Position (m)');
ylabel('Estimation error','FontSize',12);
legend('Actual error','Estimated error', 'Location','northeast','FontSize',12);
grid on; 
title('(d) $\gamma = 0.5, \quad \beta=1$', 'Interpreter','latex','FontSize',14,'FontWeight','normal');
exportgraphics(gcf, 'why_compensation.png', 'Resolution', 600);

function [err1,sigma_pos_avg,N]=simulation(noisemax,bias)
rng(0)
repeatnum=10000;
sigma_pos_avg=0;
sigma_vel_avg=0;
err1=0;
for iii=1:repeatnum
%% System definition
dt = 1;                       % sample time  (s)
A  = 1;             % state-transition (constant-velocity model)
H  = 1;                   % measurement matrix (position only)

Q  = 1e-8;           % process-noise covariance
R  = 1e-4;                    % measurement-noise variance (scalar)

N  = 50;                      % number of time steps to simulate

%% Storage pre-allocation
x_true  = zeros(1,N);         % ground-truth states
x_est   = zeros(1,N);         % filter estimates
P_store = zeros(1,N);       % filter covariances
err     = zeros(1,N);         % estimation errors
z       = zeros(1,N);         % noisy measurements

%% Initial conditions
x_true = 0;         % starts at 0 m with 1 m/s
P = 1;         % initial covariance
x_est  = x_true + mvnrnd(0,P).';
err    = x_true - x_est;
P_store(1) = P;
%% Main simulation / filtering loop
for k = 2:N
    %----- True system propagation -----
    w = mvnrnd(0,Q).';         % process noise
    x_true(k) = A * x_true(k-1) + w;

    %----- Generate measurement -----
    v = sqrt(R) * randn;           % measurement noise
    z(k) = H * x_true(k) + v;    % scalar measurement

    %----- Kalman filter prediction -----
    x_pred = A * x_est(k-1);
    P_pred = A * P * A' + Q;

    %----- Kalman filter update -----
    a = noisemax * (2 * rand) + 1 - noisemax;
    S = a * a * (H * P_pred * H')*(1+bias) + R;
    S0 = (H * P_pred * H')*(1+bias) + R;
    Pxy = a * P_pred * H';
    Pxy0 = P_pred * H';
    K      = Pxy / S;   % Kalman gain
    x_est(k) = x_pred + K * (z(k) - H * x_pred);
    P          = P_pred + K * S* K.' - Pxy*K.' - K*Pxy';
    P2 = P_pred + K * S0* K.' - Pxy0*K.' - K*Pxy0';
    %----- Logging -----
    P_store(k) = P;
    err(k)       = x_true(k) - x_est(k);
end

%% 1-sigma standard deviations (square roots of diagonal P)
sigma_pos = sqrt(P_store);
sigma_pos_avg=sigma_pos_avg+sigma_pos./repeatnum;
err1=err1+err.^2/repeatnum;
end
err1=sqrt(err1);
end