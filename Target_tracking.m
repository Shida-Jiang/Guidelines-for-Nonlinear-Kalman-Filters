%%% This code is adapted from:mean(log
% Y. Kim and H. Bang, Introduction to Kalman Filter and Its Applications, InTechOpen, 2018

% Application 1: 3D target tracking
close all
clc
clear

repeat = 10000; %The number of times that the simulation is repeated at each measurement noise setup
%This simulation takes about 30 minutes, you can reduce the repeat times to reduce the run time

%% simulation
scale=-2; % The range of measurement noise is 10^scale
beta0 = NaN(1,5); % show how the estimation convergences when the std of the measurement noise is 10^-2 
min_RMSE=9999*ones(1,5);
beta_range=10.^(-2:0.2:2);
%Defining some parameters

xplotCKF_2=[];
xerrorplotCKF_2=[];
vxerrorplotCKF_2=[];
vxplotCKF_2=[];
Time_CKF_2=[];

xplotEKF_2=[];
xerrorplotEKF_2=[];
vxerrorplotEKF_2=[];
vxplotEKF_2=[];
Time_EKF_2=[];

xplotEKF2_2=[];
xerrorplotEKF2_2=[];
vxerrorplotEKF2_2=[];
vxplotEKF2_2=[];
Time_EKF2_2=[];

xplotUKF_2=[];
xerrorplotUKF_2=[];
vxerrorplotUKF_2=[];
vxplotUKF_2=[];
Time_UKF_2=[];

xplotSKF_2=[];
xerrorplotSKF_2=[];
vxerrorplotSKF_2=[];
vxplotSKF_2=[];
Time_SKF_2=[];

for beta=beta_range
    magnitude=10^scale;

    [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(1,magnitude,repeat,0,beta);
    xplotEKF_2=[xplotEKF_2 geo_mean(x_RMSE(:))];
    xerrorplotEKF_2=[xerrorplotEKF_2 geo_mean(x_error_self_est(:))];
    vxplotEKF_2=[vxplotEKF_2 geo_mean(x_RMSE(4,:))];
    vxerrorplotEKF_2=[vxerrorplotEKF_2 geo_mean(x_error_self_est(2,:))];
    %vxplotEKF_2=[vxplotEKF_2 x_RMSE(4,end)];
    %vxerrorplotEKF_2=[vxerrorplotEKF_2 x_error_self_est(2,end)];
    Time_EKF_2=[Time_EKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(1)
        beta0(1)=beta;
        min_RMSE(1)=geo_mean(x_RMSE(:));
    end

    [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(1,magnitude,repeat,3,beta);
    xplotEKF2_2=[xplotEKF2_2 geo_mean(x_RMSE(:))];
    xerrorplotEKF2_2=[xerrorplotEKF2_2 geo_mean(x_error_self_est(:))];
    vxplotEKF2_2=[vxplotEKF2_2 geo_mean(x_RMSE(4,:))];
    vxerrorplotEKF2_2=[vxerrorplotEKF2_2 geo_mean(x_error_self_est(2,:))];
    %vxplotEKF2_2=[vxplotEKF2_2 x_RMSE(4,end)];
    %vxerrorplotEKF2_2=[vxerrorplotEKF2_2 x_error_self_est(2,end)];
    Time_EKF2_2=[Time_EKF2_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(2)
        beta0(2)=beta;
        min_RMSE(2)=geo_mean(x_RMSE(:));
    end

    [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(1,magnitude,repeat,1.1,beta);
    xplotCKF_2=[xplotCKF_2 geo_mean(x_RMSE(:))];
    xerrorplotCKF_2=[xerrorplotCKF_2 geo_mean(x_error_self_est(:))];
    vxplotCKF_2=[vxplotCKF_2 geo_mean(x_RMSE(4,:))];
    vxerrorplotCKF_2=[vxerrorplotCKF_2 geo_mean(x_error_self_est(2,:))];
    %vxplotCKF_2=[vxplotCKF_2 x_RMSE(4,end)];
    %vxerrorplotCKF_2=[vxerrorplotCKF_2 x_error_self_est(2,end)];
    Time_CKF_2=[Time_CKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(3)
        beta0(3)=beta;
        min_RMSE(3)=geo_mean(x_RMSE(:));
    end

    [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(1,magnitude,repeat,1,beta);
    xplotUKF_2=[xplotUKF_2 geo_mean(x_RMSE(:))];
    xerrorplotUKF_2=[xerrorplotUKF_2 geo_mean(x_error_self_est(:))];
    vxplotUKF_2=[vxplotUKF_2 geo_mean(x_RMSE(4,:))];
    vxerrorplotUKF_2=[vxerrorplotUKF_2 geo_mean(x_error_self_est(2,:))];
    %vxplotUKF_2=[vxplotUKF_2 x_RMSE(4,end)];
    %vxerrorplotUKF_2=[vxerrorplotUKF_2 x_error_self_est(2,end)];
    Time_UKF_2=[Time_UKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(4)
        beta0(4)=beta;
        min_RMSE(4)=geo_mean(x_RMSE(:));
    end

    [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(1,magnitude,repeat,4,beta);
    xplotSKF_2=[xplotSKF_2 geo_mean(x_RMSE(:))];
    xerrorplotSKF_2=[xerrorplotSKF_2 geo_mean(x_error_self_est(:))];
    vxplotSKF_2=[vxplotSKF_2 geo_mean(x_RMSE(4,:))];
    vxerrorplotSKF_2=[vxerrorplotSKF_2 geo_mean(x_error_self_est(2,:))];
    %vxplotSKF_2=[vxplotSKF_2 x_RMSE(4,end)];
    %vxerrorplotSKF_2=[vxerrorplotSKF_2 x_error_self_est(2,end)];
    Time_SKF_2=[Time_SKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(5)
        beta0(5)=beta;
        min_RMSE(5)=geo_mean(x_RMSE(:));
    end
end
[~, ~, x_RMSE, ~, ~, x_error_self_est] = simulation(1,magnitude,repeat,0,beta0(1));
x_EKF_2=x_RMSE(1,:);
y_EKF_2=x_RMSE(4,:);
x_EKF_2_self=x_error_self_est(1,:);
y_EKF_2_self=x_error_self_est(2,:);
[~, ~, x_RMSE, ~, ~, x_error_self_est] = simulation(1,magnitude,repeat,3,beta0(2));
x_EKF2_2=x_RMSE(1,:);
y_EKF2_2=x_RMSE(4,:);
x_EKF2_2_self=x_error_self_est(1,:);
y_EKF2_2_self=x_error_self_est(2,:);
[~, ~, x_RMSE, ~, ~, x_error_self_est] = simulation(1,magnitude,repeat,1.1,beta0(3));%CKF
x_CKF_2=x_RMSE(1,:);
y_CKF_2=x_RMSE(4,:);
x_CKF_2_self=x_error_self_est(1,:);
y_CKF_2_self=x_error_self_est(2,:);
[~, ~, x_RMSE, ~, ~, x_error_self_est] = simulation(1,magnitude,repeat,1,beta0(4));%SSKF
x_UKF_2=x_RMSE(1,:);
y_UKF_2=x_RMSE(4,:);
x_UKF_2_self=x_error_self_est(1,:);
y_UKF_2_self=x_error_self_est(2,:);
[~, ~, x_RMSE, ~, ~, x_error_self_est] = simulation(1,magnitude,repeat,4,beta0(5));%SKF
x_SKF_2=x_RMSE(1,:);
y_SKF_2=x_RMSE(4,:);
x_SKF_2_self=x_error_self_est(1,:);
y_SKF_2_self=x_error_self_est(2,:);

%% plot results
% Create figure with a specific size
f1 = figure('Position',[100 100 600 300]);  % [left bottom width height]
% Apply tiled layout
%tl = tiledlayout(f1,2,1,'TileSpacing','Compact','Padding','Compact');
%nexttile
hold on
h1_2=plot(beta_range, xplotEKF_2, '-o', 'Color', '#0072BD','LineWidth',1.5);
h1_1=plot(beta_range, xerrorplotEKF_2, '--o', 'Color', '#0072BD','LineWidth',1.5);
h2_2=plot(beta_range, xplotEKF2_2, '-s', 'Color', "#D95319",'LineWidth',1.5);
h2_1=plot(beta_range, xerrorplotEKF2_2, '--s', 'Color', "#D95319",'LineWidth',1.5);
h3_2=plot(beta_range, xplotCKF_2, '-^', 'Color', "#EDB120",'LineWidth',1.5);
h3_1=plot(beta_range, xerrorplotCKF_2, '--^', 'Color', "#EDB120",'LineWidth',1.5);
h4_2=plot(beta_range, xplotSKF_2, '-x', 'Color', "#7E2F8E",'LineWidth',1.5);
h4_1=plot(beta_range, xerrorplotSKF_2, '--x', 'Color', "#7E2F8E",'LineWidth',1.5);
h5_2=plot(beta_range, xplotUKF_2, '-d', 'Color', "#4DBEEE",'LineWidth',1.5);
h5_1=plot(beta_range, xerrorplotUKF_2, '--d', 'Color', "#4DBEEE",'LineWidth',1.5);
%set(gca, 'XScale', 'log', 'XTickLabel', [],'FontSize', 12)
%xlabel('Measurement noise (per unit)',FontSize=12)
ylabel('Geometric mean of RMSE',FontSize=12)
%ylim([10^-3 11])
%xlim([10^-4 100])
%set(gca, 'YTick', [0.001 0.01 0.1 1 10 100]);
%set(gca, 'XTick', [0.0001 0.001 0.01 0.1 1 10]);
h1_2=plot(nan, nan, '--', 'Color', 'k', 'DisplayName', 'Estimated error','LineWidth',1.5);
h2_2=plot(nan, nan, 'Color', "k", 'DisplayName', 'Actual error','LineWidth',1.5);
% leg=legend([h1_1 h2_1], 'Location','northeast',FontSize=11);
% title(leg,'Line styles')
% grid on
% nexttile
% hold on
% h1_2=plot(beta_range, vxplotEKF_2, '-o', 'Color', '#0072BD','LineWidth',1.5);
% h1_1=plot(beta_range, vxerrorplotEKF_2, '--o', 'Color', '#0072BD','LineWidth',1.5);
% h2_2=plot(beta_range, vxplotEKF2_2, '-s', 'Color', "#D95319",'LineWidth',1.5);
% h2_1=plot(beta_range, vxerrorplotEKF2_2, '--s', 'Color', "#D95319",'LineWidth',1.5);
% h3_2=plot(beta_range, vxplotCKF_2, '-^', 'Color', "#EDB120",'LineWidth',1.5);
% h3_1=plot(beta_range, vxerrorplotCKF_2, '--^', 'Color', "#EDB120",'LineWidth',1.5);
% h4_2=plot(beta_range, vxplotSKF_2, '-x', 'Color', "#7E2F8E",'LineWidth',1.5);
% h4_1=plot(beta_range, vxerrorplotSKF_2, '--x', 'Color', "#7E2F8E",'LineWidth',1.5);
% h5_2=plot(beta_range, vxplotUKF_2, '-d', 'Color', "#4DBEEE",'LineWidth',1.5);
% h5_1=plot(beta_range, vxerrorplotUKF_2, '--d', 'Color', "#4DBEEE",'LineWidth',1.5);
h1_1=plot(nan, nan, 'o', 'Color', '#0072BD', 'DisplayName', 'EKF','LineWidth',1.5);
h2_1=plot(nan, nan, 's', 'Color', "#D95319", 'DisplayName', 'EKF2','LineWidth',1.5);
h3_1=plot(nan, nan, '^', 'Color', "#EDB120", 'DisplayName', 'CKF*','LineWidth',1.5);
h4_1=plot(nan, nan, 'x', 'Color', "#7E2F8E", 'DisplayName', 'SKF*','LineWidth',1.5);
h5_1=plot(nan, nan, 'd', 'Color', "#4DBEEE", 'DisplayName', 'SSKF','LineWidth',1.5);
set(gca, 'XScale', 'log', 'YScale', 'log','FontSize', 12)
xlabel('Beta',FontSize=12)
%ylabel('Average of log speed RMSE [log(m/s)]',FontSize=12)
%ylim([10^-4 1.5])
%xlim([10^-4 100])
%set(gca, 'XTick', [0.0001 0.001 0.01 0.1 1 10 100]);
%set(gca, 'YTick', [0.0001 0.001 0.01 0.1 1 10]);
leg=legend([h1_2 h1_1 h2_1 h3_1 h2_2 h4_1 h5_1], 'Location','northeast','NumColumns',2,FontSize=11);
%title(leg,'Line colors')
grid on
exportgraphics(f1,'3D-tracking-beta.png','Resolution',600)
function [time, x_true, x_RMSE, p_sensor, Runtime, x_error_self_est] = simulation(improvement,magnitude,M,KFtype,beta)
rng(123);
N = 30; % number of time steps
dt = 1; % time between time steps
%M = 1000; % number of Monte-Carlo runs

sig_mea_true = [1.0; 1.0]*magnitude; % true value of standard deviation of measurement noise
sig_pro = [1e-3; 1e-3; 1e-3];% standard deviation of process noise
sig_mea = sig_mea_true; % user input of standard deviation of measurement noise

sig_init = [10; 10; 10; 0.1; 0.1; 0.1]; % standard deviation of initial guess

Q = [zeros(3), zeros(3); zeros(3), diag(sig_pro.^2)]; % process noise covariance matrix
R = diag(sig_mea.^2); % measurement noise covariance matrix

F = [eye(3), eye(3)*dt; zeros(3), eye(3)]; % state transition matrix
B = eye(6);
u = zeros(6,1); % noise
x_error_self_est=zeros(2,N+1);
%% true trajectory

% sensor trajectory
p_sensor = zeros(3,N+1);
for k = 1:1:N+1
    p_sensor(1,k) = 20 + 20*cos(2*pi/30 * (k-1));
    p_sensor(2,k) = 20 + 20*sin(2*pi/30 * (k-1));
    p_sensor(3,k) = 0;
end

% true target trajectory
x_true = zeros(6,N+1); 
x_true(:,1) = [10; -10; 50; 1; 2; 0]; % initial true state
for k = 2:1:N+1
    x_true(:,k) = F*x_true(:,k-1);
end

%% Kalman filter simulation
Runtime=0;
res_x_est = zeros(6,N+1,M); % Monte-Carlo estimates
res_x_err = zeros(6,N+1,M); % Monte-Carlo estimate errors
P_diag = zeros(6,N+1); % diagonal term of error covariance matrix

% filtering
for m = 1:1:M
    % initial guess
    x_est(:,1) = x_true(:,1) + normrnd(0, sig_init);
    P = [diag(sig_init(1:3).^2), zeros(3); zeros(3), diag(sig_init(4:6).^2)];
    P_diag(:,1) = diag(P);
    x_error_self_est(1)=x_error_self_est(1) + P(1,1)/M;
    x_error_self_est(2)=x_error_self_est(2) + P(4,4)/M;
    for k = 2:1:N+1
        u=[0; 0; 0; sig_pro].*randn(6,1);
        n=6;
        mnum=2;
        points=[diag(ones(n,1));ones(1,n)/(sqrt(n+1)-1)]*sqrt(n+1);
        %points=[diag(ones(n,1))]*sqrt(n);
        points=points-mean(points);
        points=[points;zeros(1,n)];
        tic;
        if(KFtype==0)
            %%% Prediction

            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u;
            

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            pp2 = x_est(1:3,k) - p_sensor(:,k); % predicted relative position
            pp1 = x_est(1:3,k);
            
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement

            % predicted meausrement
            
            z_p = [norm(pp1);
                norm(pp2)]; % predicted measurement

            % measurement residual
            y = z - z_p;

            % measurement matrix
            H = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];

            % Kalman gain
            K = P*H'/(R+H*P*H');

            % updated state estimate
            x_est0=x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*y;

            % updated error covariance
            if(improvement==1)
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                
                temp=P-K*H2*P-P*H2'*K'+K*(H2*P*H2'+R)*K';
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end
            else
                P = (eye(6) - K*H)*P;
            end
        elseif(KFtype==0.5)
            %IEKF
            x_est(:,k) = F*x_est(:,k-1) + B*u;
            

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            pp2 = x_est(1:3,k) - p_sensor(:,k); % predicted relative position
            pp1 = x_est(1:3,k);
            
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement

            % predicted meausrement
            
            z_p = [norm(pp1);
                norm(pp2)]; % predicted measurement

            % measurement residual
            y = z - z_p;

            % measurement matrix
            H = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];

            % Kalman gain
            K = P*H'/(R+H*P*H');
            % updated state estimate
            x_est0=x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*y;
            %iteration
            steplennow = norm(K*y);
            iter=1;
            change=1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                z_p = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                y = z - z_p;
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                S = H2*P*H2.'+ R;
                K2 = P*H2.'*S^(-1);
                dx = x_est0 + K2*(y-H2*(x_est0-x_est(:,k))) - x_est(:,k);
                steplen_previous=steplennow;
                steplennow=norm(dx);
                if(steplen_previous<steplennow)
                    break;
                end
                change = max(abs(dx./x_est(:,k)));
                x_est(:,k) = x_est(:,k) + dx;
                K = K2;
                H = H2;
            end
            % updated error covariance
            if(improvement==1)
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                
                temp=P-K*H2*P-P*H2'*K'+K*(H2*P*H2'+R)*K';
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end
            else
                P = (eye(6) - K*H)*P;
            end
        elseif(KFtype==1.1)%CKF*
            L=chol(P).';
            state=x_est(:,k-1);
            alpha=1;
            lambda=(alpha^2-1)*n;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);  
            end
            for i=1:2*n+1
                states(:,i)=F*states(:,i) + B*u;
                if(i==1)
                    state=states(:,i)*lambda/(n+lambda);
                else
                    state=state+states(:,i)/(n+lambda)/2;
                end
            end
            P=(states(:,1)-state)*(states(:,1)-state).'*(lambda/(n+lambda)+1-alpha^2+beta)+Q;
            for i=2:2*n+1
                P=P+(state-states(:,i))*(state-states(:,i)).'/(n+lambda)/2;
            end
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement
            % Predict Measurement From Propagated Sigma Points
            L=chol(P).';
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);  
            end
            measures=zeros(mnum,2*n+1);
            for i=1:2*n+1
                pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                pp1 = states(1:3,i);
                measures(:,i) = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                if(i==1)
                    m_exp=lambda/(n+lambda)*measures(:,i);
                else
                    m_exp=m_exp+1/(n+lambda)/2*measures(:,i);
                end
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+R;
            Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
            for i=2:2*n+1
                Py=Py+1/(n+lambda)/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                Pxy=Pxy+1/(n+lambda)/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            if(improvement==1)
%                 L=chol(P).';
%                 states=zeros(n,n*2+1);
%                 states(:,1)=state;
%                 for i=1:n
%                     states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
%                     states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
%                 end
                for i=1:2*n+1
                    states(:,i)=states(:,i)+dstate;
                end
                measures=zeros(mnum,2*n+1);
                for i=1:2*n+1
                    pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                    pp1 = states(1:3,i);
                    measures(:,i) = [norm(pp1);
                        norm(pp2)]; % predicted measurement
                    if(i==1)
                        m_exp=lambda/(n+lambda)*measures(:,i);
                    else
                        m_exp=m_exp+1/(n+lambda)/2*measures(:,i);
                    end
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+R;
                Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
                for i=2:2*n+1
                    Py=Py+1/(n+lambda)/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                    Pxy=Pxy+1/(n+lambda)/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
                end
                temp=P+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    state=state0;
                end
            else
                P=P-K*Py*K.';
            end
            x_est(:,k)=state;
        elseif(KFtype==1)
            L=chol(P).';
            state=x_est(:,k-1);
            alpha=1e-3;
            lambda=(alpha^2-1)*n;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);  
            end
            for i=1:2*n+1
                states(:,i)=F*states(:,i) + B*u;
                if(i==1)
                    state=states(:,i)*lambda/(n+lambda);
                else
                    state=state+states(:,i)/(n+lambda)/2;
                end
            end
            P=(states(:,1)-state)*(states(:,1)-state).'*(lambda/(n+lambda)+1-alpha^2+beta)+Q;
            for i=2:2*n+1
                P=P+(state-states(:,i))*(state-states(:,i)).'/(n+lambda)/2;
            end
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement
            % Predict Measurement From Propagated Sigma Points
            L=chol(P).';
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);  
            end
            measures=zeros(mnum,2*n+1);
            for i=1:2*n+1
                pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                pp1 = states(1:3,i);
                measures(:,i) = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                if(i==1)
                    m_exp=lambda/(n+lambda)*measures(:,i);
                else
                    m_exp=m_exp+1/(n+lambda)/2*measures(:,i);
                end
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+R;
            Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
            for i=2:2*n+1
                Py=Py+1/(n+lambda)/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                Pxy=Pxy+1/(n+lambda)/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            if(improvement==1)
%                 L=chol(P).';
%                 states=zeros(n,n*2+1);
%                 states(:,1)=state;
%                 for i=1:n
%                     states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
%                     states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
%                 end
                for i=1:2*n+1
                    states(:,i)=states(:,i)+dstate;
                end
                measures=zeros(mnum,2*n+1);
                for i=1:2*n+1
                    pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                    pp1 = states(1:3,i);
                    measures(:,i) = [norm(pp1);
                        norm(pp2)]; % predicted measurement
                    if(i==1)
                        m_exp=lambda/(n+lambda)*measures(:,i);
                    else
                        m_exp=m_exp+1/(n+lambda)/2*measures(:,i);
                    end
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+R;
                Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
                for i=2:2*n+1
                    Py=Py+1/(n+lambda)/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                    Pxy=Pxy+1/(n+lambda)/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
                end
                temp=P+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    state=state0;
                end
            else
                P=P-K*Py*K.';
            end
            x_est(:,k)=state;
        elseif(KFtype==4)
            L=chol(P).';
            state=x_est(:,k-1);
            pnum=(n+2);
            alpha=1;
            states=repmat(state, 1, pnum)+alpha*(points*L')';
            weight=ones(pnum,1)/(n+1)/alpha^2;
            weight(pnum)=1-1/alpha^2;
            %states-states2
            state=0;
            for i=1:pnum
                states(:,i)=F*states(:,i) + B*u;
                state=state+states(:,i)*weight(i);
            end
            dP=(states(:,pnum)-state)*(states(:,pnum)-state)';
            P=Q;
            %beta=max(0,ceil(5-trace(P)/trace(dP)));
            for i=1:pnum
                P=P+(state-states(:,i))*(state-states(:,i)).'*weight(i);
            end
            P=P+beta*dP;
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement
            %update sigma points
            L=chol(P).';
            states=repmat(state, 1, pnum)+alpha*(points*L')';
            % Predict Measurement From Propagated Sigma Points
            measures=zeros(mnum,pnum);
            m_exp=0;
            for i=1:pnum
                pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                pp1 = states(1:3,i);
                measures(:,i) = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                m_exp=m_exp+weight(i)*measures(:,i);
            end
            % Estimate Mean And Covariance of Predicted Measurement
            
            dP=(measures(:,pnum)-m_exp)*(measures(:,pnum)-m_exp)';
            Py=R;
            %beta=max(0,ceil(5-trace(Py)/trace(dP)));
            Pxy=0;
            for i=1:pnum
                Py=Py+weight(i)*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                Pxy=Pxy+weight(i)*(states(:,i)-state)*(measures(:,i)-m_exp).';
            end
            %beta = max(2, ceil((sum((z - m_exp).^2)/3 - trace(Py))/trace(dP)));
            %beta=beta*1.5;
            %beta = max(2, ceil((sum((z - m_exp).^2)/3 - trace(Py))/trace(dP)));
            %beta=2;
            Py   = Py + beta*dP;
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            if(improvement==1)
                states=repmat(state, 1, pnum)+alpha*(points*L')';
                % Predict Measurement From Propagated Sigma Points
                measures=zeros(mnum,pnum);
                m_exp=0;
                for i=1:pnum
                    pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                    pp1 = states(1:3,i);
                    measures(:,i) = [norm(pp1);
                        norm(pp2)]; % predicted measurement
                    m_exp=m_exp+weight(i)*measures(:,i);
                end
                % Estimate Mean And Covariance of Predicted Measurement
                dP=(measures(:,pnum)-m_exp)*(measures(:,pnum)-m_exp)';
                Py=R;
                Pxy=0;
                for i=1:pnum
                    Py=Py+weight(i)*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                    Pxy=Pxy+weight(i)*(states(:,i)-state)*(measures(:,i)-m_exp).';
                end
                %beta = max(2, ceil((sum((z - m_exp).^2)/3 - trace(Py))/trace(dP)));
                %beta=max(2,ceil(100-trace(Py)/trace(dP)));
                %beta = max(2, ceil((sum((z - m_exp).^2)/3 - trace(Py))/trace(dP)));
                %beta=beta/1.5;
                Py=Py+beta*dP;
                temp=P+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    state=state0;
                end
            else
                P=P-K*Py*K.';
            end
            x_est(:,k)=state;
        elseif(KFtype==2)
            %CKF
            L=chol(P).';
            state=x_est(:,k-1);
            states=zeros(n,n*2);
            for i=1:n
                states(:,i)=state+sqrt(n)*L(:,i);
                states(:,n+i)=state-sqrt(n)*L(:,i);  
            end
            state=0;
            for i=1:2*n
                states(:,i)=F*states(:,i) + B*u;
                state=state+states(:,i)/n/2;
            end
            P=Q;
            for i=1:2*n
                P=P+(state-states(:,i))*(state-states(:,i)).'/n/2;
            end
            % obtain measurement
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement
            %update sigma points
            L=chol(P).';
            states=zeros(n,n*2);
            for i=1:n
                states(:,i)=state+sqrt(n)*L(:,i);
                states(:,n+i)=state-sqrt(n)*L(:,i);  
            end
            % Predict Measurement From Propagated Sigma Points
            measures=zeros(mnum,2*n);
            m_exp=0;
            for i=1:2*n
                pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                pp1 = states(1:3,i);
                measures(:,i) = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                m_exp=m_exp+1/n/2*measures(:,i);
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=R;
            Pxy=0;
            for i=1:2*n
                Py=Py+1/n/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                Pxy=Pxy+1/n/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            state=state+K*(z-m_exp);
            if(improvement==1)
                states=zeros(n,n*2);
                for i=1:n
                    states(:,i)=state+sqrt(n)*L(:,i);
                    states(:,n+i)=state-sqrt(n)*L(:,i);
                end
                % Predict Measurement From Propagated Sigma Points
                measures=zeros(mnum,2*n);
                m_exp=0;
                for i=1:2*n
                    pp2 = states(1:3,i) - p_sensor(:,k); % predicted relative position
                    pp1 = states(1:3,i);
                    measures(:,i) = [norm(pp1);
                        norm(pp2)]; % predicted measurement
                    m_exp=m_exp+1/n/2*measures(:,i);
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=R;
                Pxy=0;
                for i=1:2*n
                    Py=Py+1/n/2*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                    Pxy=Pxy+1/n/2*(states(:,i)-state)*(measures(:,i)-m_exp).';
                end
                
                temp=P+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    state=state0;
                end
            else
                P=P-K*Py*K.';
            end
            x_est(:,k)=state;
        elseif(KFtype==3)
            %2-EKF
            %%% Prediction

            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u;
            n=length(x_est(:,k));

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            pp2 = x_est(1:3,k) - p_sensor(:,k); % predicted relative position
            pp1 = x_est(1:3,k);
            
            % obtain measurement
            
            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement

            % predicted meausrement
            z_p = [norm(pp1);
                norm(pp2)]; % predicted measurement

            % measurement matrix
            H = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
            Hxx=zeros(n,n,mnum);
            Hxx(1:3,1:3,1)=[1/norm(pp1)-pp1(1)^2/norm(pp1)^3 -pp1(1)*pp1(2)/norm(pp1)^3 -pp1(1)*pp1(3)/norm(pp1)^3;
               -pp1(1)*pp1(2)/norm(pp1)^3 1/norm(pp1)-pp1(2)^2/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3;
               -pp1(1)*pp1(3)/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3 1/norm(pp1)-pp1(3)^2/norm(pp1)^3];
            Hxx(1:3,1:3,2)=[1/norm(pp2)-pp2(1)^2/norm(pp2)^3 -pp2(1)*pp2(2)/norm(pp2)^3 -pp2(1)*pp2(3)/norm(pp2)^3;
               -pp2(1)*pp2(2)/norm(pp2)^3 1/norm(pp2)-pp2(2)^2/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3;
               -pp2(1)*pp2(3)/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3 1/norm(pp2)-pp2(3)^2/norm(pp2)^3];
            S=H*P*H.'+R;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.5*beta/2*sum(diag(Hxx(:,:,ii)*P*Hxx(:,:,jj)*P));
                end
            end
            % Kalman gain
            K = P*H.'*S^(-1);
            % measurement residual
            residual=z-z_p;
            for ii=1:mnum
                residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*P));
            end
            % updated state estimate
            x_est0 = x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*residual;

            % updated error covariance
            if(improvement==1)
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                Hxx2=zeros(n,n,mnum);
                Hxx2(1:3,1:3,1)=[1/norm(pp1)-pp1(1)^2/norm(pp1)^3 -pp1(1)*pp1(2)/norm(pp1)^3 -pp1(1)*pp1(3)/norm(pp1)^3;
               -pp1(1)*pp1(2)/norm(pp1)^3 1/norm(pp1)-pp1(2)^2/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3;
               -pp1(1)*pp1(3)/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3 1/norm(pp1)-pp1(3)^2/norm(pp1)^3];
                Hxx2(1:3,1:3,2)=[1/norm(pp2)-pp2(1)^2/norm(pp2)^3 -pp2(1)*pp2(2)/norm(pp2)^3 -pp2(1)*pp2(3)/norm(pp2)^3;
                   -pp2(1)*pp2(2)/norm(pp2)^3 1/norm(pp2)-pp2(2)^2/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3;
                   -pp2(1)*pp2(3)/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3 1/norm(pp2)-pp2(3)^2/norm(pp2)^3];
                S2=H2*P*H2.'+R;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*beta/2*sum(diag(Hxx2(:,:,ii)*P*Hxx2(:,:,jj)*P));
                    end
                end
                temp=P+K*S2*K.'-P*H2.'*K.'-K*H2*P;
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end

            else
                P = P-K*S*K.';
            end
        else
            %2-EKF
            %%% Prediction

            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u;
            n=length(x_est(:,k));

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            pp2 = x_est(1:3,k) - p_sensor(:,k); % predicted relative position
            pp1 = x_est(1:3,k);

            % obtain measurement

            p = x_true(1:3,k) - p_sensor(:,k); % true relative position
            z_true = [norm(x_true(1:3,k));
                norm(p)]; % true measurement

            z = z_true + normrnd(0, sig_mea_true); % erroneous measurement

            % predicted meausrement
            z_p = [norm(pp1);
                norm(pp2)]; % predicted measurement

            % measurement matrix
            H = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
            Hxx=zeros(n,n,mnum);
            Hxx(1:3,1:3,1)=[1/norm(pp1)-pp1(1)^2/norm(pp1)^3 -pp1(1)*pp1(2)/norm(pp1)^3 -pp1(1)*pp1(3)/norm(pp1)^3;
                -pp1(1)*pp1(2)/norm(pp1)^3 1/norm(pp1)-pp1(2)^2/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3;
                -pp1(1)*pp1(3)/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3 1/norm(pp1)-pp1(3)^2/norm(pp1)^3];
            Hxx(1:3,1:3,2)=[1/norm(pp2)-pp2(1)^2/norm(pp2)^3 -pp2(1)*pp2(2)/norm(pp2)^3 -pp2(1)*pp2(3)/norm(pp2)^3;
                -pp2(1)*pp2(2)/norm(pp2)^3 1/norm(pp2)-pp2(2)^2/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3;
                -pp2(1)*pp2(3)/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3 1/norm(pp2)-pp2(3)^2/norm(pp2)^3];
            S=H*P*H.'+R;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.5*sum(diag(Hxx(:,:,ii)*P*Hxx(:,:,jj)*P));
                end
            end
            % Kalman gain
            K = P*H.'*S^(-1);
            % measurement residual
            residual=z-z_p;
            for ii=1:mnum
                residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*P));
            end
            % updated state estimate
            x_est0 = x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*residual;
            
            % iteration
            steplennow = norm(K*residual);
            iter=1;
            change=1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                z_p = [norm(pp1);
                    norm(pp2)]; % predicted measurement
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                Hxx2=zeros(n,n,mnum);
                Hxx2(1:3,1:3,1)=[1/norm(pp1)-pp1(1)^2/norm(pp1)^3 -pp1(1)*pp1(2)/norm(pp1)^3 -pp1(1)*pp1(3)/norm(pp1)^3;
                    -pp1(1)*pp1(2)/norm(pp1)^3 1/norm(pp1)-pp1(2)^2/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3;
                    -pp1(1)*pp1(3)/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3 1/norm(pp1)-pp1(3)^2/norm(pp1)^3];
                Hxx2(1:3,1:3,2)=[1/norm(pp2)-pp2(1)^2/norm(pp2)^3 -pp2(1)*pp2(2)/norm(pp2)^3 -pp2(1)*pp2(3)/norm(pp2)^3;
                    -pp2(1)*pp2(2)/norm(pp2)^3 1/norm(pp2)-pp2(2)^2/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3;
                    -pp2(1)*pp2(3)/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3 1/norm(pp2)-pp2(3)^2/norm(pp2)^3];
                residual=z-z_p;
                for ii=1:mnum
                    residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*P));
                    residual(ii)=residual(ii)-0.5*(x_est0-x_est(:,k)).'*Hxx2(:,:,ii)*(x_est0-x_est(:,k));
                end
                S2 = H2*P*H2.'+ R;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*sum(diag(Hxx2(:,:,ii)*P*Hxx2(:,:,jj)*P));
                    end
                end
                K2 = P*H2.'*S2^(-1);
                dx = x_est0 + K2*(residual-H2*(x_est0-x_est(:,k))) - x_est(:,k);
                steplen_previous=steplennow;
                steplennow=norm(dx);
                if(steplen_previous<steplennow)
                    break;
                end
                change = max(abs(dx./x_est(:,k)));
                x_est(:,k) = x_est(:,k) + dx;
                K = K2;
                S = S2;
            end
            % updated error covariance
            if(improvement==1)
                pp2 = x_est(1:3,k) - p_sensor(:,k);
                pp1 = x_est(1:3,k);
                H2 = [pp1(1)/norm(pp1), pp1(2)/norm(pp1), pp1(3)/norm(pp1), zeros(1,3);
                    pp2(1)/norm(pp2), pp2(2)/norm(pp2), pp2(3)/norm(pp2), zeros(1,3)];
                Hxx2=zeros(n,n,mnum);
                Hxx2(1:3,1:3,1)=[1/norm(pp1)-pp1(1)^2/norm(pp1)^3 -pp1(1)*pp1(2)/norm(pp1)^3 -pp1(1)*pp1(3)/norm(pp1)^3;
                    -pp1(1)*pp1(2)/norm(pp1)^3 1/norm(pp1)-pp1(2)^2/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3;
                    -pp1(1)*pp1(3)/norm(pp1)^3 -pp1(2)*pp1(3)/norm(pp1)^3 1/norm(pp1)-pp1(3)^2/norm(pp1)^3];
                Hxx2(1:3,1:3,2)=[1/norm(pp2)-pp2(1)^2/norm(pp2)^3 -pp2(1)*pp2(2)/norm(pp2)^3 -pp2(1)*pp2(3)/norm(pp2)^3;
                    -pp2(1)*pp2(2)/norm(pp2)^3 1/norm(pp2)-pp2(2)^2/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3;
                    -pp2(1)*pp2(3)/norm(pp2)^3 -pp2(2)*pp2(3)/norm(pp2)^3 1/norm(pp2)-pp2(3)^2/norm(pp2)^3];
                S2=H2*P*H2.'+R;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*sum(diag(Hxx2(:,:,ii)*P*Hxx2(:,:,jj)*P));
                    end
                end
                temp=P+K*S2*K.'-P*H2.'*K.'-K*H2*P;
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end

            else
                P = P-K*S*K.';
            end

        end
        temp=toc;
        Runtime = Runtime + temp/M;
        P_diag(:,k) = diag(P);
        x_error_self_est(1,k)=x_error_self_est(1,k) + P(1,1)/M;
        x_error_self_est(2,k)=x_error_self_est(2,k) + P(4,4)/M;
    end    
    res_x_est(:,:,m) = x_est;
    res_x_err(:,:,m) = x_est - x_true;
    time = (0:1:N)*dt;
end
x_error_self_est = sqrt(x_error_self_est);
%% get result statistics

x_RMSE = zeros(6,N+1); % root mean square error
for k = 1:1:N+1
    x_RMSE(1,k) = sqrt(mean(res_x_err(1,k,:).^2,3));
    x_RMSE(2,k) = sqrt(mean(res_x_err(2,k,:).^2,3));
    x_RMSE(3,k) = sqrt(mean(res_x_err(3,k,:).^2,3));
    x_RMSE(4,k) = sqrt(mean(res_x_err(4,k,:).^2,3));
    x_RMSE(5,k) = sqrt(mean(res_x_err(5,k,:).^2,3));
    x_RMSE(6,k) = sqrt(mean(res_x_err(6,k,:).^2,3));
end
end



