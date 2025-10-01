%%% This code is adapted from:
% Y. Kim and H. Bang, Introduction to Kalman Filter and Its Applications, InTechOpen, 2018

% Application 2:  Terrain-referenced navigation

close all
clc
clear

repeat=10000;%The number of times that the simulation is repeated at each measurement noise setup
%This simulation takes about 30 minutes, you can reduce the repeat times to reduce the run time
%% settings
scale=0; % The range of measurement noise is 10^scale
beta0 = NaN(1,5); % show how the estimation convergences when the std of the measurement noise is 10^-2 
min_RMSE=9999*ones(1,5);
beta_range=10.^(-2:0.2:2);
%Defining some parameters

xplotCKF_2=[];
xerrorplotCKF_2=[];
yerrorplotCKF_2=[];
yplotCKF_2=[];
Time_CKF_2=[];

xplotEKF_2=[];
xerrorplotEKF_2=[];
yerrorplotEKF_2=[];
yplotEKF_2=[];
Time_EKF_2=[];

xplotEKF2_2=[];
xerrorplotEKF2_2=[];
yerrorplotEKF2_2=[];
yplotEKF2_2=[];
Time_EKF2_2=[];

xplotUKF_2=[];
xerrorplotUKF_2=[];
yerrorplotUKF_2=[];
yplotUKF_2=[];
Time_UKF_2=[];

xplotSKF_2=[];
xerrorplotSKF_2=[];
yerrorplotSKF_2=[];
yplotSKF_2=[];
Time_SKF_2=[];
for beta=beta_range
    magnitude=10^(scale-3);

    [x_RMSE, Runtime, x_error_self_est]=simulation(1,magnitude,repeat,0,beta);
    len=length(x_RMSE(1,:));
    xplotEKF_2=[xplotEKF_2 geo_mean(x_RMSE(:))];
    yplotEKF_2=[yplotEKF_2 mean(log(x_RMSE(2,:)))];
    xerrorplotEKF_2=[xerrorplotEKF_2 geo_mean(x_error_self_est(:))];
    yerrorplotEKF_2=[yerrorplotEKF_2 mean(log(x_error_self_est(2,:)))];
    Time_EKF_2=[Time_EKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(1)
        beta0(1)=beta;
        min_RMSE(1)=geo_mean(x_RMSE(:));
    end

    [x_RMSE, Runtime, x_error_self_est]=simulation(1,magnitude,repeat,3,beta);
    xplotEKF2_2=[xplotEKF2_2 geo_mean(x_RMSE(:))];
    yplotEKF2_2=[yplotEKF2_2 mean(log(x_RMSE(2,:)))];
    xerrorplotEKF2_2=[xerrorplotEKF2_2 geo_mean(x_error_self_est(:))];
    yerrorplotEKF2_2=[yerrorplotEKF2_2 mean(log(x_error_self_est(2,:)))];
    Time_EKF2_2=[Time_EKF2_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(2)
        beta0(2)=beta;
        min_RMSE(2)=geo_mean(x_RMSE(:));
    end

    [x_RMSE, Runtime, x_error_self_est]=simulation(1,magnitude,repeat,1.1,beta);
    xplotCKF_2=[xplotCKF_2 geo_mean(x_RMSE(:))];
    yplotCKF_2=[yplotCKF_2 mean(log(x_RMSE(2,:)))];
    xerrorplotCKF_2=[xerrorplotCKF_2 geo_mean(x_error_self_est(:))];
    yerrorplotCKF_2=[yerrorplotCKF_2 mean(log(x_error_self_est(2,:)))];
    Time_CKF_2=[Time_CKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(3)
        beta0(3)=beta;
        min_RMSE(3)=geo_mean(x_RMSE(:));
    end

    [x_RMSE, Runtime, x_error_self_est]=simulation(1,magnitude,repeat,1,beta);
    xplotUKF_2=[xplotUKF_2 geo_mean(x_RMSE(:))];
    yplotUKF_2=[yplotUKF_2 mean(log(x_RMSE(2,:)))];
    xerrorplotUKF_2=[xerrorplotUKF_2 geo_mean(x_error_self_est(:))];
    yerrorplotUKF_2=[yerrorplotUKF_2 mean(log(x_error_self_est(2,:)))];
    Time_UKF_2=[Time_UKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(4)
        beta0(4)=beta;
        min_RMSE(4)=geo_mean(x_RMSE(:));
    end

    [x_RMSE, Runtime, x_error_self_est]=simulation(1,magnitude,repeat,4,beta);
    xplotSKF_2=[xplotSKF_2 geo_mean(x_RMSE(:))];
    yplotSKF_2=[yplotSKF_2 mean(log(x_RMSE(2,:)))];
    xerrorplotSKF_2=[xerrorplotSKF_2 geo_mean(x_error_self_est(:))];
    yerrorplotSKF_2=[yerrorplotSKF_2 mean(log(x_error_self_est(2,:)))];
    Time_SKF_2=[Time_SKF_2 Runtime];
    if geo_mean(x_RMSE(:))<min_RMSE(5)
        beta0(5)=beta;
        min_RMSE(5)=geo_mean(x_RMSE(:));
    end
end


[x_RMSE, ~, x_error_self_est] = simulation(1,magnitude,repeat,0,beta0(1));
x_EKF_2=x_RMSE(1,:);
y_EKF_2=x_RMSE(2,:);
x_EKF_2_self=x_error_self_est(1,:);
y_EKF_2_self=x_error_self_est(2,:);
[x_RMSE, ~, x_error_self_est] = simulation(1,magnitude,repeat,3,beta0(2));
x_EKF2_2=x_RMSE(1,:);
y_EKF2_2=x_RMSE(2,:);
x_EKF2_2_self=x_error_self_est(1,:);
y_EKF2_2_self=x_error_self_est(2,:);
[x_RMSE, ~, x_error_self_est] = simulation(1,magnitude,repeat,1,beta0(3));
x_CKF_2=x_RMSE(1,:);
y_CKF_2=x_RMSE(2,:);
x_CKF_2_self=x_error_self_est(1,:);
y_CKF_2_self=x_error_self_est(2,:);
[x_RMSE, ~, x_error_self_est] = simulation(1,magnitude,repeat,1.1,beta0(4));
x_UKF_2=x_RMSE(1,:);
y_UKF_2=x_RMSE(2,:);
x_UKF_2_self=x_error_self_est(1,:);
y_UKF_2_self=x_error_self_est(2,:);
[x_RMSE, ~, x_error_self_est] = simulation(1,magnitude,repeat,4,beta0(5));
x_SKF_2=x_RMSE(1,:);
y_SKF_2=x_RMSE(2,:);
x_SKF_2_self=x_error_self_est(1,:);
y_SKF_2_self=x_error_self_est(2,:);

%% plot
% x = 0:10:1200;
% y = 0:10:1200;
% 
% % Create a grid of x and y values
% [X, Y] = meshgrid(x, y);
% 
% % Define the plane equation
% Z = 100*sin(sqrt((X/40).^2 + (Y/40).^2));
% 
% % Plot the plane using surf function
% f2=figure;
% surf(X, Y, Z);
% 
% % Add colorbar to represent z values
% colorbar;
% 
% % Add labels and title
% xlabel('X-axis (km)');
% ylabel('Y-axis (km)');
% zlabel('Z-axis');
% title("Altitude map")
% view(2)
% %exportgraphics(f2,'Terrain-referenced-navigation-x.png','Resolution',900)
%%
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
%ylabel('Average of log x-position RMSE [log(m)]',FontSize=12)
%ylim([10^-3 11])
%xlim([10^-4 100])
%set(gca, 'YTick', [0.001 0.01 0.1 1 10 100]);
%set(gca, 'XTick', [0.0001 0.001 0.01 0.1 1 10]);

%leg=legend([h1_1 h2_1], 'Location','northeast',FontSize=11);
%title(leg,'Line styles')
% grid on
% 
% nexttile
% hold on
% h1_2=plot(beta_range, yplotEKF_2, '-o', 'Color', '#0072BD','LineWidth',1.5);
% h1_1=plot(beta_range, yerrorplotEKF_2, '--o', 'Color', '#0072BD','LineWidth',1.5);
% h2_2=plot(beta_range, yplotEKF2_2, '-s', 'Color', "#D95319",'LineWidth',1.5);
% h2_1=plot(beta_range, yerrorplotEKF2_2, '--s', 'Color', "#D95319",'LineWidth',1.5);
% h3_2=plot(beta_range, yplotCKF_2, '-^', 'Color', "#EDB120",'LineWidth',1.5);
% h3_1=plot(beta_range, yerrorplotCKF_2, '--^', 'Color', "#EDB120",'LineWidth',1.5);
% h4_2=plot(beta_range, yplotSKF_2, '-x', 'Color', "#7E2F8E",'LineWidth',1.5);
% h4_1=plot(beta_range, yerrorplotSKF_2, '--x', 'Color', "#7E2F8E",'LineWidth',1.5);
% h5_2=plot(beta_range, yplotUKF_2, '-d', 'Color', "#4DBEEE",'LineWidth',1.5);
% h5_1=plot(beta_range, yerrorplotUKF_2, '--d', 'Color', "#4DBEEE",'LineWidth',1.5);
h1_1=plot(nan, nan, 'o', 'Color', '#0072BD', 'DisplayName', 'EKF','LineWidth',1.5);
h2_1=plot(nan, nan, 's', 'Color', "#D95319", 'DisplayName', 'EKF2','LineWidth',1.5);
h3_1=plot(nan, nan, '^', 'Color', "#EDB120", 'DisplayName', 'CKF','LineWidth',1.5);
h4_1=plot(nan, nan, 'x', 'Color', "#7E2F8E", 'DisplayName', 'SKF','LineWidth',1.5);
h5_1=plot(nan, nan, 'd', 'Color', "#4DBEEE", 'DisplayName', 'SSKF','LineWidth',1.5);
h1_2=plot(nan, nan, '--', 'Color', 'k', 'DisplayName', 'Estimated error','LineWidth',1.5);
h2_2=plot(nan, nan, 'Color', "k", 'DisplayName', 'Actual error','LineWidth',1.5);
set(gca, 'XScale', 'log', 'YScale', 'log','FontSize', 12)
xlabel('Beta',FontSize=12)
ylabel('Geometric mean of RMSE',FontSize=12)
%ylabel('Average of log speed RMSE [log(m/s)]',FontSize=12)
%ylim([10^-4 1.5])
%xlim([10^-4 100])
%set(gca, 'XTick', [0.0001 0.001 0.01 0.1 1 10 100]);
%set(gca, 'YTick', [0.0001 0.001 0.01 0.1 1 10]);
leg=legend([h1_2 h1_1 h2_1 h3_1 h2_2 h4_1 h5_1], 'Location','northeast','NumColumns',2,FontSize=11);
%title(leg,'Line colors')
grid on
exportgraphics(f1,'navigation-beta.png','Resolution',600)

%% function to get DEM data

function h = height(x,y)
h=1000*sin(sqrt((x/40).^2 + (y/40).^2));
end

function H = dheight(x,y)
H=1000*cos(sqrt((x/40)^2 + (y/40)^2))/sqrt((x/40)^2 + (y/40)^2)*[x/1600 y/1600];
end

function H = ddheight(x,y)
t=sqrt((x/40)^2 + (y/40)^2);
H=10*[cos(t)/16/t+x/16*(-sin(t)/t-cos(t)/t^2)*x/1600/t x/16*(-sin(t)/t-cos(t)/t^2)*y/1600/t;
    x/16*(-sin(t)/t-cos(t)/t^2)*y/1600/t cos(t)/16/t+y/16*(-sin(t)/t-cos(t)/t^2)*y/1600/t];
end
%% main simulation function
function [x_RMSE, Runtime, x_error_self_est]=simulation(improvement,magnitude,M,KFtype,beta)
rng(123)
N = 100; % number of time steps
dt = 1; % time between time steps
% M is the number of Monte-Carlo runs
sig_pro_true = 0.5*[1e-3; 1e-3]; % true value of standard deviation of process noise
sig_mea_true = magnitude; % true value of standard deviation of measurement noise

sig_pro = sig_pro_true; % user input of standard deviation of process noise
sig_mea = sig_mea_true; % user input of standard deviation of measurement noise

sig_init = [1; 1]; % standard deviation of initial guess

Q = diag(sig_pro.^2); % process noise covariance matrix
R = diag(sig_mea.^2); % measurement noise covariance matrix

F = eye(2); % state transition matrix
B = eye(2); % control-input matrix

%% true trajectory

% aircraft trajectory
x_true = zeros(2,N+1);
x_true(:,1) = [10; 10]; % initial true state
u = [0.5; 0];
for k = 2:1:N+1
    x_true(:,k) = F*x_true(:,k-1) + B*u;
end

%% Kalman filter simulation
Runtime=0;
res_x_est = zeros(2,N+1,M); % Monte-Carlo estimates
res_x_err = zeros(2,N+1,M); % Monte-Carlo estimate errors
x_error_self_est=zeros(2,N+1);
P_diag = zeros(2,N+1); % diagonal term of error covariance matrix
n=2;
mnum=1;
points=[diag(ones(n,1));ones(1,n)/(sqrt(n+1)-1)]*sqrt(n+1);
%points=[diag(ones(n,1))]*sqrt(n);
points=points-mean(points);
points=[points;zeros(1,n)];
% filtering
for m = 1:1:M
    % initial guess
    x_est(:,1) = x_true(:,1) + normrnd(0, sig_init);
    P = diag(sig_init.^2);
    P_diag(:,1) = diag(P);
    x_error_self_est(1)=x_error_self_est(1) + P(1,1)/M;
    x_error_self_est(2)=x_error_self_est(2) + P(2,2)/M;
    for k = 2:1:N+1
        
        %%% Prediction
        u_p = u + normrnd(0, sig_pro_true);
        % obtain measurement
        z = height(x_true(1,k), x_true(2,k)) + normrnd(0, sig_mea_true);
        tic;
        if(KFtype==0)
            % translation


            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u_p;

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update

            % predicted meausrement
            z_p = height(x_est(1,k), x_est(2,k));

            % measurement residual
            y = z - z_p;

            % measurement matrix
            H = dheight(x_est(1,k),x_est(2,k));

            % Kalman gain
            K = P*H'/(R+H*P*H');

            % updated state estimate
            x_est0 = x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*y;

            % updated error covariance
            if(improvement==1)
                H2 = dheight(x_est(1,k),x_est(2,k));
                temp=P-K*H2*P-P*H2'*K'+K*(H2*P*H2'+R)*K';
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end
            else
                P = (eye(2) - K*H)*P;
            end
        elseif(KFtype==0.5)
            % translation


            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u_p;

            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update

            % predicted meausrement
            z_p = height(x_est(1,k), x_est(2,k));

            % measurement residual
            y = z - z_p;

            % measurement matrix
            H = dheight(x_est(1,k),x_est(2,k));

            % Kalman gain
            K = P*H'/(R+H*P*H');

            % updated state estimate
            x_est0 = x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*y;
            %IEKF
            steplennow = norm(K*y);
            change = 1;
            iter = 1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                z_p = height(x_est(1,k), x_est(2,k));
                y = z - z_p;
                H2 = dheight(x_est(1,k),x_est(2,k));
                S = H2*P*H2.'+ R;
                K2 = P*H2.'*S^(-1);
                dx = x_est0 + K2*(y-H2*(x_est0-x_est(:,k))) - x_est(:,k);
                steplen_previous=steplennow;
                steplennow=norm(dx);
                if(steplen_previous<steplennow)
                    break;
                else
                    change = max(abs(dx./x_est(:,k)));
                    x_est(:,k) = x_est(:,k) + dx;
                    K = K2;
                    H = H2;
                end
            end
            % updated error covariance
            if(improvement==1)
                H2 = dheight(x_est(1,k),x_est(2,k));
                temp=P-K*H2*P-P*H2'*K'+K*(H2*P*H2'+R)*K';
                if(sum(diag(temp))>sum(diag(P)))
                    x_est(:,k)=x_est0;
                else
                    P=temp;
                end
            else
                P = (eye(2) - K*H)*P;
            end
        elseif(KFtype==1)
            L=chol(P).';
            state=x_est(:,k-1);
            n=2;
            lambda=(1e-6-1)*n;
            alpha=1e-3;
            mnum=1;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
            end
            for i=1:2*n+1
                states(:,i)=F*states(:,i) + B*u_p;
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
            % Predict Measurement From Propagated Sigma Points
            L=chol(P).';
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
            end
            measures=zeros(mnum,2*n+1);
            for i=1:2*n+1
                measures(:,i)=height(states(1,i), states(2,i));
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
                    measures(:,i)=height(states(1,i), states(2,i));
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
        elseif(KFtype==1.1)
            L=chol(P).';
            state=x_est(:,k-1);
            n=2;
            
            alpha=1;
            lambda=(alpha^2-1)*n;
            mnum=1;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
            end
            for i=1:2*n+1
                states(:,i)=F*states(:,i) + B*u_p;
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
            % Predict Measurement From Propagated Sigma Points
            L=chol(P).';
            states(:,1)=state;
            for i=1:n
                states(:,1+i)=state+sqrt(lambda+n)*L(:,i);
                states(:,n+i+1)=state-sqrt(lambda+n)*L(:,i);
            end
            measures=zeros(mnum,2*n+1);
            for i=1:2*n+1
                measures(:,i)=height(states(1,i), states(2,i));
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
                    measures(:,i)=height(states(1,i), states(2,i));
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
            %new KF
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
            %update sigma points
            L=chol(P).';
            states=repmat(state, 1, pnum)+alpha*(points*L')';
            % Predict Measurement From Propagated Sigma Points
            measures=zeros(mnum,pnum);
            m_exp=0;
            for i=1:pnum
                measures(:,i)=height(states(1,i), states(2,i));
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
                    measures(:,i)=height(states(1,i), states(2,i));
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
            %SKF
            L=chol(P).';
            state=x_est(:,k-1);
            n=2;
            mnum=1;
            states=zeros(n,n*2);
            for i=1:n
                states(:,i)=state+sqrt(n)*L(:,i);
                states(:,n+i)=state-sqrt(n)*L(:,i);
            end
            state=0;
            for i=1:2*n
                states(:,i)=F*states(:,i) + B*u_p;
                state=state+states(:,i)/n/2;
            end
            P=Q;
            for i=1:2*n
                P=P+(state-states(:,i))*(state-states(:,i)).'/n/2;
            end
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
                measures(:,i) = height(states(1,i), states(2,i)); % predicted measurement
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
                    measures(:,i) = height(states(1,i), states(2,i)); % predicted measurement
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
            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u_p;
            n=length(x_est(:,k));
            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            mnum=1;
            Hxx=zeros(n,n,mnum);
            Hxx(:,:,1)=ddheight(x_est(1,k),x_est(2,k));

            % measurement matrix
            H = dheight(x_est(1,k),x_est(2,k));
            S=H*P*H.'+R;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.25*beta*sum(diag(Hxx(:,:,ii)*P*Hxx(:,:,jj)*P));
                end
            end
            % Kalman gain
            K=P*H.'*S^(-1);
            % predicted meausrement
            z_p = height(x_est(1,k), x_est(2,k));
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
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=ddheight(x_est(1,k),x_est(2,k));
                H2 = dheight(x_est(1,k),x_est(2,k));
                S2=H2*P*H2.'+R;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.25*beta*sum(diag(Hxx2(:,:,ii)*P*Hxx2(:,:,jj)*P));
                    end
                end
                temp=P+K*S2*K.'-P*H2.'*K.'-K*H2*P;
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    x_est(:,k)=x_est0;
                end
            else
                P = P-K*S*K.';
            end
        else
            %2-IEKF
            % predicted state estimate
            x_est(:,k) = F*x_est(:,k-1) + B*u_p;
            n=length(x_est(:,k));
            % predicted error covariance
            P = F*P*F' + Q;

            %%% Update
            mnum=1;
            Hxx=zeros(n,n,mnum);
            Hxx(:,:,1)=ddheight(x_est(1,k),x_est(2,k));

            % measurement matrix
            H = dheight(x_est(1,k),x_est(2,k));
            S=H*P*H.'+R;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.5*sum(diag(Hxx(:,:,ii)*P*Hxx(:,:,jj)*P));
                end
            end
            % Kalman gain
            K=P*H.'*S^(-1);
            % predicted meausrement
            z_p = height(x_est(1,k), x_est(2,k));
            % measurement residual
            residual=z-z_p;
            for ii=1:mnum
                residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*P));
            end
            % updated state estimate
            x_est0 = x_est(:,k);
            x_est(:,k) = x_est(:,k) + K*residual;
            %IEKF
            steplennow = norm(K*residual);
            change = 1;
            iter = 1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                z_p = height(x_est(1,k), x_est(2,k));
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=ddheight(x_est(1,k),x_est(2,k));
                residual = z - z_p;
                for ii=1:mnum
                    residual(ii)=residual(ii)-0.5*(x_est0-x_est(:,k)).'*Hxx2(:,:,ii)*(x_est0-x_est(:,k));
                    residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*P));
                end
                H2 = dheight(x_est(1,k),x_est(2,k));
                S2=H2*P*H2.'+R;
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
                else
                    change = max(abs(dx./x_est(:,k)));
                    x_est(:,k) = x_est(:,k) + dx;
                    K = K2;
                    S = S2;
                end
            end
            % updated error covariance
            if(improvement==1)
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=ddheight(x_est(1,k),x_est(2,k));
                H2 = dheight(x_est(1,k),x_est(2,k));
                S2=H2*P*H2.'+R;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*sum(diag(Hxx2(:,:,ii)*P*Hxx2(:,:,jj)*P));
                    end
                end
                temp=P+K*S2*K.'-P*H2.'*K.'-K*H2*P;
                if(sum(diag(temp))<sum(diag(P)))
                    P=temp;
                else
                    x_est(:,k)=x_est0;
                end
            else
                P = P-K*S*K.';
            end
        end
        temp = toc;
        Runtime=Runtime + temp/M;
        P_diag(:,k) = diag(P);

        res_d_err(k,m) = norm(x_est(:,k) - x_true(:,k));
        x_error_self_est(1,k)=x_error_self_est(1,k) + P(1,1)/M;
        x_error_self_est(2,k)=x_error_self_est(2,k) + P(2,2)/M;
    end

    res_x_est(:,:,m) = x_est;
    res_x_err(:,:,m) = x_est - x_true;
end
x_error_self_est = sqrt(x_error_self_est);
%% get result statistics

x_est_avg = mean(res_x_est,3);
x_err_avg = mean(res_x_err,3);
x_RMSE = zeros(2,N+1); % root mean square error
for k = 1:1:N+1
    x_RMSE(1,k) = sqrt(mean(res_x_err(1,k,:).^2,3));
    x_RMSE(2,k) = sqrt(mean(res_x_err(2,k,:).^2,3));
end
time = (0:1:N)*dt;
end