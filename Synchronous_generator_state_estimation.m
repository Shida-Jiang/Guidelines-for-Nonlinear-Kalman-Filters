% Application 3: Synchronous Generator State Estimation
close all
clc
clear
repeat=10000; %The number of times that the simulation is repeated at each measurement noise setup
%This simulation takes about 30 minutes, you can reduce the repeat times to reduce the run time
%%
noise1_ref=1;
scale=-4; % The range of measurement noise is 10^scale
beta0 = NaN(1,5); % show how the estimation convergences when the std of the measurement noise is 10^-2 
min_RMSE=9999*ones(1,5);
beta_range=10.^(-2:0.2:2);
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
    magnitude=noise1_ref*10^scale;

    
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

%% figures
f2 = figure('Position',[100 100 600 300]);  % [left bottom width height]
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

h1_1=plot(nan, nan, 'o', 'Color', '#0072BD', 'DisplayName', 'EKF','LineWidth',1.5);
h2_1=plot(nan, nan, 's', 'Color', "#D95319", 'DisplayName', 'EKF2','LineWidth',1.5);
h3_1=plot(nan, nan, '^', 'Color', "#EDB120", 'DisplayName', 'CKF*','LineWidth',1.5);
h4_1=plot(nan, nan, 'x', 'Color', "#7E2F8E", 'DisplayName', 'SKF*','LineWidth',1.5);
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
ylim([3.9 7]/10000);
exportgraphics(f2,'generator-beta.png','Resolution',600)
function [x_RMSE, Runtime, x_error_self_est]=simulation(improvement1,noise1,repeat,KFtype,beta)
%% Problem setup
delta_t=1e-4;
w0=2*60*pi;
J=13;
x_d_dot=0.375;
x_d=2.06;
x_q_dot=0.375;
x_q=1.214;
T_do=0.131;
T_qo=0.0131;
D=0.05;
Tlimit=1e-2;
x1_rmse=0;
x2_rmse=0;
x3_rmse=0;
x4_rmse=0;
process_sigma=[1e-5;1e-8;1e-5;1e-5];
Q=diag(process_sigma.^2);
sig_init = [0.01; 1e-5; 0.01; 0.01]; % standard deviation of initial guess
rng(123)
Runtime=0;
n=4;
mnum=1;
t=delta_t:delta_t:Tlimit;
x_error_self_est=zeros(n,length(t)+1);
res_x_err = zeros(4,length(t)+1,repeat); % Monte-Carlo estimate errors
points=[diag(ones(n,1));ones(1,n)/(sqrt(n+1)-1)]*sqrt(n+1);
points=points-mean(points);
points=[points;zeros(1,n)];
for iii=1:repeat
    states_true=[0.4;0;0;0];
    states_est=states_true+normrnd(0, sig_init);
    Variance=diag(sig_init.^2);
    %% generate profile
    Measurementnoise=noise1^2;
    v=zeros(3,length(t));
    v(1,:)=0.8*ones(1,length(t));
    v(2,:)=2.11+2*(t-delta_t);
    v(3,:)=1.002*ones(1,length(t));
    stepnum=length(t);
    Measurements=[];
    for k=1:stepnum
        states=states_true(:,k);
        states=states+delta_t*[w0*states(2);
            (v(1,k)-D*states(2)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*sin(2*states(1))*(1/x_q-1/x_d_dot)))/J;
            (v(2,k)-states(3)-(x_d-x_d_dot)*(states(3)-v(3,k)*cos(states(1)))/x_d_dot)/T_do;
            (-states(4)+(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q)/T_qo]+process_sigma.*randn(4,1);
        states_true=[states_true states];
        measurement=v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1));
        Measurements=[Measurements measurement];
    end
    x=states_true(1,:);
    y=states_true(2,:);
    Measurements = [0 Measurements];
    Measurements_noisy=Measurements+noise1*randn(1,stepnum+1);
    x_error_self_est(1)=x_error_self_est(1) + Variance(1,1)/repeat;
    x_error_self_est(2)=x_error_self_est(2) + Variance(2,2)/repeat;
    x_error_self_est(3)=x_error_self_est(3) + Variance(3,3)/repeat;
    x_error_self_est(4)=x_error_self_est(4) + Variance(4,4)/repeat;
    %% KF
    tic;
    for k=1:stepnum
        if(KFtype==0)
            states=states_est(:,k);
            F=diag([1 1 1 1])+delta_t*[0 w0 0 0;
                -(v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*cos(2*states(1))*(1/x_q-1/x_d_dot))/J -D/J -v(3,k)/x_d_dot*sin(states(1))/J 0;
                -(x_d-x_d_dot)*v(3,k)*sin(states(1))/x_d_dot/T_do 0 -(1+(x_d-x_d_dot)/x_d_dot)/T_do 0;
                (x_q-x_q_dot)*v(3,k)*cos(states(1))/x_q/T_qo 0 0 -1/T_qo];
            states=states+delta_t*[w0*states(2);
            (v(1,k)-D*states(2)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*sin(2*states(1))*(1/x_q-1/x_d_dot)))/J;
            (v(2,k)-states(3)-(x_d-x_d_dot)*(states(3)-v(3,k)*cos(states(1)))/x_d_dot)/T_do;
            (-states(4)+(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q)/T_qo];
            Variance=F*Variance*F.'+Q;
            residual=Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
            H=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
            S=H*Variance*H.'+Measurementnoise;
            K=Variance*H.'*S^(-1);
            states0=states;
            states=states+K*residual;
            if(improvement1>=1)
                H2=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                temp=Variance-K*H2*Variance-Variance*H2'*K'+K*(H2*Variance*H2'+Measurementnoise)*K';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    states=states0;
                end
            else
                Variance=Variance-K*H*Variance;
            end
            states_est(:,k+1)=states;
        elseif(KFtype==0.5)
            states=states_est(:,k);
            F=diag([1 1 1 1])+delta_t*[0 w0 0 0;
                -(v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*cos(2*states(1))*(1/x_q-1/x_d_dot))/J -D/J -v(3,k)/x_d_dot*sin(states(1))/J 0;
                -(x_d-x_d_dot)*v(3,k)*sin(states(1))/x_d_dot/T_do 0 -(1+(x_d-x_d_dot)/x_d_dot)/T_do 0;
                (x_q-x_q_dot)*v(3,k)*cos(states(1))/x_q/T_qo 0 0 -1/T_qo];
            states=states+delta_t*[w0*states(2);
            (v(1,k)-D*states(2)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*sin(2*states(1))*(1/x_q-1/x_d_dot)))/J;
            (v(2,k)-states(3)-(x_d-x_d_dot)*(states(3)-v(3,k)*cos(states(1)))/x_d_dot)/T_do;
            (-states(4)+(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q)/T_qo];
            Variance=F*Variance*F.'+Q;
            residual=Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
            H=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
            S=H*Variance*H.'+Measurementnoise;
            K=Variance*H.'*S^(-1);
            states0=states;
            states=states+K*residual;
            steplennow = norm(K*residual);
            change = 1;
            iter = 1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                residual = Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
                H2 = [v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                S2 = H2*Variance*H2.'+Measurementnoise;
                K2 = Variance*H2.'*S2^(-1);
                dx = states0 + K2*(residual-H2*(states0-states))-states;
                steplen_previous=steplennow;
                steplennow=norm(dx);
                if(steplen_previous<steplennow)
                    break;
                else
                    change = max(abs(dx./states));
                    states = states + dx;
                    K = K2;
                    H = H2;
                end
            end
            if(improvement1>=1)
                H2=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                temp=Variance-K*H2*Variance-Variance*H2'*K'+K*(H2*Variance*H2'+Measurementnoise)*K';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    states=states0;
                end
            else
                Variance=Variance-K*H*Variance;
            end
            states_est(:,k+1)=states;
        elseif(KFtype==1)
            L=chol(Variance).';
            state=states_est(:,k);
            n=4;
            lambda=(1e-6-1)*n;
            alpha=1e-3;
            mnum=1;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for ii=1:n
                states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
                states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);  
            end
            for ii=1:2*n+1
                states(:,ii)=states(:,ii)+delta_t*[w0*states(2,ii);
                    (v(1,k)-D*states(2,ii)-(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*sin(2*states(1,ii))*(1/x_q-1/x_d_dot)))/J;
                    (v(2,k)-states(3,ii)-(x_d-x_d_dot)*(states(3,ii)-v(3,k)*cos(states(1,ii)))/x_d_dot)/T_do;
                    (-states(4,ii)+(x_q-x_q_dot)*v(3,k)*sin(states(1,ii))/x_q)/T_qo];
                if(ii==1)
                    state=states(:,ii)*lambda/(n+lambda);
                else
                    state=state+states(:,ii)/(n+lambda)/2;
                end
            end
            Variance=(states(:,1)-state)*(states(:,1)-state).'*(lambda/(n+lambda)+1-alpha^2+beta)+Q;
            for ii=2:2*n+1
                Variance=Variance+(state-states(:,ii))*(state-states(:,ii)).'/(n+lambda)/2;
            end
            % obtain measurement
            z = Measurements_noisy(:,k+1);
            % Predict Measurement From Propagated Sigma Points
            L=chol(Variance).';
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for ii=1:n
                states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
                states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);  
            end
            measures=zeros(mnum,2*n+1);
            for ii=1:2*n+1
                measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                if(ii==1)
                    m_exp=lambda/(n+lambda)*measures(:,ii);
                else
                    m_exp=m_exp+1/(n+lambda)/2*measures(:,ii);
                end
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+Measurementnoise;
            Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
            for ii=2:2*n+1
                Py=Py+1/(n+lambda)/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                Pxy=Pxy+1/(n+lambda)/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            %update
            if(improvement1>=1)
%                 L=chol(Variance).';
%                 states=zeros(n,n*2+1);
%                 states(:,1)=state;
%                 for ii=1:n
%                     states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
%                     states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);
%                 end
                for ii=1:2*n+1
                    states(:,ii)=states(:,ii)+dstate;
                end
                measures=zeros(mnum,2*n+1);
                for ii=1:2*n+1
                    measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                    if(ii==1)
                        m_exp=lambda/(n+lambda)*measures(:,ii);
                    else
                        m_exp=m_exp+1/(n+lambda)/2*measures(:,ii);
                    end
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+Measurementnoise;
                Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
                for ii=2:2*n+1
                    Py=Py+1/(n+lambda)/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                    Pxy=Pxy+1/(n+lambda)/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
                end
                temp=Variance+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    state=state0;
                end
            else
                Variance=Variance-K*Py*K.';
            end

            states_est(:,k+1)=state;
        elseif(KFtype==1.1)
            L=chol(Variance).';
            state=states_est(:,k);
            alpha=1;
            lambda=(alpha^2-1)*n;
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for ii=1:n
                states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
                states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);  
            end
            for ii=1:2*n+1
                states(:,ii)=states(:,ii)+delta_t*[w0*states(2,ii);
                    (v(1,k)-D*states(2,ii)-(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*sin(2*states(1,ii))*(1/x_q-1/x_d_dot)))/J;
                    (v(2,k)-states(3,ii)-(x_d-x_d_dot)*(states(3,ii)-v(3,k)*cos(states(1,ii)))/x_d_dot)/T_do;
                    (-states(4,ii)+(x_q-x_q_dot)*v(3,k)*sin(states(1,ii))/x_q)/T_qo];
                if(ii==1)
                    state=states(:,ii)*lambda/(n+lambda);
                else
                    state=state+states(:,ii)/(n+lambda)/2;
                end
            end
            Variance=(states(:,1)-state)*(states(:,1)-state).'*(lambda/(n+lambda)+1-alpha^2+beta)+Q;
            for ii=2:2*n+1
                Variance=Variance+(state-states(:,ii))*(state-states(:,ii)).'/(n+lambda)/2;
            end
            % obtain measurement
            z = Measurements_noisy(:,k+1);
            % Predict Measurement From Propagated Sigma Points
            L=chol(Variance).';
            states=zeros(n,n*2+1);
            states(:,1)=state;
            for ii=1:n
                states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
                states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);  
            end
            measures=zeros(mnum,2*n+1);
            for ii=1:2*n+1
                measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                if(ii==1)
                    m_exp=lambda/(n+lambda)*measures(:,ii);
                else
                    m_exp=m_exp+1/(n+lambda)/2*measures(:,ii);
                end
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+Measurementnoise;
            Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
            for ii=2:2*n+1
                Py=Py+1/(n+lambda)/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                Pxy=Pxy+1/(n+lambda)/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            %update
            if(improvement1>=1)
%                 L=chol(Variance).';
%                 states=zeros(n,n*2+1);
%                 states(:,1)=state;
%                 for ii=1:n
%                     states(:,1+ii)=state+sqrt(lambda+n)*L(:,ii);
%                     states(:,n+ii+1)=state-sqrt(lambda+n)*L(:,ii);
%                 end
                for ii=1:2*n+1
                    states(:,ii)=states(:,ii)+dstate;
                end
                measures=zeros(mnum,2*n+1);
                for ii=1:2*n+1
                    measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                    if(ii==1)
                        m_exp=lambda/(n+lambda)*measures(:,ii);
                    else
                        m_exp=m_exp+1/(n+lambda)/2*measures(:,ii);
                    end
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=(lambda/(n+lambda)+1-alpha^2+beta)*(measures(:,1)-m_exp)*(measures(:,1)-m_exp).'+Measurementnoise;
                Pxy=(lambda/(n+lambda)+1-alpha^2+beta)*(states(:,1)-state)*(measures(:,1)-m_exp).';
                for ii=2:2*n+1
                    Py=Py+1/(n+lambda)/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                    Pxy=Pxy+1/(n+lambda)/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
                end
                temp=Variance+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    state=state0;
                end
            else
                Variance=Variance-K*Py*K.';
            end
            
            states_est(:,k+1)=state;
        elseif(KFtype==4)
            L=chol(Variance).';
            state=states_est(:,k);
            pnum=n+2;
            alpha=1;
            mnum=1;
            states=repmat(state, 1, pnum)+alpha*(points*L')';
            weight=ones(pnum,1)/(n+1)/alpha^2;
            weight(pnum)=1-1/alpha^2;
            state=0;
            for ii=1:pnum
                states(:,ii)=states(:,ii)+delta_t*[w0*states(2,ii);
                    (v(1,k)-D*states(2,ii)-(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*sin(2*states(1,ii))*(1/x_q-1/x_d_dot)))/J;
                    (v(2,k)-states(3,ii)-(x_d-x_d_dot)*(states(3,ii)-v(3,k)*cos(states(1,ii)))/x_d_dot)/T_do;
                    (-states(4,ii)+(x_q-x_q_dot)*v(3,k)*sin(states(1,ii))/x_q)/T_qo];
                state=state+states(:,ii)*weight(ii);
            end
            dP=(states(:,pnum)-state)*(states(:,pnum)-state)';
            Variance=Q;
            for ii=1:pnum
                Variance=Variance+(state-states(:,ii))*(state-states(:,ii)).'*weight(ii);
            end
            Variance=Variance+beta*dP;
            % obtain measurement
            z = Measurements_noisy(:,k+1);
            % Predict Measurement From Propagated Sigma Points
            L=chol(Variance).';
            states=repmat(state, 1, pnum)+alpha*(points*L')';
            measures=zeros(mnum,pnum);
            m_exp=0;
            for ii=1:pnum
                measures(:,ii)=(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                m_exp=m_exp+weight(ii)*measures(:,ii);
            end
            % Estimate Mean And Covariance of Predicted Measurement
            dP=(measures(:,pnum)-m_exp)*(measures(:,pnum)-m_exp)';
            Py=Measurementnoise;
            Pxy=0;
            for i=1:pnum
                Py=Py+weight(i)*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                Pxy=Pxy+weight(i)*(states(:,i)-state)*(measures(:,i)-m_exp).';
            end
            Py   = Py + beta*dP;
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            dstate=K*(z-m_exp);
            state=state+dstate;
            if(improvement1>=1)
                for ii=1:pnum
                    states(:,ii)=states(:,ii)+dstate;
                end
                measures=zeros(mnum,pnum);
                m_exp=0;
                for ii=1:pnum
                    measures(:,ii)=(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                    m_exp=m_exp+weight(ii)*measures(:,ii);
                end
                % Estimate Mean And Covariance of Predicted Measurement
                dP=(measures(:,pnum)-m_exp)*(measures(:,pnum)-m_exp)';
                Py=Measurementnoise;
                Pxy=0;
                for i=1:pnum
                    Py=Py+weight(i)*(measures(:,i)-m_exp)*(measures(:,i)-m_exp).';
                    Pxy=Pxy+weight(i)*(states(:,i)-state)*(measures(:,i)-m_exp).';
                end
                Py   = Py + beta*dP;
                temp=Variance+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    state=state0;
                end
            else     
                Variance=Variance-K*Py*K.';
            end
            states_est(:,k+1)=state;
        elseif(KFtype==2)
            %CKF
            L=chol(Variance).';
            state=states_est(:,k);
            n=4;
            mnum=1;
            states=zeros(n,n*2);
            for ii=1:n
                states(:,ii)=state+sqrt(n)*L(:,ii);
                states(:,n+ii)=state-sqrt(n)*L(:,ii);  
            end
            state=0;
            for ii=1:2*n
                states(:,ii)=states(:,ii)+delta_t*[w0*states(2,ii);
                    (v(1,k)-D*states(2,ii)-(v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*sin(2*states(1,ii))*(1/x_q-1/x_d_dot)))/J;
                    (v(2,k)-states(3,ii)-(x_d-x_d_dot)*(states(3,ii)-v(3,k)*cos(states(1,ii)))/x_d_dot)/T_do;
                    (-states(4,ii)+(x_q-x_q_dot)*v(3,k)*sin(states(1,ii))/x_q)/T_qo];
                state=state+states(:,ii)/n/2;
            end
            Variance=Q;
            for ii=1:2*n
                Variance=Variance+(state-states(:,ii))*(state-states(:,ii)).'/n/2;
            end
            % obtain measurement
            z = Measurements_noisy(:,k+1);
            % Predict Measurement From Propagated Sigma Points
            L=chol(Variance).';
            states=zeros(n,n*2);
            for ii=1:n
                states(:,ii)=state+sqrt(n)*L(:,ii);
                states(:,n+ii)=state-sqrt(n)*L(:,ii);
            end
            measures=zeros(mnum,2*n);
            m_exp=0;
            for ii=1:2*n
                measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                m_exp=m_exp+1/n/2*measures(:,ii);
            end
            % Estimate Mean And Covariance of Predicted Measurement
            Py=Measurementnoise;
            Pxy=0;
            for ii=1:2*n
                Py=Py+1/n/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                Pxy=Pxy+1/n/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
            end
            %kalman gain
            K=Pxy/Py;
            %update
            state0=state;
            state=state+K*(z-m_exp);
            if(improvement1>=1)
                states=zeros(n,n*2);
                for ii=1:n
                    states(:,ii)=state+sqrt(n)*L(:,ii);
                    states(:,n+ii)=state-sqrt(n)*L(:,ii);
                end
                measures=zeros(mnum,2*n);
                m_exp=0;
                for ii=1:2*n
                    measures(:,ii) = (v(3,k)/x_d_dot*states(3,ii)*sin(states(1,ii))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1,ii))); % predicted measurement
                    m_exp=m_exp+1/n/2*measures(:,ii);
                end
                % Estimate Mean And Covariance of Predicted Measurement
                Py=Measurementnoise;
                Pxy=0;
                for ii=1:2*n
                    Py=Py+1/n/2*(measures(:,ii)-m_exp)*(measures(:,ii)-m_exp).';
                    Pxy=Pxy+1/n/2*(states(:,ii)-state)*(measures(:,ii)-m_exp).';
                end
                temp=Variance+K*Py*K.'-Pxy*K.'-K*Pxy.';
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    state=state0;
                end
            else
                Variance=Variance-K*Py*K.';
            end
            states_est(:,k+1)=state; 
            
        elseif(KFtype==3)
            %2-EKF
            states=states_est(:,k);
            states=states+delta_t*[w0*states(2);
            (v(1,k)-D*states(2)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*sin(2*states(1))*(1/x_q-1/x_d_dot)))/J;
            (v(2,k)-states(3)-(x_d-x_d_dot)*(states(3)-v(3,k)*cos(states(1)))/x_d_dot)/T_do;
            (-states(4)+(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q)/T_qo];
            n=length(states);
            Fxx=zeros(n,n,n);
            deltaP=zeros(n,n);
            Fxx(1,1,2)=(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2*2*sin(2*states(1))*(1/x_q-1/x_d_dot))/J*delta_t;
            Fxx(1,3,2)=-v(3,k)/x_d_dot*cos(states(1))/J*delta_t;
            Fxx(3,1,2)=-v(3,k)/x_d_dot*cos(states(1))/J*delta_t;
            Fxx(1,1,3)=-(x_d-x_d_dot)*v(3,k)*cos(states(1))/x_d_dot/T_do*delta_t;
            Fxx(1,1,4)=-(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q/T_qo*delta_t;
            for ii=1:n
                states(ii)=states(ii)+0.5*sum(diag(Fxx(:,:,ii)*Variance));
                for jj=1:n
                    deltaP(ii,jj)=sum(diag(Fxx(:,:,ii)*Variance*Fxx(:,:,jj)*Variance));
                end
            end
            F=diag([1 1 1 1])+delta_t*[0 w0 0 0;
                -(v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*cos(2*states(1))*(1/x_q-1/x_d_dot))/J -D/J -v(3,k)/x_d_dot*sin(states(1))/J 0;
                -(x_d-x_d_dot)*v(3,k)*sin(states(1))/x_d_dot/T_do 0 -(1+(x_d-x_d_dot)/x_d_dot)/T_do 0;
                (x_q-x_q_dot)*v(3,k)*cos(states(1))/x_q/T_qo 0 0 -1/T_qo];
            Variance=F*Variance*F.'+0.25*beta*deltaP+Q;
            %update
            mnum=1;
            Hxx=zeros(n,n,mnum);
            Hxx(:,:,1)=[-v(3,k)/x_d_dot*states(3)*sin(states(1))-2*v(3,k)^2*(1/x_q-1/x_d_dot)*sin(2*states(1)) 0 v(3,k)/x_d_dot*cos(states(1)) 0;
                0 0 0 0;
                v(3,k)/x_d_dot*cos(states(1)) 0 0 0;
                0 0 0 0];
            H=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
            S=H*Variance*H.'+Measurementnoise;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.25*beta*sum(diag(Hxx(:,:,ii)*Variance*Hxx(:,:,jj)*Variance));
                end
            end
            K=Variance*H.'*S^(-1);
            states0=states;
            residual=Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
            for ii=1:mnum
                residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*Variance));
            end
            states=states+K*residual;
            if(improvement1>=1)
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=[-v(3,k)/x_d_dot*states(3)*sin(states(1))-2*v(3,k)^2*(1/x_q-1/x_d_dot)*sin(2*states(1)) 0 v(3,k)/x_d_dot*cos(states(1)) 0;
                    0 0 0 0;
                    v(3,k)/x_d_dot*cos(states(1)) 0 0 0;
                    0 0 0 0];
                H2=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                S2=H2*Variance*H2.'+Measurementnoise;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.25*beta*sum(diag(Hxx2(:,:,ii)*Variance*Hxx2(:,:,jj)*Variance));
                    end
                end
                temp=Variance+K*S2*K.'-Variance*H2.'*K.'-K*H2*Variance;
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    states=states0;
                end
            else
                Variance=Variance-K*S*K.';
            end
            states_est(:,k+1)=states;
        else
            %2-IEKF
            states=states_est(:,k);
            states=states+delta_t*[w0*states(2);
            (v(1,k)-D*states(2)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*sin(2*states(1))*(1/x_q-1/x_d_dot)))/J;
            (v(2,k)-states(3)-(x_d-x_d_dot)*(states(3)-v(3,k)*cos(states(1)))/x_d_dot)/T_do;
            (-states(4)+(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q)/T_qo];
            n=length(states);
            Fxx=zeros(n,n,n);
            deltaP=zeros(n,n);
            Fxx(1,1,2)=(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2*2*sin(2*states(1))*(1/x_q-1/x_d_dot))/J*delta_t;
            Fxx(1,3,2)=-v(3,k)/x_d_dot*cos(states(1))/J*delta_t;
            Fxx(3,1,2)=-v(3,k)/x_d_dot*cos(states(1))/J*delta_t;
            Fxx(1,1,3)=-(x_d-x_d_dot)*v(3,k)*cos(states(1))/x_d_dot/T_do*delta_t;
            Fxx(1,1,4)=-(x_q-x_q_dot)*v(3,k)*sin(states(1))/x_q/T_qo*delta_t;
            for ii=1:n
                states(ii)=states(ii)+0.5*sum(diag(Fxx(:,:,ii)*Variance));
                for jj=1:n
                    deltaP(ii,jj)=sum(diag(Fxx(:,:,ii)*Variance*Fxx(:,:,jj)*Variance));
                end
            end
            F=diag([1 1 1 1])+delta_t*[0 w0 0 0;
                -(v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*cos(2*states(1))*(1/x_q-1/x_d_dot))/J -D/J -v(3,k)/x_d_dot*sin(states(1))/J 0;
                -(x_d-x_d_dot)*v(3,k)*sin(states(1))/x_d_dot/T_do 0 -(1+(x_d-x_d_dot)/x_d_dot)/T_do 0;
                (x_q-x_q_dot)*v(3,k)*cos(states(1))/x_q/T_qo 0 0 -1/T_qo];
            Variance=F*Variance*F.'+0.5*deltaP+Q;
            %update
            mnum=1;
            Hxx=zeros(n,n,mnum);
            Hxx(:,:,1)=[-v(3,k)/x_d_dot*states(3)*sin(states(1))-2*v(3,k)^2*(1/x_q-1/x_d_dot)*sin(2*states(1)) 0 v(3,k)/x_d_dot*cos(states(1)) 0;
                0 0 0 0;
                v(3,k)/x_d_dot*cos(states(1)) 0 0 0;
                0 0 0 0];
            H=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
            S=H*Variance*H.'+Measurementnoise;
            for ii=1:mnum
                for jj=1:mnum
                    S(ii,jj)=S(ii,jj)+0.5*sum(diag(Hxx(:,:,ii)*Variance*Hxx(:,:,jj)*Variance));
                end
            end
            K=Variance*H.'*S^(-1);
            states0=states;
            residual=Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
            for ii=1:mnum
                residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*Variance));
            end
            states=states+K*residual;
            steplennow = norm(K*residual);
            change = 1;
            iter = 1;
            while(change>0.001 && iter < 1000)
                iter = iter + 1;
                residual = Measurements_noisy(:,k+1)-(v(3,k)/x_d_dot*states(3)*sin(states(1))+v(3,k)^2/2*(1/x_q-1/x_d_dot)*sin(2*states(1)));
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=[-v(3,k)/x_d_dot*states(3)*sin(states(1))-2*v(3,k)^2*(1/x_q-1/x_d_dot)*sin(2*states(1)) 0 v(3,k)/x_d_dot*cos(states(1)) 0;
                    0 0 0 0;
                    v(3,k)/x_d_dot*cos(states(1)) 0 0 0;
                    0 0 0 0];
                for ii=1:mnum
                    residual(ii)=residual(ii)-0.5*(states0-states).'*Hxx2(:,:,ii)*(states0-states);
                    residual(ii)=residual(ii)-0.5*sum(diag(Hxx(:,:,ii)*Variance));  
                end
                H2 = [v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                S2 = H2*Variance*H2.'+Measurementnoise;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*sum(diag(Hxx2(:,:,ii)*Variance*Hxx2(:,:,jj)*Variance));
                    end
                end
                K2 = Variance*H2.'*S2^(-1);
                dx = states0 + K2*(residual-H2*(states0-states))-states;
                steplen_previous=steplennow;
                steplennow=norm(dx);
                if(steplen_previous<steplennow)
                    break;
                else
                    change = max(abs(dx./states));
                    states = states + dx;
                    K = K2;
                    S = S2;
                end
            end
            if(improvement1>=1)
                Hxx2=zeros(n,n,mnum);
                Hxx2(:,:,1)=[-v(3,k)/x_d_dot*states(3)*sin(states(1))-2*v(3,k)^2*(1/x_q-1/x_d_dot)*sin(2*states(1)) 0 v(3,k)/x_d_dot*cos(states(1)) 0;
                    0 0 0 0;
                    v(3,k)/x_d_dot*cos(states(1)) 0 0 0;
                    0 0 0 0];
                H2=[v(3,k)/x_d_dot*states(3)*cos(states(1))+v(3,k)^2*(1/x_q-1/x_d_dot)*cos(2*states(1)) 0 v(3,k)/x_d_dot*sin(states(1)) 0];
                S2=H2*Variance*H2.'+Measurementnoise;
                for ii=1:mnum
                    for jj=1:mnum
                        S2(ii,jj)=S2(ii,jj)+0.5*sum(diag(Hxx2(:,:,ii)*Variance*Hxx2(:,:,jj)*Variance));
                    end
                end
                temp=Variance+K*S2*K.'-Variance*H2.'*K.'-K*H2*Variance;
                if(sum(diag(temp))<sum(diag(Variance)))
                    Variance=temp;
                else
                    states=states0;
                end
            else
                Variance=Variance-K*S*K.';
            end
            states_est(:,k+1)=states;
        end
        x_error_self_est(1,k+1)=x_error_self_est(1,k+1) + Variance(1,1)/repeat;
        x_error_self_est(2,k+1)=x_error_self_est(2,k+1) + Variance(2,2)/repeat;
        x_error_self_est(3,k+1)=x_error_self_est(3,k+1) + Variance(3,3)/repeat;
        x_error_self_est(4,k+1)=x_error_self_est(4,k+1) + Variance(4,4)/repeat;
    end
    temp = toc;
    Runtime = Runtime + temp/repeat;
    res_x_err(:,:,iii) = states_est - states_true;
end
x_error_self_est = sqrt(x_error_self_est);
x_RMSE = zeros(n,length(t)+1); % root mean square error
for k = 1:1:length(t)+1
    x_RMSE(1,k) = sqrt(mean(res_x_err(1,k,:).^2,3));
    x_RMSE(2,k) = sqrt(mean(res_x_err(2,k,:).^2,3));
    x_RMSE(3,k) = sqrt(mean(res_x_err(3,k,:).^2,3));
    x_RMSE(4,k) = sqrt(mean(res_x_err(4,k,:).^2,3));
end
end
