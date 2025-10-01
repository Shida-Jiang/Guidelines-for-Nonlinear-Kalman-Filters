sita=0:0.01:pi*2;
beta=2;
alpha=1e-3;
x1=[cos(sita);cos(sita+2*pi/3);cos(sita+4*pi/3)];
y1=[sin(sita);sin(sita+2*pi/3);sin(sita+4*pi/3)];
x1=x1./sqrt(mean((x1(:,1)-mean(x1(:,1))).^2));
y1=y1./sqrt(mean((y1(:,1)-mean(y1(:,1))).^2));
x2=[cos(sita);cos(sita+pi/2);cos(sita+pi);cos(sita+3*pi/2)];
y2=[sin(sita);sin(sita+pi/2);sin(sita+pi);sin(sita+3*pi/2)];
x2=x2./sqrt(mean((x2(:,1)-mean(x2(:,1))).^2));
y2=y2./sqrt(mean((y2(:,1)-mean(y2(:,1))).^2));
z_mean1=(f(x1(1,:),y1(1,:))+f(x1(2,:),y1(2,:))+f(x1(3,:),y1(3,:)))/3;
z_cov1=((f(x1(1,:),y1(1,:))-z_mean1).^2+(f(x1(2,:),y1(2,:))-z_mean1).^2+(f(x1(3,:),y1(3,:))-z_mean1).^2)/3;

z_mean2=(f(x2(1,:),y2(1,:))+f(x2(2,:),y2(2,:))+f(x2(3,:),y2(3,:))+f(x2(4,:),y2(4,:)))/4;
z_cov2=((f(x2(1,:),y2(1,:))-z_mean2).^2+(f(x2(2,:),y2(2,:))-z_mean2).^2+(f(x2(3,:),y2(3,:))-z_mean2).^2+(f(x2(4,:),y2(4,:))-z_mean2).^2)/4;

zxy_cov1=zeros(2,length(sita));
zxy_cov2=zeros(2,length(sita));
for i=1:length(sita)
    zxy_cov1(:,i)=([x1(1,i);y1(1,i)]*(f(x1(1,i),y1(1,i))-z_mean1(i))+[x1(2,i);y1(2,i)]*(f(x1(2,i),y1(2,i))-z_mean1(i))+[x1(3,i);y1(3,i)]*(f(x1(3,i),y1(3,i))-z_mean1(i)))/3;
    zxy_cov2(:,i)=([x2(1,i);y2(1,i)]*(f(x2(1,i),y2(1,i))-z_mean2(i))+[x2(2,i);y2(2,i)]*(f(x2(2,i),y2(2,i))-z_mean2(i))+[x2(3,i);y2(3,i)]*(f(x2(3,i),y2(3,i))-z_mean2(i))+[x2(4,i);y2(4,i)]*(f(x2(4,i),y2(4,i))-z_mean2(i)))/4;
end
z_cov3=z_cov1+(beta+1).*(f(0,0)-z_mean1).^2;
zxy_cov3=[0.1;1];
z_EKF=1.01;
z_EKF2=4.03;
z_SUKF=1.01+2.42;
%% 
f1=figure('Position', [100 100 600 450]); % [left bottom width height]
tiledlayout(2,1,'TileSpacing','Compact','Padding','Compact');
nexttile
hold on
plot(sita,z_EKF*ones(size(sita)),'DisplayName',"EKF",'LineWidth',1)
plot(sita,z_EKF2*ones(size(sita)),'DisplayName',"EKF2 $(\beta = 2)$",'LineWidth',1)
plot(sita,z_cov1,'DisplayName',"SKF $(\beta = 0)$",'LineWidth',1)
plot(sita,z_cov2,'DisplayName',"CKF $(\beta = 0)$",'LineWidth',1)
plot(sita,z_cov3,'DisplayName',"SKF* $(\beta = 2)$",'LineWidth',1)
plot(sita,z_SUKF*ones(size(sita)),'DisplayName',"SSKF $(\beta = 2)$",'LineWidth',1)


ylabel('Estimated variance of z', 'Interpreter','tex', 'FontSize',12);
set(gca, 'FontSize', 12)
set(gca, 'XTick', [0 pi/2 pi 3*pi/2 2*pi]);
xticklabels({'0','\pi/2','\pi','3\pi/2','2\pi'});
xlim([0 2*pi])
lgd=legend('Interpreter','latex');   % enable LaTeX rendering
legend(FontSize=11)
lgd.NumColumns = 2;
grid on
title('$z=f(x_1,x_2)=0.1x_1^2+0.1x_1+x_2^2+x_2+x_1x_2$', 'Interpreter','latex','FontWeight','normal')

nexttile
hold on
plot(sita,zxy_cov1(1,:),'DisplayName',"SKF and SKF* $P_{xz}(1)$",'LineWidth',1)
plot(sita,zxy_cov1(2,:),'DisplayName',"SKF and SKF* $P_{xz}(2)$",'LineWidth',1)
plot(sita,zxy_cov2(1,:),'DisplayName',"Other KFs $P_{xz}(1)$",'LineWidth',1)
plot(sita,zxy_cov2(2,:),'DisplayName',"Other KFs $P_{xz}(2)$",'LineWidth',1)
xlabel("Rotation angle (rad)",FontSize=12)
ylabel("Estimated cross-covariance",FontSize=12)
set(gca, 'FontSize', 12)
set(gca, 'XTick', [0 pi/2 pi 3*pi/2 2*pi]);
xticklabels({'0','\pi/2','\pi','3\pi/2','2\pi'});
xlim([0 2*pi])
lgd=legend('Interpreter','latex');
legend(FontSize=11)
lgd.NumColumns = 2;
grid on

exportgraphics(f1,'Problem-of-3-points.png','Resolution',600)
function z=f(x,y)
a1=0.1;
b1=0.1;
a2=1;
b2=1;
c=1;
z=a1*x.^2+b1.*x+a2.*y.^2+b2.*y+c.*x.*y;
end