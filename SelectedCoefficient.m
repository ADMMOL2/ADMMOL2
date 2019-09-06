f = figure('Color', 'white');
%set(gca,'Fontsize',18);
%clf; 
%L1data = load('data/L1-lasso_q.txt');
%OL1data = load('data/OL1-q.txt');
%L2data = load('data/L2_q.txt');
%OL2data = load('data/OL2-q.txt');
data = load('data/Coefficientbeta.csv');
lambdadata = data(:,1);
x = lambdadata;
y=data(:,2);
plot(x,y,'r-.','MarkerSize', 10, 'LineWidth', 3);

%xlim([1,5]);
%set(gca,'XTick',(5:1:1));
%set(gca,'YScale','log');
xlabel('$$\lambda$$','Fontsize',18,'Interpreter','latex'); 
ylabel('Selected Coefficients','Fontsize',18);
% Create axes

legend1 = legend('$$O\ell_{1,2}$$');
set(legend1,'FontSize',18,'FontName','Times New Roman','Interpreter','latex');
set(gca,'FontSize',18);
set ( gca, 'xdir', 'reverse' )
% Enlarge figure to full screen.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Maximize figure.
%title('q = 0.4');
%print('E:\Deep Learning\Research\Paper Writting\graphs\AvgMse_q.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\paper\ICBK 2019\Fig7.eps','-depsc2','-r900');

