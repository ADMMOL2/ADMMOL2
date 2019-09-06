f = figure('Color', 'white');
%set(gca,'Fontsize',18);
%clf; 
%L1data = load('data/L1-lasso_q.txt');
%OL1data = load('data/OL1-q.txt');
%L2data = load('data/L2_q.txt');
%OL2data = load('data/OL2-q.txt');
L1data = load('data/peLasso_q.txt');
OL1data = load('data/peOL1_q.txt');
OL2data = load('data/peOL2_q.txt');
OL12data = load('data/peOL12_q.txt');


x = linspace(0,1,10);
plot(x,L1data,'r-.','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(x,OL1data,'b-','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(x,OL2data,'g--','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(x,OL12data,'m--','MarkerSize', 10, 'LineWidth', 3);
hold off;
xlim([0.1,1.0]);
set(gca,'XTick',(0:0.1:10));
set(gca,'YScale','log');
xlabel('q','Fontsize',18); 
ylabel('Estimated MSE','Fontsize',18);
% Create axes

legend1 = legend('Lasso','$$O\ell_{1}$$','$$O\ell_{2}$$','$$O\ell_{1,2}$$','Orientation','horizontal');
set(legend1,'FontSize',18,'FontName','Times New Roman','Interpreter','latex');
set(gca,'FontSize',18);

%title('q = 0.4');
%print('E:\Deep Learning\Research\Paper Writting\graphs\AvgMse_q.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\paper\ICBK 2019\Fig7.eps','-depsc2','-r900');
