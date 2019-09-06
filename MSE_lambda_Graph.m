figure('Color', 'white');
set(gcf,'units','points','position',[0,0,650,550]);
%set(gca,'Fontsize',18,'FontName','Times New Roman');
clf; 
L1data = load('data/L1-Lasso.txt');
OL1data = load('data/OL1.txt');
L2data = load('data/L2.txt');
OL2data = load('data/OL2.txt');
lambda = load('data/lambda.txt');
%lambda = min(data)+rand(1,length(OL2data))*(max(data)-min(data)); % x=xmin+rand(1,n)*(xmax-xmin)
%size(L1data)
%size(OL1data)
%size(OL2data)
%size(lambda)
plot(L1data,'r-.','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL1data,'b-','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(L2data,'c:.','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL2data,'m--','MarkerSize', 10, 'LineWidth', 3);
hold off;
xlim([1,max(lambda)]);
set(gca,'YScale','log');
xlabel('\lambda','FontSize',18); 
ylabel('Estimated Square Error','FontSize',18);
legend1 = legend('Lasso','OL1','L2','OL2','Location','northwest','Orientation','horizontal');
set(legend1,'FontSize',18,'FontName','Times New Roman');
title('q=0.4','FontSize',18);
set(gca,'Fontsize',18);
%print('E:\Deep Learning\Research\Paper Writting\graphs\coefficient_mse.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\coefficient_mse.eps','-depsc2','-r900');

