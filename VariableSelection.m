figure('Color', 'white');
set(gcf,'units','points','position',[0,0,550,350]);
%set(gca,'Fontsize',18);
clf; 
L1data = load('data/L1coefficient.txt');
OL1data = load('data/OL1coefficient.txt');
OL12data = load('data/L12coefficient.txt');
OL2data = load('data/OL2coefficient.txt');
L1data =  L1data(L1data~=0);
OL1data = OL1data(OL1data~=0);
OL12data = OL12data(OL12data~=0);
OL2data = OL2data(OL2data~=0);

%size(OL2data)
%lambda = min(data)+rand(1,length(OL2data))*(max(data)-min(data)); % x=xmin+rand(1,n)*(xmax-xmin)
%size(L1data)
%size(OL1data)
%size(OL2data)
%size(lambda)
plot(L1data,'r-.','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL1data,'b-','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL12data,'c-','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL2data,'m--','MarkerSize', 10, 'LineWidth', 3);
hold off;
%xlim([0,50]);
%set(gca,'YScale','log');
set(gca,'XTick',(0:200:2000));
xlabel('Number of relevant variables','FontSize',18); 
h = ylabel('\textbf{Estimated Coefficients( $$\hat{x}$$ )}','FontSize',18);
set(h,'unit','character');
set(h,'interpreter','latex');
set(gca,'FontSize',18,'FontName','times');
legend1 = legend('Lasso','$$O\ell_{1}$$','$$O\ell_{1,2}$$','$$O\ell_{2}$$');
set(legend1,'FontSize',18,'FontName','times','Interpreter','latex');
title('q = 0.4','FontSize',18);
set(gca,'Fontsize',18);
set(gca,'color','none');
%print('E:\Deep Learning\Research\Paper Writting\graphs\variableselection.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\variableselection.eps','-depsc2','-r900');

