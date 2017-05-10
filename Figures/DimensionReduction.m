figure('Color', 'white');
set(gcf,'units','points','position',[0,0,550,350])
set(gca,'Fontsize',18);
%clf; 
L1data = load('data/L1coefficient.txt');
OL1data = load('data/OL1coefficient.txt');
OL2data = load('data/OL2coefficient.txt');
%ol2 = [0.0001 0.0004 0.0019 0.0095 0.0201 0.0278 0.0298 0.0344 0.0459 0.3240 0.4262 0.5719 0.6528 0.7590 1.000];
L1data =  sort(abs(L1data),'descend');
OL1data = sort(abs(OL1data),'descend');
OL2data = sort(abs(OL2data),'descend');
len = length(OL2data);
newp = zeros(1,len);
i = 1;
for i = 1:len  % q=4  
  if OL2data(i) >= (i/len)*0.4
  res = sprintf('%f   %f      %f \n',i,OL2data(i),(i/len)*0.4);
  newp(i) = OL2data(i);
  else
      newp(i) = 0;
  end
  %disp(res);
  
end


plot(L1data,'r-.','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL1data,'b-','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(OL2data,'m--','MarkerSize', 10, 'LineWidth', 3);
hold on
plot(51,0.01,'g*','MarkerSize', 10, 'LineWidth', 3)
hold off;
%xlim([0,50]);
%set(gca,'YScale','log');
%set(gcf,'units','points','position',[0,0,750,400])
set(gca,'XTick',(0:200:2000))
xlabel('Number of relevant variables','FontSize',18);
h = ylabel('\textbf{Estimated Coefficients( $$\hat{x}$$ )}','FontSize',18);
set(h,'unit','character');
set(h,'interpreter','latex');
set(gca,'FontSize',18,'FontName','times');
legend1 = legend('Lasso','OL1','OL2');
set(legend1,'FontSize',18,'FontName','times');
title('q = 0.4','FontSize',18);
set(gca,'Fontsize',18);
%print('E:\Deep Learning\Research\Paper Writting\graphs\DimensionReduction.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\DimensionReduction.eps','-depsc2','-r900');

