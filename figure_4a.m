q = 0.4;
p = 5000;
i = 1:1:500;
l1 = norminv(1-q*i/p/2);
l2 = l1*sqrt(1+cumsum(l1.^2)/(p-i-1));
l3 = l1*sqrt(1+cumsum(l1.^2)/(2*p-i-1));
figure;
set(gca, 'FontSize', 18); %<- Set properties
plot(i,l1,'r','LineStyle','-','LineWidth',3,'MarkerSize',10);
hold on
plot(i,l2,'g','LineStyle','--','LineWidth',3,'MarkerSize',10);
hold on
plot(i,l3,'k','LineStyle',':','LineWidth',3,'MarkerSize',10);
ylim([2,5.5]);
set(gca,'xTick',0:100:500);
%title(sprintf('q=%f',q),'FontSize',18);
xlabel('\it{i}','FontSize',18)
ylabel('\lambda','FontSize',18)
legend1=legend('\lambda_{BH} given by eq(13)','\lambda given by eq(14), n=p','\lambda given by eq(14), n=2p');
set(legend1,'FontSize',18,'FontName','Times New Roman');
% Save the file as PNG
%print('E:\Deep Learning\Research\Paper Writting\graphs\lambda04.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\lambda04.eps','-depsc2','-r900');





