
OL  = 1:1:45;
OL1 = 1:1:100;
OL2 = 1:1:8;%[1 2 3]

data=[1956; 10000; 6];
xLabels={'L1'; 'OL1'; 'OL2' };
figure('Color', 'white');
for i=1:length(data)
    if i==1
        colorcode = 'r';
    elseif i==2
        colorcode = 'b';
    elseif i==3
        colorcode = 'm';
    else
        colorcode = 'k';
    end
    bar(i, data(i), colorcode,'BarWidth', 0.3);
    hold on;
end
set(gca,'FontSize',18,'FontName','Times New Roman');
set(gca,'XTick',1:length(data),'XTickLabel',xLabels,'FontSize',18)
ylabel('Number of Iterations','FontSize',18);
set(gca,'YScale','log');
set(gca,'Fontsize',18);
% Save the file as PNG
%print('E:\Deep Learning\Research\Paper Writting\graphs\iteration.png','-dpng','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\iteration.eps','-depsc2','-r900');





