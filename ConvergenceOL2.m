%Generate problem data
data = xlsread('data/convergencedata_OL2.xls');
iter        = data(:,1);
r_norm      = data(:,2);
eps_pri     = data(:,3);
s_norm      = data(:,4);
eps_dual    = data(:,5);
objval      = data(:,6);

%Solve problem



%Reporting

K = length(objval);

h = figure('Color', 'white');
set(gca, 'FontSize', 18); %<- Set properties
plot(1:K, objval,'k', 'MarkerSize', 10, 'LineWidth', 3);
ylabel('f(x^k) + g(z^k)','Fontsize',18); 
xlabel('iter (k)','Fontsize',18);
plotHandle = findobj(gca,'Type','line');
set(plotHandle(1),'Color','b');
title('Objective Function','Fontsize',18);
%print('E:\Deep Learning\Research\Paper Writting\graphs\objective.eps','-depsc2','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\objective.png','-dpng','-r900');

g = figure('Color', 'white');
set(gcf,'units','points','position',[0,0,500,610]);
set(gca, 'FontSize', 18);
subplot(2,1,1);
semilogy(1:K, max(1e-8, r_norm),'k', ...
    1:K, eps_pri,'k--',  'LineWidth', 3,'MarkerSize', 10);
ylabel('||r||_2','Fontsize',18);
plotHandle = findobj(gca,'Type','line');
set(plotHandle(1),'Color','r');
set(plotHandle(2),'Color','m');
p_legend = legend('Primal Residual ( ||r^k||_2 )','Primal Feasibility tol ( \epsilon^{pri} )');
set(p_legend,'FontSize',18,'FontName','Times New Roman');
title('||r^k||_2 \leq \epsilon^{pri}','FontSize',18);
set(gca, 'FontSize', 18);
subplot(2,1,2);
semilogy(1:K, max(1e-8,s_norm), 'k', ...
    1:K, eps_dual, 'k--', 'LineWidth', 3,'MarkerSize', 10);
ylabel('||s||_2','Fontsize',18); 
xlabel('iter (k)','Fontsize',18);
plotHandle = findobj(gca,'Type','line');
set(plotHandle(1),'Color','r');
set(plotHandle(2),'Color','b');
d_legend = legend('Dual Residual ( ||s^k||_2 )','Dual Feasibility tol ( \epsilon^{dual} )');
set(d_legend,'FontSize',18,'FontName','Times New Roman');
title('||s^k||_2 \leq \epsilon^{dual}','FontSize',18);
set(gca, 'FontSize', 18);
%print('E:\Deep Learning\Research\Paper Writting\graphs\primaldual.eps','-depsc2','-r900');
%print('E:\Deep Learning\Research\Paper Writting\graphs\primaldual.png','-dpng','-r900');
