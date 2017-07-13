%Generate problem data
data = xlsread('data/convergencedata_OL1.xls');
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
set(gcf,'units','points','position',[0,0,600,610]);
set(gca, 'FontSize', 18);
subplot(2,1,1);
semilogy(1:K, max(1e-8, r_norm),'k', ...
    1:K, eps_pri,'k--',  'LineWidth', 3,'MarkerSize', 10);
ylabel('Relative primal-dual gap(\delta(b))','Fontsize',18);
plotHandle = findobj(gca,'Type','line');
set(plotHandle(1),'Color','r');
set(plotHandle(2),'Color','m');
p_legend = legend('Rel.primal-dual gap($\delta(b)$)','TolRelGap($\epsilon^{gap}$)');
set(p_legend,'Interpreter','latex','FontSize',18);
title('Relative primal-dual gap($\delta(b)$) $\leq \epsilon^{gap}$','FontSize',18,'Interpreter','latex');
set(gca, 'FontSize', 18);
subplot(2,1,2);
semilogy(1:K, max(1e-8,s_norm), 'k', ...
    1:K, eps_dual, 'k--', 'LineWidth', 3,'MarkerSize', 10);
ylabel('Infeasibility($\hat{w}$)','Fontsize',18,'Interpreter','latex'); 
xlabel('iter (k)','Fontsize',18);
plotHandle = findobj(gca,'Type','line');
set(plotHandle(1),'Color','r');
set(plotHandle(2),'Color','b');
d_legend = legend('Infeasibility($\hat{w}$)','TolInfeas($\epsilon^{infeas}$)');
set(d_legend,'Interpreter','latex','FontSize',18);
title('Infeasibility($\hat{w}$) $\leq \epsilon^{infeas}$','FontSize',18,'Interpreter','latex');
set(gca, 'FontSize', 18);
print('E:\Deep Learning\Research\Paper Writting\graphs\ConvergenceOL1.eps','-depsc2','-r900');
print('E:\Deep Learning\Research\Paper Writting\graphs\ConvergenceOL1.png','-dpng','-r900');
