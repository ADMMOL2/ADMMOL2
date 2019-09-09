
%EL_struct = open('data/coefficientenetpath_lue.mat');
%EL_transposed = structfun(@transpose,EL_struct,'UniformOutput',false);
%data = struct2table(EL_transposed);
%writetable(data,'data/coefficientenetpath_lue.csv');

%[data,delimiterOut]=importdata('data/coefficientenetpath_lue.mat');
data = xlsread('data/coefficientenetpath_lue.csv');
X = data(1,:); % training label
max(X)
index = data(2,:); % number of non-zero element
%df = index(index>0);
Y = data(3:end,:); % training
%plotxx(X,Y,index,Y);
%glmnetPrint(fit.beta)
%set ( gca, 'xdir', 'reverse' );
figure
set(gca,'FontSize',18);
set(gcf,'units','points','position',[0,0,700,600]);
ax1 = gca;
hold on
plot(X,Y','MarkerSize', 10, 'LineWidth', 2)
xlabel('Iterations'); 
ylabel('Standardized Coefficients');
ax2 = axes('Position',get(ax1,'Position'),...
       'XAxisLocation','top',...
       'YAxisLocation','right',...
       'Color','none',...
       'XColor','k','YColor','k');
set(ax2,'XLim',[100 108]);
%set(ax2,'XTick',min(df):12:max(df));
%set(ax1,'XTick',0:200:2200);
set(ax2,'Color','none','YTick',[])
%linkaxes([ax1 ax2],'off');
hold on
SP=106.6; %your point goes here y-axis step = 2718.2
line([SP SP],get(ax2,'YLim'),'Color','b','LineStyle',':','MarkerSize', 10, 'LineWidth', 2)
set(gca,'FontSize',18);
xlabel('Number of selected genes');
hold off;



