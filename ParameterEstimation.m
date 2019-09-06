%Generate problem data

% Colon Dataset For testing
load('data/colon.mat','X','Y');
%x0     = load('data/testData_x0.txt');
A      = X;
b      = Y;
[n,p] = size(A);
q = 0.4;
i = 1:1:p;
lambda = norminv(1-q*i/p/2);
lambda = lambda';

%Solve problem

[x history] = ridge(A, b, lambda, 1.0, 1.0);

%Relevant Variable Selection Methods

%x = sort(abs(x),'descend');
len = length(x);
newx = zeros(1,len);
i = 1;
for i = 1:len  % q=4  
  if x(i) >= (i/len)*0.4
  res = sprintf('%f   %f      %f \n',i,x(i),(i/len)*0.4);
  newx(i) = x(i);
  else
      newx(i) = 0;
  end
  %disp(res);
  
end

%Means Square Error
bnew = sort(A,'descend')*newx'; % new label
mse = mean((bnew-sort(b,'descend')).^2) % Mean square error

