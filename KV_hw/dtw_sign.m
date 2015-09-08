function [res, path, D] = dtw_sign(X, algoptions)
% X - [M x N] matrix of dissimlarities  
% algoptions.penalty - penalty for stretching
% res - dissimilarity of two signals 
% path - table of pairwise correspondences of two signals 
% (optimal alignment
% D - alignment matrix

pnt = algoptions.penalty;
[M,N] = size(X);

D = zeros(M+1, N+1);
D(1,:) = NaN;
D(:,1) = NaN;
D(1,1) = 0;
D(2:(M+1), 2:(N+1)) = X;

phi = zeros(M,N); %???????????????????
for i = 1:M 
  for j = 1:N
    [dmin, tb] = min([D(i,j), D(i,j+1)+pnt, D(i+1,j)+pnt]);
    D(i+1,j+1) = D(i+1,j+1)+dmin;
    phi(i,j) = tb;
  end
end

% Traceback from top left
i = M; 
j = N;
path = [i;j];
while i > 1 & j > 1 
  
  tb = phi(i,j);
  if (tb == 1)
    i = i-1;
    j = j-1;
  elseif (tb == 2)
    i = i-1;
  elseif (tb == 3)
    j = j-1;
  else    
    error;
  end 
  
  path = [[i;j], path];
  
end
if (i>1 || j>1)
    path = [[1;1],path];
end
res = D(M+1,N+1);
D = D(2:(M+1), 2:(N+1));
