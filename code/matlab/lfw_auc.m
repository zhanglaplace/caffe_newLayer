function auc  = lfw_auc(fn)
% LFW_AUC computes area under ROC curve
% auc = LFW_AUC(fn)
%
% fn is the filename of the file containing the ROC curve
% this file should have two columns:
%   first column is true positive rate
%   second column is false positive rate
  fn = 'deepid.txt';
  fid = fopen(fn);
  % fscanf is column major, so transpose
  pts = fscanf(fid, '%f', [2,Inf])';
  fclose(fid);
  
  % ensure false positive rate is in increasing order
  [~,idx] = sort(pts(:,2));
  pts = pts(idx,:);
  
  fpr = pts(:,2);
  tpr = pts(:,1);
  n   = size(pts,1);
  
  % apply trapezoid rule
  auc = .5*sum( (fpr(2:n)-fpr(1:(n-1))) .* ...
                (tpr(2:n)+tpr(1:(n-1))) );

end
