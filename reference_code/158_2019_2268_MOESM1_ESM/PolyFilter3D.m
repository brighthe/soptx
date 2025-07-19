%------------------------------ Poly3DScript -----------------------------%
% Ref: H Chi,A Pereira, IFM Menezes, GH Paulino , "Virtual Element Method %
% (VEM)-based topology optimization: an integrated framework",            %
% Struct Multidisc Optim, DOI 10.1007/s00158-019-02268-w                  %
%-------------------------------------------------------------------------%
function [P] = PolyFilter3D(PS1,PS2,R,q)
[d] = DistPntSets(PS1,PS2,R);  %Obtain distance values & indices
P = sparse(d(:,1),d(:,2),(1-d(:,3)/R).^q,size(PS1,1),size(PS2,1)); % Assemble the filtering matrix
AA=sum(P,2);
id=find(AA~=0);
P(id,:)=spdiags(1./AA(id),0,length(id),length(id))*P(id,:);
%---------------------------------- COMPUTE DISTANCE BETWEEN TWO POINT SETS
function [d] = DistPntSets(PS1,PS2,R)
M = size(PS2,1);
[idx,D] = rangesearch(PS1,PS2,R);
ivec = cell2mat(idx');
jvec = cell2mat(cellfun(@(x,k)k*ones(1,length(x)),idx',num2cell(1:M),'UniformOutput',false));
svec = cell2mat(D');
d = [ivec(:),jvec(:),svec(:)];
%---------------------------------- COMPUTE DISTANCE BETWEEN TWO POINT SETS
% function [d] = DistPntSets(PS1,PS2,R)
% d = cell(size(PS1,1),1);
% for el = 1:size(PS1,1)       %Compute the distance information
%     dist = sqrt((PS1(el,1)-PS2(:,1)).^2 +...
%                 (PS1(el,2)-PS2(:,2)).^2 +...
%                 (PS1(el,3)-PS2(:,3)).^2);
%     [I,J] = find(dist<=R);   %Find the indices for distances less that R
%     d{el} = [I,J+(el-1),dist(I)];
% end
% d = cell2mat(d);             %Matrix of indices and distance value
