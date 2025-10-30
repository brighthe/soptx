function [sigmapn, derivative0] = pnorm(p, von_mises, nc, nelx, nely, sigmay)
    % 应力排序和分组
    [von_mises_desc, sort_index] = sort(von_mises, 'descend');
    cluster = zeros(nc, nelx*nely/nc);
    cluster = reshape(von_mises_desc, [(nelx*nely/nc), nc])';
    [row, col] = size(cluster);
    % p-norm 聚合计算
    cluster_p = (cluster / sigmay).^p;
    cluster_sum = sum(cluster_p, 2);
    cluster_mean = cluster_sum./col;
    % 导数计算
    sigmapn = (cluster_mean.^(1/p)) - 1;
    sigmapn1 = (1/p) * (cluster_mean.^((1/p)-1)) .* (1/(col * sigmay^p));
    derivative0 = zeros(nc, nelx*nely);
    
    for i = 1:row
        for j = ((i-1)*col)+1:i*col
               derivative0(i, sort_index(j)) = sigmapn1(i);
               
        end       
    end

end