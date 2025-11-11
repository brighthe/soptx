function [cluster_idx, cluster_vm, Ni] = stress_clustering( ...
                                            stress_vm, nc, nele, cluster_idx_prev)

    % 计算每个聚类的单元数
    base_size = floor(nele / nc);
    remainder = mod(nele, nc);
    Ni = repmat(base_size, nc, 1);
    Ni(1:remainder) = Ni(1:remainder) + 1;
    
    % 判断是重新聚类还是更新现有聚类
    if nargin < 4 || isempty(cluster_idx_prev)
        % 重新聚类：按应力水平排序
        [stress_vm_sorted, sort_idx] = sort(stress_vm, 'descend');
        
        cluster_idx = cell(nc, 1);
        cluster_vm = cell(nc, 1);
        start_pos = 1;
        
        for i = 1:nc
            end_pos = start_pos + Ni(i) - 1;
            cluster_idx{i} = sort_idx(start_pos:end_pos);
            cluster_vm{i} = stress_vm_sorted(start_pos:end_pos);
            start_pos = end_pos + 1;
        end
    else
        % 更新现有聚类：保持聚类结构，只更新应力值
        cluster_idx = cluster_idx_prev;
        cluster_vm = cell(nc, 1);
        
        for i = 1:nc
            cluster_vm{i} = stress_vm(cluster_idx{i});
        end
    end
end