function[f0val,df0dx,fval] = Load(F,U,x,M,KE,m,n,nelx,nely,penal,unit_size_x,unit_size_y,rmin, volfrac)
    f0val = 0.;
    fval_1=0;
    for ely = 1:nely
        for elx = 1:nelx
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2], 1);
            % 目标函数
            f0val = f0val + x(ely, elx)^penal*Ue'*KE*Ue;
            % 目标函数灵敏度
            dc(ely,elx) = -penal*x(ely, elx)^(penal-1)*Ue'*KE*Ue;
            fval_1 = fval_1 + unit_size_x*unit_size_y*x(ely,elx);
        end
    end

    [dc] = check(nelx, nely, rmin, x, dc);
      
    count_2 = 1;
    for g = 1:nely
        for h = 1:nelx
            df0dx(count_2, 1) = dc(g, h);
            count_2 = count_2 + 1;
        end
    end
    
    fval(1, 1) = (fval_1 / M) - volfrac;

end