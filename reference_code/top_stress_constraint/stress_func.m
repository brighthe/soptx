function [von_mises,derivative] = stress_func(C,B,U,nelx,nely,x,p)
    stress_counter = 1;
    
    for ely = 1:nely
        for elx = 1:nelx
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
            stress=C*B*Ue;
            stress_val=(x(ely,elx)^0.5)*stress;
            von_mises(stress_counter,1)=sqrt((stress_val(1,1)^2)+(stress_val(2,1)^2)+(3*stress_val(3,1)^2)-stress_val(1,1)*stress_val(2,1));
            derivative_1=von_mises(stress_counter,1)^(p-1);
            derivative_2=[2*stress_val(1,1)-stress_val(2,1);2*stress_val(2,1)-stress_val(1,1);6*stress_val(3,1)]./von_mises(stress_counter,1);
            derivative_3=0.5*x(ely,elx)^(-0.5)*stress;
            derivative(stress_counter,1)=derivative_1*derivative_2'*derivative_3;
            stress_counter=stress_counter+1;              
        end
    end
end