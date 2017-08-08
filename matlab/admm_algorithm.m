function Theta = admm_algorithm(Theta,Z0,Z1,Z2,U0,U1,U2,...
    rho,lambda,beta,nju,empirical_covariance_matrices)
    
    max_iteration = 20000;
    index = 0;
    stopping_criteria = 0;
    
    total_stamps = size(Theta,1);
    dimension = size(Theta{1},1);
    
    while index < max_iteration && stopping_criteria == 0
        
        %% Theta-Update
        for i = 1:total_stamps
            A = (Z0{i}+Z1{i}+Z2{i}-U0{i}-U1{i}-U2{i})/3;
            M = nju*(A+A')/2 - empirical_covariance_matrices{i};
            [Q,D] = eig(M)
            
            Theta{i} = nju/2*Q*(D+sqrt(D^2 + 4/nju*diag(ones(dimension,1))))*Q';
        end

        %% Z0-Update
        for i = 1:total_stamps
            Z0{i} = soft_thres_od(Theta{i} + U0{i},lambda,rho); % soft threshold odd
        end

        %% Z1,Z2-Update
        for i = 2:total_stamps
            A = Theta{i} - Theta{i-1} + U2{i} - U1{i-1};
            E = group_lasso_penalty(A,2*beta/rho);
            Z_pre = 0.5*[Theta{i-1}+Theta{i}+U1{i}+U2{i};Theta{i-1}+Theta{i}+U1{i}+U2{i}] + ...
                0.5*[-E;E];
            Z1{i-1} = Z_pre(1:dimension,:);
            Z2{i} = Z_pre(dimension+1:end,:);
        end

        %% U0-Update
        for i = 1:total_stamps
            U0{i} = U0{i} + Theta{i} - Z0{i};
        end

        %% U1,U2-Update
        for i = 2:total_stamps
            U1{i-1} = U1{i-1} + Theta{i-1} - Z1{i-1};
            U2{i} = U2{i} + Theta{i} - Z2{i};
        end
        
        %% Stopping condition
        if index > 0
            fro_norm = 0;
            for i = 1:total_stamps
                dif = Theta{i} - Theta_pre{i};
                fro_norm = fro_norm + norm(dif,'fro');
            end
            if fro_norm < 1e-5
                stopping_criteria = 1;
            end
        end
        Theta_pre = Theta;
        index = index + 1;
        
    end
    index

end