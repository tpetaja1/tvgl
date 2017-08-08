function E = soft_thres_od(A, lambda, rho)
    % Computes soft-threshold for all elements of matrix except diagonal
    % elements
    dimension = size(A,1);
    E = ones(dimension,dimension);
    for i = 1:dimension
        for j = 1:dimension
            if i ~= j
                if abs(A(i,j)) <= lambda/rho
                    E(i,j) = 0;
                else
                    E(i,j) = sign(A(i,j))*(abs(A(i,j)) - lambda/rho);
                end
            end
        end
    end

end