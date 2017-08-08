function E = group_lasso_penalty(A, nju)
    
    dimension = size(A,1);
    E = zeros(dimension,dimension);
    for j = 1:dimension
        l2_norm = norm(A(:,j));
        if l2_norm <= nju
            E(:,j) = zeros(dimension,1);
        else
            E(:,j) = (1 - nju/l2_norm)*A(:,j);
        end
    end

end