function M = init_cells(dimension, count, type)
    
    M = cell(count,1);
    if strcmp(type,'random')
        for i = 1:count
            M{i} = rand(dimension,dimension);
        end
    elseif strcmp(type,'ones')
        for i = 1:count
            M{i} = ones(dimension,dimension);
        end
    elseif strcmp(type,'zeros')
        for i = 1:count
            M{i} = zeros(dimension,dimension);
        end
    end
end