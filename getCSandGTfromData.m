function [currentshapes, groundtruthes] = getCSandGTfromData(Data, t, params)
    
    currentshapes = zeros([size(params.meanshape), params.N_img]);
    groundtruthes = zeros([size(params.meanshape), params.N_img]);
    for i = 1: DataLen
        currentshapes(:, :, i) = Data{i}.intermediate_shapes{t};
        groundtruthes(:, :, i) = Data{i}.shape_gt;
    end
end