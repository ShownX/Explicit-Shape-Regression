function [currentshapes, groundtruthes] = getCSandGTfromData(Data, t, params)
    DataLen = size(Data,1);
    currentshapes = zeros([size(params.meanshape), DataLen]);
    groundtruthes = zeros([size(params.meanshape), DataLen]);
    for i = 1: DataLen
        currentshapes(:, :, i) = Data{i}.intermediate_shapes{t};
        groundtruthes(:, :, i) = Data{i}.shape_gt;
    end
end