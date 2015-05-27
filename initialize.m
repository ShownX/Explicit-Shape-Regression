function Data = initialize(data, initSet, params)
% initialize the data
% data: image
% initSet: the inital set of initialization
% D: the number of initialization
    Index = 0;
    N_data = size(data, 1);
    
    Data = cell(N_data*params.N_aug, 1);
    
    for i = 1: N_data
        % random select initial shapes without replacement
        rand_index_ = randperm(params.N_img, params.N_aug);
        while ismember(i, rand_index_)
            rand_index_ = randperm(params.N_img, params.N_aug);
        end
        % expand the data
        for j = 1: params.N_aug
            r_index = rand_index_(j);
            
            Index = Index + 1;
            % copy the original stuff
            Data{Index}.img_gray = data{i}.img_gray;
            Data{Index}.width_orig  = data{i}.width_orig;
            Data{Index}.height_orig = data{i}.height_orig;    
            Data{Index}.width       = data{i}.width;
            Data{Index}.height      = data{i}.height;
            Data{Index}.shape_gt    = data{i}.shape_gt;
            Data{Index}.bbox_gt     = data{i}.bbox_gt;
            % add the new element
            Data{Index}.intermediate_shapes = cell(1, params.T+1);
            Data{Index}.intermediate_bboxes = cell(1, params.T+1);
            % scale and translate the sampled shape to ground-truth
            % face rectangle region
            select_shape = resetshape(data{i}.bbox_gt, initSet{r_index}.shape_gt(params.ind_usedpts, :));
            
            Data{Index}.intermediate_shapes{1} = select_shape;
            Data{Index}.intermediate_bboxes{1} = getbbox(select_shape);

            meanshape_resize = resetshape(Data{Index}.intermediate_bboxes{1}, params.meanshape);

            Data{Index}.tf2meanshape = fitgeotrans(bsxfun(@minus, ...
                Data{Index}.intermediate_shapes{1}, mean(Data{Index}.intermediate_shapes{1})), ...
                bsxfun(@minus, meanshape_resize, mean(meanshape_resize)),...
                'nonreflectivesimilarity');
            Data{Index}.meanshape2tf = fitgeotrans(bsxfun(@minus, meanshape_resize, mean(meanshape_resize)), ...
                bsxfun(@minus, Data{Index}.intermediate_shapes{1}, mean(Data{Index}.intermediate_shapes{1})),...
                'nonreflectivesimilarity');                

            shape_residual = bsxfun(@rdivide, Data{Index}.shape_gt - Data{Index}.intermediate_shapes{1},...
                Data{Index}.intermediate_bboxes{1}(3: 4));  

            [u, v] = transformPointsForward(Data{Index}.tf2meanshape, shape_residual(:, 1), shape_residual(:, 2));
            Data{Index}.shapes_residual = [u, v];
        end
    end
end