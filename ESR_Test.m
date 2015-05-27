function ESR_Test()
%% load parameters
    params = Test_params;
    % create paralllel local jobs note
    if isempty(gcp('nocreate'))
        parpool(2);
    end
    %% load data
    if exist('Data/train_init.mat', 'file')
        load('Data/train_init.mat', 'data');
        initSet = data;
        clear data;
    else
        disp('lack of initial set of shapes');
    end
    
    Data = loadsamples('lfpw/testset', 'png');
    
    %% choose corresponding points for training
    parfor i = 1:length(Data)
        Data{i}.shape_gt = Data{i}.shape_gt(params.ind_usedpts, :);
        Data{i}.bbox_gt = getbbox(Data{i}.shape_gt);
    end
    %%
    params.N_img = size(Data, 1); 
    load('Data/InitialShape_68');
    dist_pupils_ms = getDistPupils(S0);
    params.meanshape = S0(params.ind_usedpts, :);
    params.N_fp = size(params.meanshape, 1);

    load('Data/Model.mat', 'Model'); 
    %%
    prediction = zeros([size(params.meanshape), params.N_img]);
    groundtruth = zeros([size(params.meanshape), params.N_img]);
    for i = 1: 1%params.N_img
        Prediction = ShapeRegression(Data(i), initSet, Model, params);
        prediction(:,:, i) = Prediction;
        groundtruth(:,:, i) = Data{i}.shape_gt;
    end
    fprintf('MSRE is %f\n', mean(compute_error(prediction, groundtruth)));
end

function predict = ShapeRegression(data, initSet, Model, params)
    % Multiple initializations
    Data = initialize(data, initSet, params);
    Error = zeros(1, params.T+1);
    
    for t = 1: params.T
        for i = 1: params.N_aug
            prediction_delta = fernCascadeTest(Data{i}, Model{t}.fernCascade, params, t);
            % update the shape, convert to the current shape
            bbx = Data{i}.intermediate_bboxes{t};
            shape_stage = Data{i}.intermediate_shapes{t};
            delta_shape = prediction_delta;
            
            [u, v] = transformPointsForward(Data{i}.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
            delta_shape_interm_coord = [u, v];
            shape_residual = bsxfun(@times, delta_shape_interm_coord, [bbx(3),bbx(4)]);
            shape_newstage = shape_stage + shape_residual;
                        
            % update the shape
            Data{i}.intermediate_bboxes{t+1} = getbbox(shape_newstage);
            Data{i}.intermediate_shapes{t+1} = shape_newstage;
            meanshape_reproject = resetshape(Data{i}.intermediate_bboxes{t+1}, params.meanshape);
            Data{i}.tf2meanshape = fitgeotrans( bsxfun(@minus, shape_newstage, mean(shape_newstage)), ...
                bsxfun(@minus, meanshape_reproject, mean(meanshape_reproject)),...
                'nonreflectivesimilarity');
            Data{i}.meanshape2tf = fitgeotrans( bsxfun(@minus, meanshape_reproject, mean(meanshape_reproject)),...
                bsxfun(@minus, shape_newstage, mean(shape_newstage)), ...
                'nonreflectivesimilarity');
            % shape_residual = bsxfun(@rdivide, Data{i}.shape_gt - shape_newstage, Data{i}.intermediate_bboxes{t+1}(3:4));
            % [u, v] = transformPointsForward(Data{i}.tf2meanshape, shape_residual(:, 1), shape_residual(:, 2));   
            % Data{i}.shapes_residual = [u, v];
        end
        [ct, gt] = getCSandGTfromData(Data, t, params);
        Error(t) = mean(compute_error(gt, ct));
    end
    
    %%
    [ct, gt] = getCSandGTfromData(Data, params.T+1, params);
    Error(params.T+1) = mean(compute_error(gt, ct));
    bar(Error);
    xlabel('iterations');
    ylabel('Root Mean Square Error (RMSE)');
    %%
    predict = mean(ct, 3);
end

function delta_shape = fernCascadeTest(image, fernCascade, params, t)
    %extract shape indexed pixels
    candidate_pixel_location = fernCascade.candidate_pixel_location;
    nearest_landmark_index = fernCascade.nearest_landmark_index;
    intensities = zeros(1, params.P);
    for j = 1: params.P
        x = candidate_pixel_location(j, 1)*image.intermediate_bboxes{t}(3);
        y = candidate_pixel_location(j, 2)* image.intermediate_bboxes{t}(4);
        [project_x, project_y] = transformPointsForward(image.meanshape2tf, x, y);
        index = nearest_landmark_index(j);

        real_x = round(project_x + image.intermediate_shapes{t}(index, 1));
        real_y = round(project_y + image.intermediate_shapes{t}(index, 2));
        real_x = max(1, min(real_x, size(image.img_gray, 2)-1));
        real_y = max(1, min(real_y, size(image.img_gray, 1)-1));
        intensities(j)= image.img_gray(real_y, real_x);
    end
    
    delta_shape = zeros(size(params.meanshape));
    for k = 1: params.K
        fern = fernCascade.ferns{k}.fern;
        delta_shape = delta_shape + fernTest(intensities, fern, params);
    end
    
    %convert to the currentshape model
%     [u, v] = transformPointsForward(image.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
%     prediction_delta = [u, v];
%     prediction_delta = bsxfun(@times, prediction_delta, [image.intermediate_bbx(3),image.intermediate_bbx(4)]);
end

function fern_pred = fernTest(intensities, fern, params)
    index = 0;
    for i = 1: params.F
        m_f = fern.selected_pixel_index(i, 1);
        n_f = fern.selected_pixel_index(i, 2);
        
        intensity_1 = intensities(m_f);
        intensity_2 = intensities(n_f);
        
        if intensity_1 - intensity_2 >= fern.threshold(i)
            index = index + 2^(i-1);
        end
    end
    index = index + 1;
    fern_pred = fern.bin_output{index};
end