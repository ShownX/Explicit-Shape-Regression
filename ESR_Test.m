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
    
    Data = loadsamples('/Volumes/LG_SDJet/Datasets/lfpw/annotations/trainset', 'png');
    params.N_img = size(Data, 1); 
    load('Data/InitialShape_68');
    dist_pupils_ms = getDistPupils(S0);
    params.meanshape = S0(params.ind_usedpts, :);
    params.N_fp = size(params.meanshape, 1);

    load('Data/Model.mat', 'Model'); 
    %%
    for i = 1: 1
        Prediction = ShapeRegression(Data{i}, initSet, Model, params);
        figure
        imshow(Data{i}.img_gray)
        hold on
        plot(Prediction(:, 1), Prediction(:, 2), 'g+');
        hold off
    end
    
end

function predict = ShapeRegression(data, initSet, Model, params)
    % Multiple initializations
    
    %predict = zeros(params.N_fp, 2);
    ctshapes = initialTest(data, initSet, params);

    for t = 1: params.T
        for i = 1: params.N_init
            prediction_delta = fernCascadeTest(data, ctshapes(:, :, i), Model{t}.fernCascade, params);
            ctshapes(:, :, i) = ctshapes(:, :, i) + prediction_delta;
        end
    end
    predict = mean(ctshapes, 3);
end

function prediction_delta = fernCascadeTest(image, current_shape, fernCascade, params)
    image.intermediate_bbx = getbbox(current_shape);
    meanshape = resetshape(image.intermediate_bbx, params.meanshape);
    image.tf2meanshape = fitgeotrans(bsxfun(@minus, current_shape, mean(current_shape)), ...
                bsxfun(@minus, meanshape, mean(meanshape)),...
                'nonreflectivesimilarity');
    image.meanshape2tf = fitgeotrans( bsxfun(@minus, meanshape, mean(meanshape)),...
                bsxfun(@minus, current_shape, mean(current_shape)), ...
                'nonreflectivesimilarity');
    
    %extract shape indexed pixels
    candidate_pixel_location = fernCascade.candidate_pixel_location;
    nearest_landmark_index = fernCascade.nearest_landmark_index;
    intensities = zeros(1, params.P);
    for j = 1: params.P
        x = candidate_pixel_location(j, 1)*image.intermediate_bbx(3);
        y = candidate_pixel_location(j, 2)* image.intermediate_bbx(4);
        [project_x, project_y] = transformPointsForward(image.meanshape2tf, x, y);
        index = nearest_landmark_index(j);

        real_x = round(project_x + current_shape(index, 1));
        real_y = round(project_y + current_shape(index, 2));
        real_x = max(1, min(real_x, size(image.img_gray, 2)-1));
        real_y = max(1, min(real_y, size(image.img_gray, 1)-1));
        intensities(j)= image.img_gray(real_y, real_x);
    end
    
    delta_shape = zeros(size(params.meanshape));
    for i = 1: params.K
        fern = fernCascade.ferns{i}.fern;
        delta_shape = delta_shape + fernTest(intensities, fern, params);
    end
    
    %convert to the currentshape model
    [u, v] = transformPointsForward(image.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
    prediction_delta = [u, v];
    prediction_delta = bsxfun(@times, prediction_delta, [image.intermediate_bbx(3),image.intermediate_bbx(4)]);
end

function fern_pred = fernTest(intensities, fern, params)
    index = 0;
    for i = 1: params.F
        intensity_1 = intensities(fern.selected_pixel_index(i, 1));
        intensity_2 = intensities(fern.selected_pixel_index(i, 2));
        
        if (intensity_1 - intensity_2) >= fern.threshold(i)
            index = index + 2^(i-1);
        end
    end
    index = index + 1;
    fern_pred = fern.bin_output{index};
end