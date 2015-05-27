function ESR_Train()
    %% load parameters
    params = Train_params;
    % create paralllel local jobs note
    if isempty(gcp('nocreate'))
        parpool(2);
    end
    %% load data
    if exist('Data/train_init.mat', 'file')
        load('Data/train_init.mat', 'data');
    else
        data = loadsamples('/lfpw/annotations/trainset', 'png');
        %mkdir Data;
        save('Data/train_init.mat', 'data');
    end
    
    load('Data/InitialShape_68');
    dist_pupils_ms = getDistPupils(S0);
    params.meanshape = S0(params.ind_usedpts, :);
    params.N_fp = size(params.meanshape, 1);
    params.N_img = size(data, 1);
    
    initSet = data;
    %% flip data
    if params.flip
        data_flip = fliplrdata(data);
    else
        data_flip = [];
    end
    Data = [data; data_flip];
    %Data = Data(1);
    %% choose corresponding points for training
    parfor i = 1:length(Data)
        Data{i}.shape_gt = Data{i}.shape_gt(params.ind_usedpts, :);
        Data{i}.bbox_gt = getbbox(Data{i}.shape_gt);
    end
    %% augment the data
    Data = initialize(Data, initSet, params);
    
    params.N_img = size(Data, 1);
    params.k = params.k*dist_pupils_ms;
    %% get the pupil distance and groundtruth shape
%     dist_pupils = zeros(params.N_img, 1);
    gtshapes = zeros([size(params.meanshape), params.N_img]);
    ctshapes = zeros([size(params.meanshape), params.N_img]);
    parfor i = 1: length(Data)
        gtshapes(:, :, i) = Data{i}.shape_gt;
        ctshapes(:, :, i) = Data{i}.intermediate_shapes{1};
    end
    
    % Initialization
    Y = cell(params.N_img, 1);
    Error = zeros(1, params.T+1);
    Error(1) = mean(compute_error( gtshapes, ctshapes));
    fprintf('Mean Root Square Error: Initial is %f\n', Error(1));
    Model = cell(params.T, 1);
    %% Explicit Shape Regression
    for t = 1: params.T
        %% normalized targets
        parfor i = 1:params.N_img
            Y{i} = Data{i}.shapes_residual;
        end
        %% learn stage regressors
        fprintf('Start %d th Training...\n', t);
        [prediction_delta, fernCascade] = ShapeRegression(Y, Data, params, t);
        
        % reproject and update the current shape
        parfor i = 1:params.N_img
            % regression targets
            bbx = Data{i}.intermediate_bboxes{t};
            shape_stage = Data{i}.intermediate_shapes{t};
            delta_shape = prediction_delta{i};
            
            [u, v] = transformPointsForward(Data{i}.meanshape2tf, delta_shape(:, 1), delta_shape(:, 2));
            delta_shape_interm_coord = [u, v];
            shape_residual = bsxfun(@times, delta_shape_interm_coord, [bbx(3),bbx(4)]);
            shape_newstage = shape_stage + shape_residual;
            
            ctshapes(:, :, i) = shape_newstage;
            
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
            shape_residual = bsxfun(@rdivide, Data{i}.shape_gt - shape_newstage, Data{i}.intermediate_bboxes{t+1}(3:4));
            [u, v] = transformPointsForward(Data{i}.tf2meanshape, shape_residual(:, 1), shape_residual(:, 2));   
            Data{i}.shapes_residual = [u, v];
        end
        Error(t+1) = mean(compute_error(ctshapes, gtshapes));
        fprintf('Mean Root Square Error in %d iteration is %f\n', t, Error(t+1));
        Model{t}.fernCascade = fernCascade;
    end
    save('Data/Model.mat', 'Model');
    %% show 
    bar(Error);
    xlabel('iterations');
    ylabel('Root Mean Square Error (RMSE)');
end

function [prediction, fernCascade]= ShapeRegression(Y, Data, params, t)
    %% generate local coordinates
    candidate_pixel_location = zeros(params.P, 2);
    nearest_landmark_index = zeros(params.P, 1);
    for i = 1: params.P
        nearest_landmark_index(i) = randi(params.N_fp);
        % sample in mean shape coordinate, [-k, k]
        candidate_pixel_location(i, :) = rand(1, 2)*2*params.k(t) - params.k(t) ;
    end
    %% extrate shape indexed pixel
    intensities = zeros(params.N_img, params.P);
    for i = 1: params.N_img
%         figure
%         imshow(Data{i}.img_gray);
%         hold on
        for j = 1: params.P
            x = candidate_pixel_location(j, 1)* Data{i}.intermediate_bboxes{t}(3);
            y = candidate_pixel_location(j, 2)* Data{i}.intermediate_bboxes{t}(4);
            [project_x, project_y] = transformPointsForward(Data{i}.meanshape2tf, x, y);
            index = nearest_landmark_index(j);
            
            real_x = round(project_x + Data{i}.intermediate_shapes{t}(index, 1));
            real_y = round(project_y + Data{i}.intermediate_shapes{t}(index, 2));
            real_x = max(1, min(real_x, size(Data{i}.img_gray, 2)-1));
            real_y = max(1, min(real_y, size(Data{i}.img_gray, 1)-1));
            intensities(i, j)= Data{i}.img_gray(real_y, real_x);
%             plot(Data{i}.intermediate_shapes{t}(index, 1), Data{i}.intermediate_shapes{t}(index, 2), 'ro');
%             plot(real_x, real_y, 'g+');
            %text(real_x, real_y, num2str(j));
        end
%         hold off
    end
    %% compute pixel-pixel covariance
    covariance = cov(intensities);
    %% train internal level boost regression
    regression_targets = Y; % initialization
    prediction = cell(params.N_img, 1);
    parfor i = 1: params.N_img
        prediction{i} = zeros(params.N_fp, 2);
    end
    
    ferns = cell(params.K, 1);
    
    for i = 1: params.K
        %fprintf('Fern Training: second level is %d out of %d\n', i, params.K);
        [prediction_delta, fern] = fernRegression(regression_targets, intensities, ...
            covariance, nearest_landmark_index, params);%, candidate_pixel_location
        for j = 1: size(prediction_delta,1)
            prediction{j} = prediction{j}+ prediction_delta{j};
            regression_targets{j} = regression_targets{j} - prediction_delta{j};
        end
        ferns{i}.fern = fern;
    end
    fernCascade.ferns = ferns;
    fernCascade.candidate_pixel_location = candidate_pixel_location;
    fernCascade.nearest_landmark_index = nearest_landmark_index;
end

function [prediction, fern] = fernRegression(regression_targets, intensities, ...
    covariance, nearest_landmark_index, params)%, candidate_pixel_locations
    selected_pixel_index = zeros(params.F, 2);
%     selected_pixel_locations = zeros(params.F, 4);
    selected_nearest_landmark_index = zeros(params.F, 2);
    threshold = zeros(params.F,1);
    
    for i = 1: params.F
        v = randn(params.N_fp, 2); % draw a random projection from unit Gaussian
        v = v/norm(v);
        % random projection
        Y_prob = zeros(params.N_img, 1);
        for j = 1: params.N_img
            Y_prob(j) = sum(sum(regression_targets{j}.*v));
        end
        % compute target-pixel covariance
        cov_prob = zeros(1, params.P);
        for j = 1: params.P
            covmatrix = cov(Y_prob, intensities(:, j));
            cov_prob(j) = covmatrix(2);
        end
        % compute variance of Y_prob
        % sigma_prob = std(Y_prob);
        
        max_correlation = -1;
        m_f = 1;
        n_f = 1;
        for m = 1: params.P
            for n = 1: params.P
                sigma_mn = covariance(m,m) + covariance(n, n) - 2*covariance(m, n);
                
                if(abs(sigma_mn)<1e-10)
                    continue;
                end
                flag = 0;
                for p = 1: i
                    if (m == selected_pixel_index(p, 1)) && (n == selected_pixel_index(p, 2))
                        flag = 1;
                        break;
                    elseif (m == selected_pixel_index(p, 2)) && (n == selected_pixel_index(p, 1))
                        flag = 1;
                        break;
                    end
                end
                if(flag)
                    continue; 
                end;
                
                temp = (cov_prob(m) - cov_prob(n))/sqrt(sigma_mn); % it does need sqrt(sigma_mn*sigma_prob)
                if(temp > max_correlation) % do it need abs
                    max_correlation = temp;
                    m_f = m;
                    n_f = n;
                end
            end
        end
        selected_pixel_index(i, 1) = m_f;
        selected_pixel_index(i, 2) = n_f;
        
%         selected_pixel_locations(i,1) = candidate_pixel_locations(m_f,1);
%         selected_pixel_locations(i,2) = candidate_pixel_locations(m_f,2);
%         selected_pixel_locations(i,3) = candidate_pixel_locations(n_f,1);
%         selected_pixel_locations(i,4) = candidate_pixel_locations(n_f,2);
        selected_nearest_landmark_index(i,1) = nearest_landmark_index(m_f); 
        selected_nearest_landmark_index(i,2) = nearest_landmark_index(n_f);
        
        max_diff = -1;
        for j = 1: params.N_img
            temp = intensities(j, m_f) - intensities(j, n_f);
            if (abs(temp) > max_diff)
                max_diff = abs(temp);
            end
        end

        threshold(i) = 0.4*max_diff*rand -0.2*max_diff;
    end
    
    % determine the bins of each shape
    bin_num = 2^params.F;
    shapes_in_bin = zeros(params.N_img, bin_num);
    index2 = zeros(bin_num, 1);
    for i = 1: params.N_img
        index = 0;
        for j = 1: params.F
            intensity_1 = intensities(i, selected_pixel_index(j, 1));
            intensity_2 = intensities(i, selected_pixel_index(j, 2));
            if(intensity_1 - intensity_2) >= threshold(j)
                index = index + 2^(j-1);
            end
        end
        index2(index+1) = index2(index+1) + 1;
        shapes_in_bin(index2(index+1), index+1)= i;
    end

    % get bin output
    prediction = cell(params.N_img,1);
    bin_output = cell(bin_num, 1);
    for i = 1: bin_num
        bin_size = index2(i);
        temp = zeros(params.N_fp, 2);
        if 0 ~= bin_size
            for j = 1: bin_size
                index = shapes_in_bin(j, i);
                temp = temp + regression_targets{index};
            end
        else
            bin_output{i} = temp;
            continue;
        end
        temp = (1/((1+1000/bin_size)*bin_size))*temp;
        bin_output{i} = temp;
        for j = 1: bin_size
            index = shapes_in_bin(j, i);
            prediction{index} = temp;
        end
    end

    fern.bin_output = bin_output;
    fern.selected_pixel_index = selected_pixel_index;
%     fern.selected_pixel_locations = selected_pixel_locations;
    fern.selected_nearest_landmark_index = selected_nearest_landmark_index;
    fern.threshold = threshold;
end