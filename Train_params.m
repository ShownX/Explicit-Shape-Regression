function params = Train_params()
    params.T = 10; % the iteration stages
    params.P = 400; % default = 400, the pixel number sampled on the images
    params.K = 500; % default = 500, the number of fern on the internal-level boosted regression
    params.k = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]; 
    % the local scale of search, it set 0.3 times of the distance between two pupils on the mean shape
    params.F = 5; % the number of features in fern
    
    params.N_aug = 20;
    
    params.N_fp = 0; %size(params.mean_shape, 1);
    params.N_img = 0;
    
    params.flip = 1; % mirror the data
    params.ind_usedpts = 18:68;
    %params.k = params.k*pdist([mean([params.mean_shape(20, :); params.mean_shape(23, :)]);...
    %    mean([params.mean_shape(26, :); params.mean_shape(29, :)])])/2;
end