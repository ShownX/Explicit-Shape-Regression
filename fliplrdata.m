function data_flip = fliplrdata(data)
    data_len = size(data, 1);
    data_flip = cell(data_len, 1);
    parfor i = 1:data_len
        data_flip{i}.img_gray    = fliplr(data{i}.img_gray);
        data_flip{i}.width_orig  = data{i}.width_orig;
        data_flip{i}.height_orig = data{i}.height_orig;    
        data_flip{i}.width       = data{i}.width;
        data_flip{i}.height      = data{i}.height; 
        
        data_flip{i}.shape_gt    = flipshape(data{i}.shape_gt);        
        data_flip{i}.shape_gt(:, 1)  =  data{i}.width - data_flip{i}.shape_gt(:, 1);
        
        data_flip{i}.bbox_gt        = data{i}.bbox_gt;
        data_flip{i}.bbox_gt(1)     = data_flip{i}.width - data_flip{i}.bbox_gt(1) - data_flip{i}.bbox_gt(3);       
        
    end
end