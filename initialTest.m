function [CShapes]= initialTest(image, bbx, train_bbx, train_pts, params)
    N_train_img = size(train_pts, 1);
    Index_init = randperm(N_train_img, params.N_init);
    
%     Images = cell(params.N_init, 1);
%     Bbxes = cell(params.N_init, 1);
    CShapes = zeros(params.N_fp, 2, params.N_init);
    
    for i = 1: params.N_init
        
%         Images{i} = image;
%         Bbxes{i} = bbx;
        current_shape = train_pts{Index_init(i)}.pts_chs;
        current_bbx = train_bbx{Index_init(i)}.bbx_chs;
        current_shape = projectShape(current_shape, current_bbx);
        CShapes(:, :, i) = reprojectShape(current_shape, bbx.bbx_chs);
        
%         figure
%         currentshape = CShapes(:, :, i);
%         imshow(Images{i}.faceimg)
%         hold on
%         plot(currentshape(:, 1), currentshape(:, 2), 'g+');
%         hold off
    end
end