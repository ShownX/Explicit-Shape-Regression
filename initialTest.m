function CShapes = initialTest(data, initSet, params)
    N_train_img = size(initSet, 1);
    Index_init = randperm(N_train_img, params.N_init);
    
%     Images = cell(params.N_init, 1);
%     Bbxes = cell(params.N_init, 1);
    CShapes = zeros(params.N_fp, 2, params.N_init);
    
    for i = 1: params.N_init
        
%         Images{i} = image;
%         Bbxes{i} = bbx;
        current_shape = initSet{Index_init(i)}.shape_gt(params.ind_usedpts,:);
        current_bbx = data.bbox_gt;
        temp = resetshape(current_bbx, current_shape);
        CShapes(:, :, i) = temp;
%         current_shape = projectShape(current_shape, current_bbx);
%         CShapes(:, :, i) = reprojectShape(current_shape, bbx.bbx_chs);
        
%         figure
%         currentshape = CShapes(:, :, i);
%         imshow(Images{i}.faceimg)
%         hold on
%         plot(currentshape(:, 1), currentshape(:, 2), 'g+');
%         hold off
    end
end