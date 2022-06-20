% https://github.com/cvlab-stonybrook/DewarpNet/issues/22
ref_folder = "F:/kong_model2/Prepare_dataset/scan/";
in_folder = "C:/Users/CVML/Downloads/dewarpnet-public-20220615T191454Z-001/dewarpnet-public/";

tic
files=dir(in_folder);
mean_ms=0.0;
mean_ld=0.0;
ms_all=[];
ld_all=[];
for i=1:65
    fprintf(ref_folder+string(i)+'.png')
    ref_img=rgb2gray(imread(char(ref_folder+string(i)+'.png')));
    for j=1:2
        fname=string(i)+'_'+string(j)+' copy.png';
        char(in_folder+string(fname))
        tst_img=rgb2gray(imread(char(in_folder+string(fname))));
        %imshow(tst_img)
        [rh,rw,~]=size(ref_img);
        [th,tw,tc]=size(tst_img);
        ref_img=imresize(ref_img,sqrt(598400/(rh*rw)),'bicubic');
        [rh,rw,rc]=size(ref_img);
        tst_img=imresize(tst_img,[rh rw],'bicubic');
        fprintf('\n---- mean , %f , %f ----\n', rh, rw);
        %imshow(ref_img)
        %imshow(tst_img)
        [ms, ld]=evalUnwarp(tst_img, ref_img);
        fprintf('%s , %f , %f\n',fname,ms,ld);
        mean_ms=mean_ms+ms;
        mean_ld=mean_ld+ld;
    end
end

mean_ms=mean_ms/130
mean_ld=mean_ld/130

fprintf('\n---- mean , %f , %f ----\n',mean_ms,mean_ld);
fprintf(' time = %g sec\n', toc);  % ��ܮɶ�
