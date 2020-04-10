im_chest=imread('../../data/ChestCT.png');
im_shep=imread('../../data/SheppLogan256.png');
im_chest=double(im_chest);
im_shep=double(im_shep);
%im_chest=mat2gray(double(im_chest));
%im_shep=mat2gray(double(im_shep));
%imshow(im,[]);

R_chest=radon(im_chest,0:359);
R_shep=radon(im_shep,0:359);
rrmse_chest=[];
rrmse_shep=[];
theta=[];
for i=0:180
    a=i;
    
    b=i+150;
    output_size_chest=512;
    output_size_shep=256;
    x_chest=iradon(R_chest(:,a+1:b+1),a:b,output_size_chest);
    x_shep=iradon(R_shep(:,a+1:b+1),a:b,output_size_shep);
    %x_chest=mat2gray(x_chest);
    %x_shep=mat2gray(x_shep);
    r_shep=norm(double(x_shep-im_shep),'fro')/norm(double(im_shep),'fro');
    r_chest=norm(double(x_chest-im_chest),'fro')/norm(double(im_chest),'fro');
    rrmse_chest=[rrmse_chest r_chest];
    rrmse_shep=[rrmse_shep r_shep];
    theta=[theta i];

    end
    
[val_chest,arg_chest]=min(rrmse_chest);
[val_shep,arg_shep]=min(rrmse_shep);
x_chest_best=iradon(R_chest(:,arg_chest+1:arg_chest+150+1),arg_chest:arg_chest+150,512);
figure(1)
title('ChestCT')
imshow(x_chest_best,[]);
x_shep_best=iradon(R_shep(:,arg_shep+1:arg_shep+150+1),arg_shep:arg_shep+150,256);
figure(2)
title('SheppLgan256')
imshow(x_shep_best,[]);
figure(3)
plot(theta,rrmse_chest);
xlabel('theta')
ylabel('RRMSE')
title('RRMSE vs Theta for ChestCT')
figure(4)
plot(theta,rrmse_shep);
xlabel('theta')
ylabel('RRMSE')
title('RRMSE vs Theta for SheppLogan256')
%[val, arg]=min(rrmse);
%arg



