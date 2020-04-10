s='../../data/ChestPhantom.png';
image=double(imread(s))/255;
%image=double(image);

figure(1);
imshow(image,[]);
title('Original Image');
n=128*128;
A=zeros(185*180,n);
for col=1:128
    for row=1:128
      %  col
    a=zeros(128,128);
    a(row,col)=1;
    r=radon(a);
    r1=reshape(r,[185*180,1]);
    A(:,(col-1)*128+row)=r1;
    end
end
A=sparse(A);
i=image;
x=reshape(i,[128*128,1]);
b_calc=A*x;
maxi=max(b_calc);
mini=min(b_calc);
st_dev=0.02*(maxi-mini);
noise=st_dev.*randn(180*185,1);
b=b_calc+noise;

radon_new = reshape(b,[185,180]);
theta = 0:179;
ift_R = myFilter(radon_new,'Cosine',theta,1);
backprojected = iradon(ift_R,theta,'linear','None',1,size(i,1));


figure(2);
imshow(backprojected,[]);
title('myFilter Reconstructed');
RRMSE_back = norm(double(i-backprojected),'fro')/norm(double(i),'fro');
fprintf('RRMSE for backprojected image is %f\n',RRMSE_back);




%%%%%%%%%%%%%%%%%%%%%%%%%%%tikhonov
x_initial_tikh=zeros(128*128,1);
alpha_tikh_opt=0.25;
gamma_tikh=1;
alpha_tikh = alpha_tikh_opt;
[x_final_tikh,objf_tikh]=gradient_descent(x_initial_tikh,alpha_tikh,@tikhonov,A,b,gamma_tikh);
x_final_tikh_m=reshape(x_final_tikh,[128,128]);
im1=x_final_tikh_m;
figure(3);
imshow(im1);
title('Tikhonov Reconstructed');
RRMSE_tikh_1 = norm(double(i - im1),'fro')/norm(double(i),'fro');

alpha_tikh = (1.2)*alpha_tikh_opt;
[x_final_tikh,objf_tikh]=gradient_descent(x_initial_tikh,alpha_tikh,@tikhonov,A,b,gamma_tikh);
x_final_tikh_m=reshape(x_final_tikh,[128,128]);
im1=x_final_tikh_m;
RRMSE_tikh_2 = norm(double(i - im1),'fro')/norm(double(i),'fro');

alpha_tikh = (0.8)*alpha_tikh_opt;
[x_final_tikh,objf_tikh]=gradient_descent(x_initial_tikh,alpha_tikh,@tikhonov,A,b,gamma_tikh);
x_final_tikh_m=reshape(x_final_tikh,[128,128]);
im1=x_final_tikh_m;
RRMSE_tikh_3 = norm(double(i - im1),'fro')/norm(double(i),'fro');

fprintf('Tikhonov parameters');
fprintf('alpha = %.3f\n',alpha_tikh_opt);
fprintf('RRMSE(alpha) = %f\n',RRMSE_tikh_1);
fprintf('RRMSE(1.2*alpha) = %f\n',RRMSE_tikh_2);
fprintf('RRMSE(0.8*alpha) = %f\n',RRMSE_tikh_3);
fprintf('\n');

%pause

%%%%%%%%%%%%%%%quadratic%%%%%%%%%%%%%%%%%%%
x_initial_quad=zeros(128*128,1);
alpha_grad_opt=0.5;
gamma_grad=1;
alpha_grad = alpha_grad_opt;
[x_final_quad,objf_quad]=gradient_descent(x_initial_quad,alpha_grad,@quadratic,A,b,gamma_grad);
x_final_quad_m=reshape(x_final_quad,[128,128]);
im2=x_final_quad_m;
figure(4);
imshow(im2);
title('Quadratic Prior Reconstructed');
RRMSE_quad_1 = norm(double(i-im2),'fro')/norm(double(i),'fro');

alpha_grad = 1.2*alpha_grad_opt;
[x_final_quad,objf_quad]=gradient_descent(x_initial_quad,alpha_grad,@quadratic,A,b,gamma_grad);
x_final_quad_m=reshape(x_final_quad,[128,128]);
im2=x_final_quad_m;
RRMSE_quad_2 = norm(double(i-im2),'fro')/norm(double(i),'fro');

alpha_grad = 0.8*alpha_grad_opt;
[x_final_quad,objf_quad]=gradient_descent(x_initial_quad,alpha_grad,@quadratic,A,b,gamma_grad);
x_final_quad_m=reshape(x_final_quad,[128,128]);
im2=x_final_quad_m;
RRMSE_quad_3 = norm(double(i-im2),'fro')/norm(double(i),'fro');

fprintf('QUADRATIC PRIOR PARAMETERS\n');
fprintf('alpha = %.3f\n',alpha_grad_opt);
fprintf('RRMSE(alpha) = %f\n',RRMSE_quad_1);
fprintf('RRMSE(1.2*alpha) = %f\n',RRMSE_quad_2);
fprintf('RRMSE(0.8*alpha) = %f\n',RRMSE_quad_3);
fprintf('\n');

%%%%%%%%%%%%%%%%%huber%%%%%%%%%%%%%%%%%%%%%%
x_initial_huber=zeros(128*128,1);
alpha_huber_opt=1;
gamma_huber_opt=9;

alpha_huber = alpha_huber_opt;
gamma_huber = gamma_huber_opt;
[x_final_huber,objf_huber]=gradient_descent(x_initial_huber,alpha_huber,@Huber,A,b,gamma_huber);
x_final_huber_m=reshape(x_final_huber,[128,128]);
im3=x_final_huber_m;
figure(5);
imshow(im3);
title('Huber Prior Reconstructed');
RRMSE_huber_1 = norm(double(i-im3),'fro')/norm(double(i),'fro');

alpha_huber = 0.8*alpha_huber_opt;
gamma_huber = gamma_huber_opt;
[x_final_huber,objf_huber]=gradient_descent(x_initial_huber,alpha_huber,@Huber,A,b,gamma_huber);
x_final_huber_m=reshape(x_final_huber,[128,128]);
im3=x_final_huber_m;
RRMSE_huber_2 = norm(double(i-im3),'fro')/norm(double(i),'fro');

alpha_huber = 1.2*alpha_huber_opt;
gamma_huber = gamma_huber_opt;
[x_final_huber,objf_huber]=gradient_descent(x_initial_huber,alpha_huber,@Huber,A,b,gamma_huber);
x_final_huber_m=reshape(x_final_huber,[128,128]);
im3=x_final_huber_m;
RRMSE_huber_3 = norm(double(i-im3),'fro')/norm(double(i),'fro');

alpha_huber = alpha_huber_opt;
gamma_huber = 0.8*gamma_huber_opt;
[x_final_huber,objf_huber]=gradient_descent(x_initial_huber,alpha_huber,@Huber,A,b,gamma_huber);
x_final_huber_m=reshape(x_final_huber,[128,128]);
im3=x_final_huber_m;
RRMSE_huber_4 = norm(double(i-im3),'fro')/norm(double(i),'fro');

alpha_huber = alpha_huber_opt;
gamma_huber = 1.2*gamma_huber_opt;
[x_final_huber,objf_huber]=gradient_descent(x_initial_huber,alpha_huber,@Huber,A,b,gamma_huber);
x_final_huber_m=reshape(x_final_huber,[128,128]);
im3=x_final_huber_m;
RRMSE_huber_5 = norm(double(i-im3),'fro')/norm(double(i),'fro');

fprintf('HUBER PRIOR PARAMETERS\n');
fprintf('alpha = %.3f\n',alpha_huber_opt);
fprintf('gamma = %.3f\n',gamma_huber_opt);
fprintf('RRMSE(alpha,gamma) = %f\n',RRMSE_huber_1);
fprintf('RRMSE(1.2*alpha,gamma) = %f\n',RRMSE_huber_2);
fprintf('RRMSE(0.8*alpha,gamma) = %f\n',RRMSE_huber_3);
fprintf('RRMSE(alpha,0.8*gamma) = %f\n',RRMSE_huber_4);
fprintf('RRMSE(alpha,1.2*gamma) = %f\n',RRMSE_huber_5);
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_initial_disc=zeros(128*128,1);
alpha_disc_opt=0.80;
gamma_disc_opt=4.8;
 
 alpha_disc = alpha_disc_opt;
 gamma_disc = gamma_disc_opt;
 [x_final_disc,objf_disc]=gradient_descent(x_initial_disc,alpha_disc,@Disc,A,b,gamma_disc);
x_final_disc_m=reshape(x_final_disc,[128,128]);
im4=x_final_disc_m;
figure(6);
imshow(im4);
title('Discontinuity Prior Reconstructed');
RRMSE_disc_1 = norm(double(i-im4),'fro')/norm(double(i),'fro');
 
% % 
alpha_disc = alpha_disc_opt*1.2;
gamma_disc = gamma_disc_opt;
[x_final_disc,objf_disc]=gradient_descent(x_initial_disc,alpha_disc,@Disc,A,b,gamma_disc);
x_final_disc_m=reshape(x_final_disc,[128,128]);
im4=x_final_disc_m;
RRMSE_disc_2 = norm(double(i-im4),'fro')/norm(double(i),'fro');
 
 
alpha_disc = alpha_disc_opt*0.8;
gamma_disc = gamma_disc_opt;
[x_final_disc,objf_disc]=gradient_descent(x_initial_disc,alpha_disc,@Disc,A,b,gamma_disc);
x_final_disc_m=reshape(x_final_disc,[128,128]);
im4=x_final_disc_m;
RRMSE_disc_3 = norm(double(i-im4),'fro')/norm(double(i),'fro');
%  
% % 
alpha_disc = alpha_disc_opt;
gamma_disc = gamma_disc_opt*0.8;
[x_final_disc,objf_disc]=gradient_descent(x_initial_disc,alpha_disc,@Disc,A,b,gamma_disc);
x_final_disc_m=reshape(x_final_disc,[128,128]);
im4=x_final_disc_m;
RRMSE_disc_4 = norm(double(i-im4),'fro')/norm(double(i),'fro');
 
alpha_disc = alpha_disc_opt;
gamma_disc = gamma_disc_opt*1.2;
[x_final_disc,objf_disc]=gradient_descent(x_initial_disc,alpha_disc,@Disc,A,b,gamma_disc);
x_final_disc_m=reshape(x_final_disc,[128,128]);
im4=x_final_disc_m;
RRMSE_disc_5 = norm(double(i-im4),'fro')/norm(double(i),'fro');
% % 
% 
fprintf('DISCONTINUITY PRIOR PARAMETERS\n')
fprintf('alpha = %.3f\n',alpha_disc_opt);
fprintf('gamma = %.3f\n',gamma_disc_opt);
fprintf('RRMSE(alpha,gamma) = %f\n',RRMSE_disc_1);
fprintf('RRMSE(1.2*alpha,gamma) = %f\n',RRMSE_disc_2);
fprintf('RRMSE(0.8*alpha,gamma) = %f\n',RRMSE_disc_3);
fprintf('RRMSE(alpha,0.8*gamma) = %f\n',RRMSE_disc_4);
fprintf('RRMSE(alpha,1.2*gamma) = %f\n',RRMSE_disc_5);
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





    
    