function [val,grad] = tikhonov(x_left,x_right,x_up,x_down,x,y,alpha,gamma)
 
    
    val_m=alpha*x.*x;
    val=val_m(:);
    grad_m = 2*alpha*x;
    grad=grad_m(:);
end
