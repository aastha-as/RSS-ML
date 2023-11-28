%{
 RSS-ML algorithm provides an accurate ML estimate of source location based on RSS measurements.
 The algorithm directly solves the MLE problem without any approximation 
%}
%** Input parameters:**%
%{
 y (m x 1): difference of reference power and rss mesured at m sensors, divided by \eta (10*\alpha/log(10)),
 s_in (n x m) : m sensors position coordinates (n),
 x0 (n x 1): initial source position
 del: threshold to check convergence of x
%}

%% Algorithm RSS-ML
function [xo_rssml,obj] = rss_ml(yi,s_in,x0,del)
    xt = x0;                                                                % Initial source position
    m = size(s_in,2);                                                       % Number of sensors
    for iter=1:250
        B=xt-s_in;                                                          % Matrix with each column as \rho_i
        beta2=vecnorm(B,2).^2;                                              
        betaa=1./beta2';                                                    % Vector with each element as 1/||\rho_i||^2
        a=log(beta2')-1;
        alphaa=betaa.*((vecnorm(s_in,2)).^2)';
        C=2*betaa'.*s_in;                                                   % Compute C
        D=2*yi'.*B;                                                         % Compute D
        d=D'*xt-yi.*beta2';
        cvx_begin quiet                                                     % Solve convex problem
            variables q(m,1) z(m,1)
            minimize ((q'*q)-q'*(a+alphaa)+quad_over_lin(C*q+D*z,4*q'*betaa)-(yi'*(log(z)))-(z'*d))
            subject to 
               q>0
               z>0
        cvx_end
        x=(C*q+D*z)/(2*q'*betaa);                                           % Next x iterate
        obj(iter)=sum((-yi' + log(vecnorm(x-s_in,2))).^2);                  % Objective function value
        % check for convergence
        if (iter>1) && abs((obj(iter)-obj(iter-1))/obj(iter-1))<del         % Convergence check criterion
            break;
        end
        xt = x;
    end
    xo_rssml = x;
end
 
 
 
 