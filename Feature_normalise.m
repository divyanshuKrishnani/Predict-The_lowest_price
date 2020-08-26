function [X_norm, mu ,Sigma]= Feature_normalise(X)
    mu=mean(X);
    Sigma=std(X);
    temp=bsxfun(@minus,X,mu);
    X_norm=bsxfun(@rdivide, temp,Sigma);
end