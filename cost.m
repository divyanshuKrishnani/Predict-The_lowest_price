function [J,D]=cost(X,Y,Theta,Lambda)
    m=size(X,1);
    J=((log(X*Theta+1)-log(Y+1))'*(log(X*Theta+1)-log(Y+1)))/2/m+Lambda/2/m*(Theta(2:end)'*Theta(2:end));
    D=(X'*((log(X*Theta+1)-log(Y+1))./(X*Theta+1)))/m + Lambda/m*[0;Theta(2:end)];
    D=D(:);
end

    

    
