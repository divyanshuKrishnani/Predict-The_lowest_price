function [J,D]=cost2(X,Y,Theta,Lambda)
    m=size(X,1);
    J=((X*Theta-Y)'*(X*Theta-Y))/2/m+Lambda/2/m*(Theta(2:end)'*Theta(2:end));
    D=((X)'*(X*Theta-Y))/m + Lambda/m*[0;Theta(2:end)];
    D=D(:);
end

    
    
