clc
clear
close all
Data=readtable('Dataset/Train.csv');
Data_test=readtable('Dataset/Test.csv');
Data_id=table2array(Data(:,1));
Data_id_test=table2array(Data_test(:,1));
Data_date=table2array(Data(:,2));
Data_date_test=table2array(Data_test(:,2));
Date=yyyymmdd(Data_date);
Date_test=yyyymmdd(Data_date_test);
Year=floor(Date/10000);
rest=mod(Date,10000);
Month=floor(rest/100);
Day=mod(rest,100);
Year_test=floor(Date_test/10000);
rest_test=mod(Date_test,10000);
Month_test=floor(rest_test/100);
Day_test=mod(rest_test,100);
X=[table2array(Data(:,3:end-2)) table2array(Data(:,end))];
Y=table2array(Data(:,end-1));
X=[X Day Month Year];
X_test=[table2array(Data_test(:,3:end))];
X_test=[X_test Day_test Month_test Year_test];

m=size(X,1);
m_test=size(X_test,1);
%X=[ones(m,1) X];

% segregate=randperm(m,m);
% m1=floor(0.6*m);
% m2=m-m1;
% X_train=X(segregate(1:m1),:);
% Y_train=Y(segregate(1:m1),:);
% X_val=X(segregate(m1+1:end),:);
% Y_val=Y(segregate(m1+1:end),:);
n=size(X,2);
n_test=size(X_test,2);
%% using Leave one out Encoding for State of country, Mkt Cat and Prod Cat
% State, Mkt, prod,Grade,Demand, High Cap Price, DAy, Month, Year
R=1;
UNI_State=unique(X(:,1));
UNI_Mkt=unique(X(:,2));
UNI_Product=unique(X(:,3));
count_State=zeros(size(UNI_State,1),1);
count_Mkt=zeros(size(UNI_Mkt,1),1);
count_Product=zeros(size(UNI_Product,1),1);
target_State=zeros(size(UNI_State,1),1);
target_Mkt=zeros(size(UNI_Mkt,1),1);
target_Product=zeros(size(UNI_Product,1),1);
for i=1:size(UNI_State,1)
    count_State(i)=sum(X(:,1)==UNI_State(i));
    target_State(i)=sum(Y(X(:,1)==UNI_State(i)));
end
for i=1:size(UNI_Mkt,1)
    count_Mkt(i)=sum(X(:,2)==UNI_Mkt(i));
    target_Mkt(i)=sum(Y(X(:,2)==UNI_Mkt(i)));
end
for i=1:size(UNI_Product,1)
    count_Product(i)=sum(X(:,3)==UNI_Product(i));
    target_Product(i)=sum(Y(X(:,3)==UNI_Product(i)));
end
Encoded_State=zeros(m,1);
Encoded_Mkt=zeros(m,1);
Encoded_Product=zeros(m,1);
Encoded_State_test=zeros(m_test,1);
Encoded_Mkt_test=zeros(m_test,1);
Encoded_Product_test=zeros(m_test,1);
for i=1:m
    index_State=find(UNI_State==X(i,1),1);

    Encoded_State(i)=(target_State(index_State)-Y(i))/(count_State(index_State)-1+R)*(1+randn(1));

    index_Mkt=find(UNI_Mkt==X(i,2),1);

    Encoded_Mkt(i)=(target_Mkt(index_Mkt)-Y(i))/(count_Mkt(index_Mkt)-1+R)*(1+randn(1));

    index_Product=find(UNI_Product==X(i,3),1);

    Encoded_Product(i)=(target_Product(index_Product)-Y(i))/(count_Product(index_Product)-1+R)*(1+randn(1));
end   
for i=1:m_test

    index_State_test=find(UNI_State==X_test(i,1),1);
    index_Mkt_test=find(UNI_Mkt==X_test(i,2),1);
    index_Product_test=find(UNI_Product==X_test(i,3),1);
    if size(index_State_test,1)==0
        Encoded_State_test(i)=0;
    else
        Encoded_State_test(i)=(target_State(index_State_test))/(count_State(index_State_test)+R);
    end
    if size(index_Mkt_test,1)==0
        Encoded_Mkt_test(i)=0;
    else
        Encoded_Mkt_test(i)=(target_Mkt(index_Mkt_test))/(count_Mkt(index_Mkt_test)+R);
    end
    if size(index_Product_test,1)==0
        Encoded_Product_test(i)=0;
    else
        Encoded_Product_test(i)=(target_Product(index_Product_test))/(count_Product(index_Product_test)+R);
    end
end
X=[Encoded_State Encoded_Mkt Encoded_Product X(:,4:end)];
X_test=[Encoded_State_test Encoded_Mkt_test Encoded_Product_test X_test(:,4:end)];

% z_State=[];
% z_Mkt=[];
% z_Product=[];
% for i=1:size(X,1)
%     z_State=[z_State; (X(i,1)==UNI_State)'];
%     z_Mkt=[z_Mkt; (X(i,2)==UNI_Mkt)'];
%     z_Product=[z_Product; (X(i,3)==UNI_Product)'];
% end
% X=[z_State z_Mkt z_Product X(:,4:end)];    
% 
% 

%% adding polynomial features
% p=3;
% X_Poly=[];
% for i=1:10
%     X_Poly=[X_Poly X.^(i)];
% end
% X=X_Poly;

%%
%X=Feature_normalise(X);

% 
% Lambda=0;
% 
% initial_Theta=randn(n,1);
% [J ,D]=cost(X_train,Y_train,initial_Theta,Lambda);
% Cost_function=@(Theta) cost(X_train,Y_train,Theta,Lambda);
% options=optimset('MaxIter', 1000,'GradObj','on' );
% opt_theta=fmincg(Cost_function,initial_Theta, options);


%% decreatising Demand, High Cap Price, Y





%% training decision tree
X=[X X(:,4).*X(:,5) X(:,5).*X(:,6)];
X_test=[X_test X_test(:,4).*X_test(:,5) X_test(:,5).*X_test(:,6)];
[X, mu, sigma]=Feature_normalise(X);
temp=bsxfun(@minus,X_test,mu);
X_test=bsxfun(@rdivide, temp,sigma);
tree=fitrtree(X,Y,'MaxNumSplits',10+5*(19));

prediction=predict(tree,X_test);

%scatter(Y_val,prediction)
%eval=100-((log(prediction+1)-log(Y_val+1))'*(log(prediction+1)-log(Y_val+1)))/m2;
pred_train=predict(tree,X);
eval=100-((log(pred_train+1)-log(Y+1))'*(log(pred_train+1)-log(Y+1)))/m;

Target_values=unique(pred_train);

Theta_train=zeros(size(Target_values,1),9);
for int=1:size(Target_values,1)
    Index_reg{int}=find(pred_train==Target_values(int));
    %X_train_reg_temp=[ones(size(Index_reg{int},1),1) X_train(Index_reg{int},4:end) X_train(Index_reg{int},4).*X_train(Index_reg{int},5) X_train(Index_reg{int},5).*X_train(Index_reg{int},6) X_train(Index_reg{int},4)./X_train(Index_reg{int},5) ];
    X_train_reg_temp=[ones(size(Index_reg{int},1),1) X(Index_reg{int},4:end)];
    Y_train_reg_temp=Y(Index_reg{int});
    if ~isequal(imag(X_train_reg_temp),zeros(size(X_train_reg_temp)))
        pause
    end
    if ~isequal(imag(Y_train_reg_temp),zeros(size(Y_train_reg_temp)))
        pause
    end
    initial_Theta=zeros(9,1);
    Lambda=50;
    [J ,D]=cost2(X_train_reg_temp,Y_train_reg_temp,initial_Theta,Lambda);
    
    Cost_function=@(Theta) cost2(X_train_reg_temp,Y_train_reg_temp,Theta,Lambda);
    options=optimset('MaxIter', 1000,'GradObj','on' );
    opt_theta=fminunc(Cost_function,initial_Theta, options);
    Theta_train(int,:)=(opt_theta)';
%     scatter(Y_train_reg_temp,pred_train(Index_reg{int}))
%     hold on
%     plot(Y_train_reg_temp,X_train_reg_temp*opt_theta, 'c*')
%     hold off
end
for int=1:size(X_test,1)
    Index_reg=find(Target_values==prediction(int));
    if size(Index_reg,1)==0
        continue
    end
    req_theta=Theta_train(Index_reg,:)';
    %X_val_reg_temp=[1 X_val(int,4:end) X_val(int,4)*X_val(int,5) X_val(int,5)*X_val(int,6) X_val(int,4)/X_val(int,5)];
    X_val_reg_temp=[1 X_test(int,4:end)];
    prediction(int)=X_val_reg_temp*req_theta;
    
end
%eval2=100-((log(prediction+1)-log(Y_val+1))'*(log(prediction+1)-log(Y_val+1)))/m2;

%%
% Y_pred_val=X_val*opt_theta;
% scatter(Y_val,Y_pred_val)
% hold on
% plot(linspace(1,18000,20),linspace(1,18000,20))
% hold off
% prob_val=cost(X_val,Y_val,opt_theta,0)
% [prob_train,~]=cost(X_train,Y_train,opt_theta,0)