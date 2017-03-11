% Allows the user to input training data, and shows a visualization
% as the network trains itself
close all
clear
clc

iterations = 1000;
viewsize = 15;
data = 3*[1 2; 0.2 0.2; 2 1; -1 -2; -0.2 -0.2; -2 -1; -1 2; -0.2 0.2; -2 1; 1 -2; 0.2 -0.2; 2 -1];
realval = [1;-1;1;1;1;1;1;-1;-1;1;1;1];

figure
hold on
plot(data(realval==1,1),data(realval==1,2),'*g');
plot(data(realval==-1,1),data(realval==-1,2),'*k');
legend('1','-1');
title('Training Data');
axis([-floor((viewsize-1)/2),ceil((viewsize-1)/2),-floor((viewsize-1)/2),ceil((viewsize-1)/2)]);
hold off
fig = figure;
[~,lastcost] = nn(data,realval);
for n = 1:iterations
    [output, cost] = nn(data,realval);
    if cost > 0.1 && abs((cost-lastcost)/cost) < 1*10^-5
        clear global;
    end
    lastcost = cost;
    
    classification = zeros(viewsize,viewsize);
    for x1 = -floor((viewsize-1)/2):ceil((viewsize-1)/2)
        for x2 = -floor((viewsize-1)/2):ceil((viewsize-1)/2)
            classification(x2+(1+floor((viewsize-1)/2)),x1+(1+floor((viewsize-1)/2))) = nn([x1,x2]);
        end
    end
    classification = imresize(classification,viewsize*4);
    set(groot,'CurrentFigure',fig);
    imshow(flipud(classification));
    colormap default;
    text(viewsize*1.6,viewsize*2,sprintf('Cost: %0.4f',cost),'Color','red','FontSize',14)
    pause(0.1);
end