function output = optimizenn(data,realval)
% Automatically trains the network given a set of input data until
% it is optimized
eps = 0.01;
lastcost = eps+100;
while lastcost > eps
    [output, cost] = nn(data,realval);
    if abs((cost-lastcost)/cost) < 10^-8
        clear global;
    end
    lastcost = cost;
    disp(cost);
end

viewsize = 15;
classification = zeros(viewsize,viewsize);
for x1 = -floor((viewsize-1)/2):ceil((viewsize-1)/2)
	for x2 = -floor((viewsize-1)/2):ceil((viewsize-1)/2)
        classification(x2+(1+floor((viewsize-1)/2)),x1+(1+floor((viewsize-1)/2))) = nn([x1,x2]);
    end
end
classification = imresize(classification,viewsize*4);
imshow(flipud(classification));
colormap default;