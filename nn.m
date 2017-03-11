function [output,cost] = nn(data,realval)
% data is a vector of size m x n where m in the number or tests to run, and n
% is the number of inputs into the neural network. realval is a vector of
% the training data for the neural network. If realval is not provided, the
% program outputs the predictions given the previous training data.

% f = the activation function
% W1 = the weights to be multiplied by the data to get Z1
% W2 = the weights to be multiplied by the hidden layer to get Z2
% Z1 = the input for the activation function to yield the hidden layer vals
% Z2 = the input for the activation function to yield the answer
% data = the original input data to be processed
% a = the data in the hidden layer to be processed into a final answer
% y = the final answer

global W1;
global W2;
global Z1;
global Z2;
global a;
global f;
learningspeed = 0.1;
hiddenlayersize = 6;
f = @(z) tanh(z);
df = @(z) (1 - f(z).^2);
% Choose an activation function wisely. First, make sure that the minimum
% and maximum output value of the function are correct. Next, decide how
% you want to manipulate the output. Do you want a discrete 1 or 0 as an
% output? Then choose a step function. Do you want to be able to
% differentiate between data that is clustered in one area? Choose tanh. Do
% you want regression without an activation function? Use linear.

% Creates weight matrices if the program has not yet been trained
% Makes sure input matrices have the correct dimensions
if isempty(W1)
    W1 = (rand(length(data(1,:))+1,hiddenlayersize)-0.5);
    W2 = (rand(hiddenlayersize+1,1)-0.5);
else
    if length(data(1,:)) ~= length(W1(:,1))-1
        error('There must be %d inputs, but the provided data matrix has %d.\n',length(W1(:,1))-1,length(data(1,:)));
    end
end
if nargin == 2
    if length(data(:,1)) ~= length(realval)
        error('The number of data samples (%d) must equal the number of training data outputs (%d)\n',length(data),length(realval));
    end
    [m,n] = size(realval);
    if ~(m == 1 || n == 1)
        error('The training data must be either a row or column vector. It currently has size %dx%d\n',m,n);
    end
    if ~iscolumn(realval)
        realval = realval';
    end
end

data = [ones(size(data(:,1))) data]; % add bias unit to inputs

% If no training data is given, the program simply runs the neural network
if nargin == 1
    output = evaluate(data);
    return
end

cost = norm(evaluate(data)-realval);

% Use backpropagation to find the gradient
dJdW1 = data' * ((((evaluate(data) - realval) .* df(Z2)) * W2(2:end,1)') .* df(Z1));
dJdW2 = a' * ((evaluate(data) - realval) .* df(Z2));

% Use gradient descent to approach an optimum
W1 = W1 - learningspeed * dJdW1;
W2 = W2 - learningspeed * dJdW2;

% Run neural network with updated weights
output = evaluate(data);

return


function y = evaluate(data)
global W1;
global W2;
global Z1;
global Z2;
global a;
global f;
Z1 = data * W1;
a = ones(length(Z1(:,1)),length(Z1(1,:))+1); % add bias to hidden layer
a(:,2:end) = f(Z1); % the outputs from layer 1 don't affect bias
Z2 = a*W2;
y = f(Z2);
return