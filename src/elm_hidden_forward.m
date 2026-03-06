function [IW,B,H] = elm_hidden_forward(P,N,TF)
% Random hidden layer + forward H
[Q,R] = size(P);
IW = rand(R,N)*2 - 1;
B  = rand(1,N);
BiasMatrix = repmat(B, Q, 1);
tempH = P * IW + BiasMatrix;
H = elm_activation(tempH, TF);
end
