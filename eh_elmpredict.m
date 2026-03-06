function Y = eh_elmpredict(P, IW, B, LW, TF, TYPE)
% ELMPREDICT  Predict with Extreme Learning Machine (samples in rows)
%
% Input
%   P   - Input matrix of test set  (Qt×R)   % samples × features
%   IW  - Input weight matrix       (R×N)
%   B   - Bias row vector           (1×N)
%   LW  - Output weight matrix      (N×S)
%   TF  - Transfer function: 'sig' (default) | 'sin' | 'hardlim'
%   TYPE- 0: regression (default) | 1: classification
%
% Output
%   Y   - If TYPE==0: predictions (Qt×S)
%         If TYPE==1: class indices (Qt×1, values in 1..S)
%
% See also: elmtrain

if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end

[Qt, ~] = size(P);                 % samples × features

% Hidden layer output H: Qt×N
BiasMatrix = ones(Qt,1) * B;       % Qt×N
tempH = P * IW + BiasMatrix;       % (Qt×R)*(R×N) + (Qt×N) = Qt×N

switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
    otherwise
        error('ELM:Arguments','Unknown transfer function: %s', TF);
end

Scores = H * LW;                   % Qt×S

if TYPE == 1
    % Multi-class: pick argmax per row (sample)
    [~, Y] = max(Scores, [], 2);   % Qt×1
else
    % Regression
    Y = Scores;                    % Qt×S
end
end
