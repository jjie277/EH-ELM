function [P,T,N,TF,TYPE] = elmtrain_argcheck(P,T,N,TF,TYPE)
if nargin < 2, error('ELM:Arguments','Not enough input arguments.'); end
if nargin < 3 || isempty(N), N = size(P,1); end
if nargin < 4 || isempty(TF), TF = 'sig'; end
if nargin < 5 || isempty(TYPE), TYPE = 0; end
if size(P,1) ~= size(T,1)
    error('ELM:Arguments','Rows (samples) of P and T must match.');
end
if TYPE==1
    if isrow(T), idx = T; else, idx = T'; end
    T = full(ind2vec(idx))';
end
end
