function H = elm_activation(tempH, TF)
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        if exist('hardlim','file')==2
            H = hardlim(tempH);
        else
            H = double(tempH >= 0);
        end
    otherwise
        error('ELM:Arguments','Unknown transfer function: %s', TF);
end
end
