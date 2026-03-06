function logf(CFG, lvl, varargin)
if isfield(CFG,'debug_level') && CFG.debug_level >= lvl
    fprintf(varargin{:});
end
end
