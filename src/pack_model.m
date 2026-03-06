function model = pack_model(IW,B,LW,TF,TYPE,S_use,scale_vec,rls_info,sel_info,info_readout,CFG)
model = struct();
model.IW = IW; model.B = B; model.LW = LW;
model.TF = TF; model.TYPE = TYPE;
model.S_use = S_use;
model.scale_vec = scale_vec;
model.rls = rls_info;
model.select = sel_info;
model.readout = info_readout;
model.CFG = CFG;
end
