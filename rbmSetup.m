function params = rbmSetup(data, opts)

vSize = size(data.train_x, 2);
K = size(data.train_y, 2);
params   = struct;
mW=max(opts.numhidden,vSize);
a=-mW^(-0.5) ; b = mW^(-0.5);
params.W  = a + (b-a) * rand(opts.numhidden, vSize);
params.b  = zeros(vSize, 1);
params.c  = zeros(opts.numhidden, 1);
params.d  = zeros(K, 1);
mU=max(opts.numhidden,K);
a=-mU^(-0.5) ; b = mU^(-0.5);  
params.U  =  a + (b-a) * rand(opts.numhidden, K);

end
