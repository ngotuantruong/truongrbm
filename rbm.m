function params = rbm(params,  data,opts)

m          = size(data.train_x, 1);
kk = randperm(m);
for l = 1 : m

            x1          = data.train_x(kk(l), :);
            y1          = data.train_y(kk(l), :);
            tmp1    =params.c + params.U*y1'+params.W*x1';            
            ph1=1./(1+exp(-tmp1));
            h1 = (double(ph1 > rand(size(tmp1))))';            
            Px2 = params.b' + h1 * params.W; 
            x2 = double(1./(1+exp(-Px2)) > rand(size(Px2)));
            tmp2 = exp(params.d'+h1*params.U);
            py2= tmp2/sum(tmp2);
            y2=mnrnd(1, py2);                     
            P =params.c + params.U*y2'+ params.W*x2';
            ph2= 1./(1+exp(-P));
            c1 = ph1 * x1;
            c2 = ph2 * x2;          
            u1=ph1 * y1;
            u2=ph2 * y2;
            params.W = params.W - opts.lambda  * (-c1 + c2);
            params.b = params.b - opts.lambda  * (-x1 + x2)';
            params.c = params.c - opts.lambda  * (-ph1 + ph2);
            params.d = params.d - opts.lambda  * (-y1 + y2)' ;
            params.U = params.U - opts.lambda  * (-u1 + u2);
    
end


end
