function params = drbm(params, data,opts)
tic;
        m = size(data.train_x, 1);
        kk = randperm(m); 
        
        for l = 1 : m
%            disp(['example= ',num2str(l)])            
            x1 = data.train_x(kk(l), :);
            y1=data.train_y(kk(l), :);   
            tmp=params.c+params.W*x1';
            emu= bsxfun(@plus, params.U, tmp);
            % 1. Compute p(h|y,x) matrix [numclass x Dh]
            ph=1./(1+exp(-emu));
            % 2. Compute p(h|y=k,x) vector row [1 x Dh]       
            pos=ph(:,find(y1==1,1));
            % 3. Compute p(y|x) vector column [numclass x 1]
            vt1=find(emu<10);          
            emu(vt1)=log(1+exp(emu(vt1)));        
                
            mu=sum(emu)+params.d';           
            maximum=max(mu);
            if maximum<-745 || maximum>707.4 || min(mu)<-745
                 mu=mu+(707.4-maximum);
            end
            e=exp(mu);
            pyx=e/sum(e);

            % 4. sigma( p(y'|x).p(hj'=1|x,y') )  vector row [1 x Dh]
         
            s1 = bsxfun(@times, ph, pyx);
            s=sum(s1,2);
                           
            % Update parameters
      
            params.W = params.W - opts.lambda * (-pos*x1+s*x1);
            params.c = params.c - opts.lambda * (-pos+s);
            params.d = params.d -  opts.lambda * (-y1+pyx)' ;
            params.U = params.U - opts.lambda * (-pos*y1+s1);
          
        end  
toc
end
