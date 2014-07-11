function params= semi_hdrbm( params, data,opts)

     size_lab = size(data.train_x, 1);
     size_unlab = size(data.Dunlab, 1);
     m=size_lab+size_unlab;    
     % array for unlabel train data
     arr1=ones(1,size_lab);
     % array for unlabel train data
     arr0=zeros(1,size_unlab);
     % conjuction arr1 and arr0
     arr=[arr1 arr0];
     if opts.isshuffle==1
         arr_bool=arr(randperm(length(arr)));  
     else
         arr_bool=arr; 
     end

     
     kl = randperm(size_lab); 
     ku = randperm(size_unlab); 
     index_lab=0;
     index_unlab=0;
   
     for l = 1 : m      
%        disp(['example= ',num2str(l)])  
        if arr_bool(l)==1 % Label
            %% Train label HDRBM
            index_lab=index_lab+1;
            x1 = data.train_x(kl(index_lab), :);
            y1 = data.train_y(kl(index_lab), :);
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
    
           % RBM Discriminative
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
            
            params.W = params.W - opts.lambda * ( -pos*x1+s*x1 + opts.alpha*(-c1 + c2));
            params.c = params.c - opts.lambda * (-pos+s + opts.alpha*(-ph1 + ph2));
            params.d = params.d - opts.lambda * ((-y1+pyx)+ opts.alpha*(-y1 + y2))' ;
            params.b = params.b - opts.lambda * opts.alpha*(-x1 + x2)';
            params.U = params.U - opts.lambda * (-pos*y1+s1 + opts.alpha*(-u1 + u2));
          
             
                
        else % Unlabel
            %% Train Unlabel Unsupervised
            index_unlab=index_unlab+1;
            x1 = data.Dunlab(ku(index_unlab), :);
            tmp         = params.c+params.W*x1';
            emu         = bsxfun(@plus, params.U, tmp);
            % 3. Compute p(y|x) vector column [numclass x 1]
            vt1         = find(emu<10);          
            emu(vt1)    = log(1+exp(emu(vt1)));        
                
            mu          = sum(emu)+params.d';           
            maximum     = max(mu);
            if maximum < -745 || maximum > 707.4 || min(mu) < -745
                 mu     = mu + (707.4 - maximum);
            end
            e           = exp(mu);
            pyx         = e/sum(e);
            
            y1S         = mnrnd(1, pyx);
            tmp1        = tmp+params.U*y1S';            
            ph1         = 1./(1+exp(-tmp1));
            h1          = (double(ph1 > rand(size(tmp1))))';            
            Px2         = params.b' + h1 * params.W; 
            x2          = double(1./(1+exp(-Px2)) > rand(size(Px2)));
            tmp2        = exp(params.d'+h1*params.U);
            py2         = tmp2/sum(tmp2);
            y2          = mnrnd(1, py2);                     
            P           = params.c + params.U*y2'+ params.W*x2';
            ph2         = 1./(1+exp(-P));
            c1          = ph1 * x1;
            c2          = ph2 * x2;          
            u1          = ph1 * y1S;
            u2          = ph2 * y2;
            params.W    = params.W - opts.beta * opts.lambda  * (-c1 + c2);
            params.b    = params.b - opts.beta * opts.lambda  * (-x1 + x2)';
            params.c    = params.c - opts.beta * opts.lambda  * (-ph1 + ph2);
            params.d    = params.d - opts.beta * opts.lambda  * (-y1S + y2)' ;
            params.U    = params.U - opts.beta * opts.lambda  * (-u1 + u2);
            
        end
     end

end

