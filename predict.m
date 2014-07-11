function pred = predict( params,data )
%PREDICT Summary of this function goes here
     m =size(data,1);
     pred=zeros(m,1);
     for i=1:m   
           tmp=params.c+params.W*data(i,:)';
           emu= bsxfun(@plus, params.U, tmp);
           vt1=find(emu<10);          
           emu(vt1)=log(1+exp(emu(vt1)));                      
           mu=sum(emu)+params.d';           
           maximum=max(mu);
           if maximum<-745 || maximum>707.4 || min(mu)<-745
                 mu=mu+(707.4-maximum);
           end
           e=exp(mu);
           p=e/sum(e);
          [v k]=max(p);
          pred(i)=k;
     end
end


