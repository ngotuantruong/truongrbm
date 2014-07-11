function model = train( typetrain, params, data, opts )
% Description: Train rbm with earlystopping = 15
    err         = zeros(1,200);
    patient     = opts.patience;
    model       = struct;
    epoch       = 0;
    bestEpoch   = 1;
    besterr     = 100;
    totalTime=tic;
    while(1)
         eTime      = tic;
         epoch   = epoch+1;        
         params  = typetrain(params,data,opts);
         eTime      = toc(eTime);
         pred    = predict(params,data.val_x);
         errcurr = 100 * mean(pred ~= data.val_labels);  
         disp(['Epoch/Patient = ',num2str(epoch),'/',num2str(patient),'val err = ',num2str(errcurr),' Time = ',num2str(eTime),'s']);
         if errcurr < besterr
             besterr    = errcurr;
             bestEpoch  = epoch;  
             patient    = opts.patience;        
             err(epoch) = errcurr;     
             paramsbest = params;
             
             %save ('BestParam' ,'paramsbest','epoch','errcurr','typetrain');
         
         else
             err(epoch) = errcurr; 
             if patient == 0
                 break;
             end
             patient    = patient-1;        
         end
        
    end
    totalTime=toc(totalTime);
    model.params     = paramsbest;
    model.err        = err;
    model.opts       = opts;
    model.totalEpoch = epoch;
    model.bestEpoch  = bestEpoch; 
    model.totalTime  = totalTime; 
    model.typetrain  = typetrain;
    
end

