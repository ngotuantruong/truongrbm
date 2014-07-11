function labels = convert( inputdata )
    [r,c]=size(inputdata);
    labels=zeros(r,1);
    for i=1:r
        labels(i)=find(inputdata(i,:),1,'first');
    end
end

