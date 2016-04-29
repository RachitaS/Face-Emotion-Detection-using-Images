%Open a file for writing output fisher vector in write mode
fid=fopen('fishereigen1000cluster30pcalib29apr1.txt','wt');

%Opening a file containing complete path of all images
fpng=fopen('pngfiles.txt','r');

%Opening a file containing labels of images
ftext=fopen('textfiles.txt','r');

%Number of features for reducing dimensionality by principal component analysis
features=40;

%Variable used for calculating fisher vector
numClusters=30;

%Gives count of number of images processed
counter=0;

while true
    
    %read lines of images file one by one
    counter=counter+1
    tline = fgetl(fpng)

    %break the loop at the end of the file
    if tline==-1
        break
    end

    %Read label of corresponding image file
    tlabel = fgetl(ftext);
    ftemp=fopen(tlabel);    
    line=fgetl(ftemp);
    fclose(ftemp);
    filename= tline;
    
    %reading image data using imread function
    I=imread(filename);

%if counter<295
%    continue
%else
%{    
if counter==257
    I=single(rgb2gray(I));
else if counter<295
 %       I=single(I);
   I=single(rgb2gray(I));
else
    I = single(rgb2gray(I)) ;
    end
end

%}
%end


if size(I,3)==3
    I=single(rgb2gray(I));
else
    I=single(I);
end

%[f,d] = vl_sift(I);
d=I;

%imagesc(d); axis image; colormap(gray);
meanmatrix=mean(d');
r=size(d,2); %1331
c1=size(meanmatrix',1) %128
meanT=meanmatrix';
[newmr,newmc]=size(meanT)
for i=1:r
    for k=1:c1
        d(k,i)=double(d(k,i))-meanT(k,1);
    end
end
imagesc(d); axis image; colormap(gray);
%{
[newr,newc]=size(d)
%imagesc(d); axis image; colormap(gray);
scattermatrix=double(d)'*double(d);
[V,D] = eig(scattermatrix);
D=diag(D);
D1=sort(D,'descend');
[s1,s2]=size(scattermatrix);
newdim=zeros(1,s1);
newdim=newdim';

for i=1:features
    maxeigval= V(:,D==D1(i));
    newdim=horzcat(newdim,maxeigval);
end

newdim(:,1)=[];
finaldata=double(d)*double(newdim);

[nr,nc]=size(newdim) %1331*200
[fr,fc]=size(finaldata) %128*200
[ri,ci]=size(d) %128*1331
imagesc(finaldata); axis image; colormap(gray);
%}
%{
meanmatrix=mean(d);
r=size(d,1);
c1=size(meanmatrix,2);

for i=1:r
    for k=1:c1
        d(i,k)=double(d(i,k))-meanmatrix(k);
    end
end

%imagesc(d); axis image; colormap(gray);
%}
x=princomp(double(d'));

[ri,ci]=size(I) %725*778
[ri,ci]=size(d) %128*1331
[rx,cx]=size(x) %1331*1331

%eigenfaces = reshape(x,[size(I,1),size(I,2)]);
%imagesc(x(:,1:features)); axis image; colormap(gray);

y=x(:,1:features);
[ry,cy]=size(y)
%eigenfaces = reshape(y,[128,725]);
%imagesc(y); axis image; colormap(gray);
%y=x;
finaldata=double(d')*double(y);
[r,c]=size(finaldata);
[fr,fc]=size(finaldata);
[rows,cols]=size(finaldata);
%eigenfaces = reshape(finaldata,[128,1331]);
%imagesc(eigenfaces); axis image; colormap(gray);

%}
[rows,cols]=size(finaldata)
x=reshape(finaldata,[1,rows*cols]);


[means, covariances, priors] = vl_gmm(x, numClusters);
encoding = vl_fisher(x, means, covariances, priors);


[r,c]=size(encoding);
label=strtrim(line);
str=single(label);
%str=num2str(uint8(single(label)));
%str=(int(str2num(label)));
%str=' '
for i=1:r
    str=strcat(str,',',num2str(i),':',num2str(encoding(i)));
%    str=strcat(str,',',num2str(encoding(i)));
end

fprintf(fid,'%s\n',str);

end


%{
[r,c]=size(x)
label=strtrim(line);
str=single(label);

for i=1:c
    str=strcat(str,',',num2str(i),':',num2str(x(i)));
end

fprintf(fid,'%s\n',str);

end

%}

fclose(fid);
fclose(fpng);
fclose(ftext);

