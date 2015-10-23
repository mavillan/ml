function generating_datasets(dataname)
%generating datasets receive the daname as input and return a set of files
%with the training and testing sets. The original.data file and the shuffle.m file should be either in
%the same directory of this function, or in a directory from the MATLAB
%path.
ns=20; %number of sets that we want to generate.
rng('default'); %we reset the random seed
if strcmp(dataname,'cereales') 
    orig_name='cereales.data';
elseif strcmp(dataname,'credit')
    orig_name='credit.data';
else disp('Error: data resquested is not in this database');
end
for i=1:ns
    A=load(orig_name); %load the original dataset
    [f,c]=size(A); %we obtain the number of rows and columns of A.
    B=shuffle(A);  %shuffle data
    %we choose a name for training and testing files
    name_tr=strcat(dataname,'-tr-',num2str(i),'.data'); 
    name_ts=strcat(dataname,'-ts-',num2str(i),'.data');
    %we write these files
    dlmwrite(name_tr,B([1:round(f*.75)],:),' ');
    dlmwrite(name_ts,B([(round(f*.75)+1):f],:),' ');
end
