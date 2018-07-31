clc
clearvars
close all

files=dir('*_parsed.txt');

bad_marks{1}='­';
bad_marks{1}='­';

allowed = {};
k=0;
for r1=0:9
    for r2=0:9
        k=k+1;
        allowed{k}=sprintf('%i,%i',r1,r2);
    end
end
allowed{end+1}='0,E';

for n=1:length(files)
    flag=0;
      
    fid=fopen(files(n).name,'rt','n','UTF-8');
    text='';
    while ~feof(fid)
        Line=fgetl(fid);
        Line = strtrim(Line);
        if length(text)>0        
            text=sprintf('%s\nPARAGRAPH_SEPARATOR\n%s',text,Line);
        else
            text=Line; 
        end
    end
    fclose(fid);
        
    ind = strfind(text,',');
    assert(ind(end)<length(text));
    assert(ind(1)>1);
    removed=0;
    N1=length(text);
    while ~isempty(ind)
        i=ind(end);
        if ~strcmp(text(i+1),' ')
            good=0;
            for k=1:length(allowed)
                if strcmp(text((i-1):(i+1)),allowed{k})
                    good=1;
                    fprintf('...  %s...\n',text(max(1,(i-30)):min(length(text),(i+30))));
                    text(i)=[];
                    fprintf('...  %s...\n',text(max(1,(i-30)):min(length(text),(i+30))));
                    removed=removed+1;
                    break;
                end
            end
            if good==0,
                fprintf('!!! File ''%s'': ...%s...\n',files(n).name,text(max(1,(i-50)):min(length(text),(i+50))));
                %error('Bad file found!')
                flag=1;
            end
        end
        ind(end)=[];
    end
    N2=length(text);
    assert(N2==N1-removed);       
    
    ind = strfind(text,' %');
    if ~isempty(ind)
       fprintf('!!! File ''%s'': ...%s...\n',files(n).name,text(max(1,(i-50)):min(length(text),(i+50))));
       error('trailing percentage')
    end
    
    if flag==1 || removed>0,
        fprintf('\n');
    end
    
    [a,b,c]=fileparts(files(n).name);
    fid=fopen([b,'_fixed',c],'wt','n','UTF-8');
    fprintf(fid,'%s',text);
    fclose(fid);    
    
end