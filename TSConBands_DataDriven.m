% ======================= Sieve Bootstrap, EU ETS ========================%
%  Replication codes for:
%   "Sieve Bootstrap Inference for Linear Time-Varying Coefficient Models"
%      by Friedrich and Lin (2022).
%  Please cite this paper for using the codes.
% 
%  This version: June 24, 2022
% 
%  @ Copyright: Marina Friedrich and Yicong Lin
%  @ Maintainer: Yicong Lin, yc.lin@vu.nl
% 
%  This program is free: you can redistribute it and/or modify it so long as
%  original authorship credit is given and you in no way impinge on its free
%  distribution. This program is distributed in the hope that it will be useful, 
%  but WITHOUT ANY WARRANTY.
% ========================================================================%


function [mbetahat,vhDataDriven,PW,ST] = TSConBands_DataDriven(vy,mX,kx,gridBW,siglevel,datatime,datavars)

warning('off','all')

B = 9999;                % number of bootstrap replicates
ChtildeBands = 2;
ChtildeTest = 2;
lList = [0 2 4 6];
GsetOptionTest = upper('FullSample');
siglevelTest = siglevel(1);
siglevelBands = siglevel(2);

fprintf(1,'\n\n')
disp('------------------------------------------------------------------------------')

%======= Data-driven bandwidth selections =======%
disp('Start with the data-driven bandwidth selection.')
fprintf(1,'\n')
disp('Please wait for about 20 secs.')
disp('......')
fprintf(1,'\n')
tic
mhListDataDriven = BandwidthSelection_OptLocal(vy,mX,kx,gridBW,lList);
toc
dlmwrite('mhListDataDriven.txt',mhListDataDriven,'precision',10);
vhDataDriven = mhListDataDriven(end,:);

% display results
hL = gridBW(1);
hU = gridBW(end);
fprintf(1,'\n')
disp(['The data-driven bandwidth in [',num2str(hL),', ',num2str(hU),'] by AIC is ', num2str(vhDataDriven(1))])
disp(['The data-driven bandwidth in [',num2str(hL),', ',num2str(hU),'] by GCV is ', num2str(vhDataDriven(2))])
for lid = 1:length(lList)
    disp(['The data-driven bandwidth in [',num2str(hL),', ',num2str(hU),'] by LMCV',...
        num2str(lList(lid)),' is ', num2str(vhDataDriven(2+lid))])
end
disp(['The data-driven bandwidth in [',num2str(hL),', ',num2str(hU),'] by AVG is ', num2str(vhDataDriven(end))])
fprintf(1,'\n')
% if  isempty(find(vhDataDriven==hL, 1))
%     disp('The data-driven bandwidth by AVG will be used later.')
%     hselected = vhDataDriven(end);
%     htildeTest = ChtildeTest*hselected^(5/9);
%     htildeBands = ChtildeBands*hselected^(5/9);
% else
%     disp('The data-driven bandwidth by LMCV6 will be used later.')
%     hselected = vhDataDriven(end-1);
%     htildeTest = ChtildeTest*hselected^(5/9);
%     htildeBands = ChtildeBands*hselected^(5/9);
% end
disp('The data-driven bandwidth by AVG will be used later.')
hselected = vhDataDriven(end);
htildeTest = ChtildeTest*hselected^(5/9);
htildeBands = ChtildeBands*hselected^(5/9);

%======= Parameter constancy test =======%
disp('------------------------------------------------------------------------------')
disp('Test for parameter constancy next.')
fprintf(1,'\n')
disp('The progress will be reported below.')
fprintf(1,'\n')
tic
mCriVals = SieveBootstrap_ParameterTests(B,vy,mX,hselected,htildeTest,kx,siglevelTest,GsetOptionTest);
toc
mCriVals = mCriVals{1};

vchat = mX\vy;
mbetahat = local_linear(vy,mX,hselected,kx,'GsetOption',GsetOptionTest);
WaldHat = (mbetahat - vchat').^2;
RejAlphaS = zeros(1,size(mX,2));
for idreg = 1:size(mX,2)
    Reji = (WaldHat(:,idreg) >= mCriVals(:,idreg));
    RejAlphaS(idreg) = (sum(Reji,1)>0);
end
Rej = (sum(RejAlphaS,2)>0);

fprintf(1,'\n')
if Rej
    disp(['Given α = ',num2str(siglevelTest),' and h = ',num2str(hselected),', parameter constancy is rejected.'])
else
    disp(['Given α = ',num2str(siglevelTest),' and h = ',num2str(hselected),', parameter constancy is not rejected.'])
end


%======= Local linear estimation and confidence bands =======%
disp('------------------------------------------------------------------------------')
disp('Estimate the coefficients and construct simultaneous confidence bands now.')
fprintf(1,'\n')
disp('The progress will be reported below.')
fprintf(1,'\n')
disp('A soft fire makes sweet malt. It may take some time.')
disp('......')
tic
[mbetahat,betaPWInt,betaSTIntFullSpan] = SieveBootstrap(B,vy,mX,hselected,htildeBands,kx,siglevelBands);
PW = betaPWInt{1,1};                       % SB: equal-tailed
ST = squeeze(betaSTIntFullSpan(:,1,:));    % SB: equal-tailed
ST_Gsub = ST(:,1);        
ST_G = ST(:,2); 
ST_Full = ST(:,3); 
toc

dlmwrite('BetaEstimates.txt',mbetahat,'precision',10);
dlmwrite('PWInts.txt',PW,'precision',10);
dlmwrite('BandsGsub.txt',ST_Gsub,'precision',10);
dlmwrite('BandsG.txt',ST_G,'precision',10);
dlmwrite('BandsFull.txt',ST_Full,'precision',10);


%======= Make plots =======%
MakeGraphs(mbetahat,PW,ST,datatime,datavars);

fprintf(1,'\n')
disp('Finished!')
disp('The graphs and results have been saved in your source folder.')
disp('------------------------------------------------------------------------------')

end


%########## Local Linear Estimation ##########%
function mbetahat = local_linear(vy,mX,h,kx,varargin)

warning('off','all')

defaultGsetOption = upper('FullSample');
expectedGsetOption = {upper('AllGSubset'),upper('AllGset'),upper('FullSample'),upper('Subsample'),upper('FullUnion')};
defaultSmoothOption = upper('Regular');
expectedSmoothOption = {upper('Regular'),upper('Oversmooth')};
OptionalInputs = inputParser;
addParameter(OptionalInputs,'GsetOption',defaultGsetOption,@(x) any(validatestring(upper(x),expectedGsetOption))); 
addParameter(OptionalInputs,'SmoothOption',defaultSmoothOption,@(x) any(validatestring(upper(x),expectedSmoothOption))); 
addOptional(OptionalInputs,'OversmoothBW',h,@(x) isnumeric(x) && isscalar(x) && (x > 0)); 
parse(OptionalInputs,varargin{:});
GsetOption = upper(OptionalInputs.Results.GsetOption);
SmoothOption = upper(OptionalInputs.Results.SmoothOption);

switch SmoothOption
    case upper('Regular')
        htilde = h;
    case upper('Oversmooth')
        htilde = OptionalInputs.Results.OversmoothBW;
end

[n, dReg] = size(mX);
AllGsets = generateUniformSet(h,n);
Gsub = AllGsets{1};
Guni = AllGsets{2};
Gall = AllGsets{3};

switch GsetOption
    case upper('FullSample')
        Gset = Gall;
        selectFun = 1;
    case upper('AllGSubset')
        Gset = {Gsub,Guni};
        selectFun = 2;
    case upper('AllGset')
        Gset = AllGsets;
        selectFun = 2;
    case upper('Subsample')
        Gset = Gsub;
        selectFun = 1;
    case upper('FullUnion')
        Gset = Guni;
        selectFun = 1;
end

trend = (1:n)'./n;
if selectFun == 1
    GsetNum = length(Gset); 
    mbetahat = zeros(dReg,GsetNum);
    for j = 1:GsetNum
        tau = Gset(j);  
        vKernelTauRoot = sqrt(kx((trend-tau)/htilde));
        mZtauTilde = vKernelTauRoot.*[mX mX.*(trend-tau)/htilde];
        vytauTilde = vKernelTauRoot.*vy;
        vthetahattau = mZtauTilde\vytauTilde;
        mbetahat(:,j) = vthetahattau(1:dReg);              
    end
    mbetahat = mbetahat';
    
else
    mbetahat = cell(1,length(Gset));    
    for Gid = 1:length(Gset)   
        Gseti = Gset{Gid};
        GsetNum = length(Gseti); 
        mbetahatGid = zeros(dReg,GsetNum);
        for j = 1:GsetNum
            tau = Gseti(j);  
            vKernelTauRoot = sqrt(kx((trend-tau)/htilde));
            mZtauTilde = vKernelTauRoot.*[mX mX.*(trend-tau)/htilde];
            vytauTilde = vKernelTauRoot.*vy;
            vthetahattau = mZtauTilde\vytauTilde;
            mbetahatGid(:,j) = vthetahattau(1:dReg);
        end
        mbetahat{Gid} = mbetahatGid';
    end
  
end

end


%########## Unions of G Subsets ##########%
function AllGsets = generateUniformSet(h,n)
 
   Uh = zeros(4,ceil(200*h)+1);
   for j = 1:4       
       Uh(j,:) = (j/5)-h+(0:ceil(200*h))/100;      
   end
   Gsub = unique([Uh(1,:) Uh(4,:)]');
   Gsub = Gsub((Gsub<=1)&(Gsub>=1/n));
   Guni = unique(Uh(:));
   Guni = Guni((Guni<=1)&(Guni>=1/n));   
   Gall = (1:n)'/n;
   AllGsets = {Gsub,Guni,Gall};  
 
end


%########## AIC: Select Lag Length ##########%
function k = aic(u)

    n = length(u);
    kmax = floor(10*log10(n)); 
    aic = zeros(kmax,1);
    ulag = lagmatrix(u,0:kmax);
    ulag(any(isnan(ulag),2),:) = [];
    y = ulag(:,1);
    for j = 2:kmax+1
        x = ulag(:,2:j);
        b = x\y; 
        e = y-x*b; 
        s2 = e'*e/length(e); 
        aic(j-1) = log(s2) + 2*size(x,2)/length(e);
    end    
    aic = [log(mean(y.^2)); aic];
    [~,k] = min(aic);
    k = k-1;  
    
end  


%########## Sieve (Wild) Bootstrap ##########%
function [mbetahat,betaPWInt,betaSTIntFullSpan,betaSTIntGset] = SieveBootstrap(B,vy,mX,h,htilde,kx,siglevel)

[n,dReg] = size(mX);

%== Parameter specifications ==%
burnin = 20;
AllGsets = generateUniformSet(h,n);

%== Preliminary estimation for constructing intervals ==%
mbetahatAllGset = local_linear(vy,mX,h,kx,'GsetOption','AllGset');
mbetahat = mbetahatAllGset{3};

%== Dynamics of residuals ==%

% Estimation under oversmoothing
mbetahathtildeAllGset = local_linear(vy,mX,h,kx,'GsetOption','AllGset',...
    'SmoothOption','Oversmooth','OversmoothBW',htilde);

mbetatilde = mbetahathtildeAllGset{3};    % n-dimensional
vyhat = sum(mX.*mbetatilde,2);

% Calculate residuals
vresi = vy - vyhat;

% Bootstrap residuals
popt = aic(vresi);                        % select optimal lags
vresilag = lagmatrix(vresi,0:popt);
vresilag(any(isnan(vresilag),2),:) = [];
if popt>0
    x = vresilag(:,2:1+popt);
    y = vresilag(:,1);
    b = x\y;
    vepsi = y-x*b; 
    vepsiTilde = vepsi - mean(vepsi);
else
    vepsi = vresi;
    vepsiTilde = vepsi - mean(vepsi);
end

%== Bootstrap ==%
GsetSize = cellfun(@length,AllGsets,'UniformOutput',false);
GsetSizeBoot = num2cell([GsetSize{:};[B;dReg]*ones(1,length(AllGsets))]',2);
mbetaHatSieveBootG = cellfun(@zeros,GsetSizeBoot,'UniformOutput',false);
mbetaHatSieveWildBootG = cellfun(@zeros,GsetSizeBoot,'UniformOutput',false);
for Biter = 1:B
    
    % Report progress
    if (mod(Biter,1000)) == 0
        fprintf('Confidence bands: Bootstrap iteration %5d out of %5d \n',Biter,B);
    end
    
    % Generate samples
    if popt>0       
        vepsiSieveBoot = datasample(vepsiTilde,n+burnin);
        vzSieveBoot = zeros(n+burnin,1);
        vzSieveBoot(1:popt) = vepsiSieveBoot(1:popt);         % Sieve bootstrap
        
        vepsiSieveWildBoot = [datasample(vepsiTilde,popt+burnin); normrnd(0,1,[length(vepsi),1]).*vepsi];
        vzSieveWildBoot = zeros(n+burnin,1);
        vzSieveWildBoot(1:popt) = vepsiSieveWildBoot(1:popt); % Sieve wild bootstrap
        
        for j = popt+1:n+burnin         
            vzSieveBoot(j) = vzSieveBoot(j-1:-1:j-popt)'*b + vepsiSieveBoot(j);               % Sieve bootstrap
            vzSieveWildBoot(j) = vzSieveWildBoot(j-1:-1:j-popt)'*b + vepsiSieveWildBoot(j);   % Sieve wild bootstrap 
        end
        vzSieveBoot = vzSieveBoot(end-n+1:end);           % Sieve bootstrap
        vzSieveWildBoot = vzSieveWildBoot(end-n+1:end);   % Sieve wild bootstrap 
            
    else
        vzSieveBoot = datasample(vepsiTilde,n);           % Sieve bootstrap
        vzSieveWildBoot = normrnd(0,1,[n,1]).*vepsi;      % Sieve wild bootstrap 
    end
    vySieveBoot = vyhat + vzSieveBoot;           % Sieve bootstrap
    vySieveWildBoot = vyhat + vzSieveWildBoot;   % Sieve wild bootstrap
       
    mbetahatstarSieveBootAllGset = local_linear(vySieveBoot,mX,h,kx,'GsetOption','AllGset');          % Sieve bootstrap
    mbetahatstarSieveWildBootAllGset = local_linear(vySieveWildBoot,mX,h,kx,'GsetOption','AllGset');  % Sieve wild bootstrap
    
    for Gid = 1:length(mbetahatstarSieveBootAllGset)
        for regid = 1:dReg
            mbetaHatSieveBootG{Gid}(:,Biter,regid) = mbetahatstarSieveBootAllGset{Gid}(:,regid); 
            mbetaHatSieveWildBootG{Gid}(:,Biter,regid) = mbetahatstarSieveWildBootAllGset{Gid}(:,regid);    
        end
    end
      
end

%========= Pointwise Intervals =========%
% 1. Sieve Bootstrap: equal-tailed (ET) | symmetric (SY)
% 2. Sieve Wild Bootstrap: equal-tailed (ET) | symmetric (SY)
betaPWIntSieveBootET = zeros(n,2,dReg);
betaPWIntSieveBootSY = zeros(n,2,dReg);
betaPWIntSieveWildBootET = zeros(n,2,dReg);
betaPWIntSieveWildBootSY = zeros(n,2,dReg);
for regid = 1:dReg
    mbetaHatSieveBootid = mbetaHatSieveBootG{3}(:,:,regid);
    mbetaHatSieveWildBootid = mbetaHatSieveWildBootG{3}(:,:,regid);
    
    %== Sieve Bootstrap ==%
    % (1). Equal-tailed (ET)
    betaPWSieveBootET = quantile(mbetaHatSieveBootid-mbetatilde(:,regid),[siglevel/2 1-siglevel/2],2);
    betaPWIntSieveBootET(:,:,regid) = [mbetahat(:,regid) - betaPWSieveBootET(:,2) mbetahat(:,regid) - betaPWSieveBootET(:,1)];
    
    % (2). Symmetric (SY)
    betaPWSieveBootSY = quantile(abs(mbetaHatSieveBootid-mbetatilde(:,regid)),1-siglevel,2);
    betaPWIntSieveBootSY(:,:,regid) = mbetahat(:,regid) + [-betaPWSieveBootSY betaPWSieveBootSY];
    
    %== Sieve Wild Bootstrap ==%
    % (1). Equal-tailed (ET)
    betaPWSieveWildBootET = quantile(mbetaHatSieveWildBootid-mbetatilde(:,regid),[siglevel/2 1-siglevel/2],2);
    betaPWIntSieveWildBootET(:,:,regid) = [mbetahat(:,regid) - betaPWSieveWildBootET(:,2) mbetahat(:,regid) - betaPWSieveWildBootET(:,1)];
    
    % (2). Symmetric (SY)
    betaPWSieveWildBootSY = quantile(abs(mbetaHatSieveWildBootid-mbetatilde(:,regid)),1-siglevel,2);
    betaPWIntSieveWildBootSY(:,:,regid) = mbetahat(:,regid) + [-betaPWSieveWildBootSY betaPWSieveWildBootSY];
    
end
betaPWInt = {betaPWIntSieveBootET,betaPWIntSieveBootSY;
             betaPWIntSieveWildBootET,betaPWIntSieveWildBootSY};


%========= Simultaneous Intervals =========%
alphaPmax = siglevel;
alphaPmin = 1/1000;
TheoCov = 1-siglevel;

%- layers: based on G sets -%
%      ET     |     SY      | 
%     beta1   |    beta1    |
%     beta2   |    beta2    |
%     .....   |    .....    |
%     betad   |    betad    |
%---------------------------%

betaSTIntSBFullSpan = cell(dReg,2,length(AllGsets));    % over full time-span t = 1,2,...,T, for easy plots
betaSTIntSBGset = cell(dReg,2,length(AllGsets));        % over G, for easy evaluation of size and length
betaSTIntSWBFullSpan = cell(dReg,2,length(AllGsets));   % over full time-span t = 1,2,...,T, for easy plots
betaSTIntSWBGset = cell(dReg,2,length(AllGsets));       % over G, for easy evaluation of size and length
for Gid = 1:length(AllGsets)
    for regid = 1:dReg
        % Estimation under oversmoothing
        mbetahathtildeGiRegid = mbetahathtildeAllGset{Gid}(:,regid);
        
        % Bootstrap estimates
        mbetaHatSieveBootSTGiRegid = mbetaHatSieveBootG{Gid}(:,:,regid);
        mbetaHatSieveWildBootSTGiRegid = mbetaHatSieveWildBootG{Gid}(:,:,regid);
        
        % Differences
        betaDiffSieveBootGiRegid = mbetaHatSieveBootSTGiRegid-mbetahathtildeGiRegid;
        betaDiffSieveWildBootGiRegid = mbetaHatSieveWildBootSTGiRegid-mbetahathtildeGiRegid;
        
        %=====================================================================================%
        %== 1. Equal-tailed (ET) constructions ==%
        betaSieveBootST_ET_GiRegidMin = quantile(betaDiffSieveBootGiRegid,[alphaPmin/2 1-alphaPmin/2],2); 
        betaSieveWildBootST_ET_GiRegidMin = quantile(betaDiffSieveWildBootGiRegid,[alphaPmin/2 1-alphaPmin/2],2); 
        
        %-------------------------------------------------------------------------------------%
        %= (1) Sieve Bootstrap =%
        % Simultaneous coverage (sum over columns, covering for all tau counts as a success)
        betaSBMinCov_ET = mean((betaDiffSieveBootGiRegid <= betaSieveBootST_ET_GiRegidMin(:,2)).* (betaDiffSieveBootGiRegid >= betaSieveBootST_ET_GiRegidMin(:,1)),1);
        betaSBMinCov_ET = 1- mean(betaSBMinCov_ET<1);
        if betaSBMinCov_ET >= 1-siglevel
            betaCov1 = betaSBMinCov_ET;
            alphaS_SB_ET = alphaPmin + 1/B;
            while alphaS_SB_ET <= alphaPmax 
                betaQuanalphaS = quantile(betaDiffSieveBootGiRegid,[alphaS_SB_ET/2 1-alphaS_SB_ET/2],2);
                betaCov2 = mean((betaDiffSieveBootGiRegid <= betaQuanalphaS(:,2)).* (betaDiffSieveBootGiRegid >= betaQuanalphaS(:,1)),1);
                betaCov2 = 1- mean(betaCov2<1);
                if abs(betaCov2 - TheoCov) <= abs(betaCov1 - TheoCov)
                    alphaS_SB_ET = alphaS_SB_ET + 1/B;
                    betaCov1 = betaCov2;
                else
                    break
                end
            end
            alphaS_SB_ET = alphaS_SB_ET - 1/B;
        else
            alphaS_SB_ET = alphaPmin;
        end
        % over full time span t = 1,...,T
        betaSB_FullSpan_ET = quantile(mbetaHatSieveBootG{3}(:,:,regid)-mbetatilde(:,regid),[alphaS_SB_ET/2 1-alphaS_SB_ET/2],2);
        betaSTIntSBFullSpan{regid,1,Gid} = [mbetahat(:,regid) - betaSB_FullSpan_ET(:,2) mbetahat(:,regid) - betaSB_FullSpan_ET(:,1)];
        
        % over G only
        betaSB_G_ET = quantile(betaDiffSieveBootGiRegid,[alphaS_SB_ET/2 1-alphaS_SB_ET/2],2);
        betaSTIntSBGset{regid,1,Gid} = [mbetahatAllGset{Gid}(:,regid) - betaSB_G_ET(:,2) mbetahatAllGset{Gid}(:,regid) - betaSB_G_ET(:,1)]; 
        
        %-------------------------------------------------------------------------------------%
        %= (2) Sieve Wild Bootstrap =%
        % Simultaneous coverage (sum over columns, covering for all tau counts as a success)
        betaSWBMinCov_ET = mean((betaDiffSieveWildBootGiRegid <= betaSieveWildBootST_ET_GiRegidMin(:,2)).* (betaDiffSieveWildBootGiRegid >= betaSieveWildBootST_ET_GiRegidMin(:,1)),1);
        betaSWBMinCov_ET = 1- mean(betaSWBMinCov_ET<1);
        if betaSWBMinCov_ET >= 1-siglevel
            betaCov1 = betaSWBMinCov_ET;
            alphaS_SWB_ET = alphaPmin + 1/B;
            while alphaS_SWB_ET <= alphaPmax 
                betaQuanalphaS = quantile(betaDiffSieveWildBootGiRegid,[alphaS_SWB_ET/2 1-alphaS_SWB_ET/2],2);
                betaCov2 = mean((betaDiffSieveWildBootGiRegid <= betaQuanalphaS(:,2)).* (betaDiffSieveWildBootGiRegid >= betaQuanalphaS(:,1)),1);
                betaCov2 = 1- mean(betaCov2<1);
                if abs(betaCov2 - TheoCov) <= abs(betaCov1 - TheoCov)
                    alphaS_SWB_ET = alphaS_SWB_ET + 1/B;
                    betaCov1 = betaCov2;
                else
                    break
                end
            end
            alphaS_SWB_ET = alphaS_SWB_ET - 1/B;
        else
            alphaS_SWB_ET = alphaPmin;
        end
        % over full time span t = 1,...,T
        betaSWB_FullSpan_ET = quantile(mbetaHatSieveWildBootG{3}(:,:,regid)-mbetatilde(:,regid),[alphaS_SWB_ET/2 1-alphaS_SWB_ET/2],2);
        betaSTIntSWBFullSpan{regid,1,Gid} = [mbetahat(:,regid) - betaSWB_FullSpan_ET(:,2) mbetahat(:,regid) - betaSWB_FullSpan_ET(:,1)];
        
        % over G only
        betaSWB_G_ET = quantile(betaDiffSieveWildBootGiRegid,[alphaS_SWB_ET/2 1-alphaS_SWB_ET/2],2);
        betaSTIntSWBGset{regid,1,Gid} = [mbetahatAllGset{Gid}(:,regid) - betaSWB_G_ET(:,2) mbetahatAllGset{Gid}(:,regid) - betaSWB_G_ET(:,1)];
        
        %-------------------------------------------------------------------------------------%
        
        
        %=====================================================================================%     
        %== 2. Symmetric (SY) constructions ==%
        
        %-------------------------------------------------------------------------------------%
        %= (1) Sieve Bootstrap =%
        % Simultaneous coverage (sum over columns, covering for all tau counts as a success)
        betaSieveBootST_SY_GiRegidMin = quantile(abs(betaDiffSieveBootGiRegid),1-alphaPmin,2); 
        betaSieveWildBootST_SY_GiRegidMin = quantile(abs(betaDiffSieveWildBootGiRegid),1-alphaPmin,2); 
        
        %-------------------------------------------------------------------------------------%
        %= (1) Sieve Bootstrap =%
        % Simultaneous coverage (sum over columns, covering for all tau counts as a success)
        betaSBMinCov_SY = mean(abs(betaDiffSieveBootGiRegid)<=betaSieveBootST_SY_GiRegidMin,1);
        betaSBMinCov_SY = 1- mean(betaSBMinCov_SY<1);
        if betaSBMinCov_SY >= 1-siglevel
            betaCov1 = betaSBMinCov_SY;
            alphaS_SB_SY = alphaPmin + 1/B;
            while alphaS_SB_SY <= alphaPmax 
                betaQuanalphaS = quantile(abs(betaDiffSieveBootGiRegid),1-alphaS_SB_SY,2);
                betaCov2 = mean(abs(betaDiffSieveBootGiRegid)<=betaQuanalphaS,1);
                betaCov2 = 1- mean(betaCov2<1);
                if abs(betaCov2 - TheoCov) <= abs(betaCov1 - TheoCov)
                    alphaS_SB_SY = alphaS_SB_SY + 1/B;
                    betaCov1 = betaCov2;
                else
                    break
                end
            end
            alphaS_SB_SY = alphaS_SB_SY - 1/B;
        else
            alphaS_SB_SY = alphaPmin;
        end
        % over full time span t = 1,...,T
        betaSB_FullSpan_SY = quantile(abs(mbetaHatSieveBootG{3}(:,:,regid)-mbetatilde(:,regid)),1-alphaS_SB_SY,2);
        betaSTIntSBFullSpan{regid,2,Gid} = mbetahat(:,regid) + [-betaSB_FullSpan_SY betaSB_FullSpan_SY];
        
        % over G only
        betaSB_G_SY = quantile(abs(betaDiffSieveBootGiRegid),1-alphaS_SB_SY,2);
        betaSTIntSBGset{regid,2,Gid} = mbetahatAllGset{Gid}(:,regid) + [-betaSB_G_SY betaSB_G_SY]; 
        
        %-------------------------------------------------------------------------------------%
        %= (2) Sieve Wild Bootstrap =%
        % Simultaneous coverage (sum over columns, covering for all tau counts as a success)
        betaSWBMinCov_SY = mean(abs(betaDiffSieveWildBootGiRegid)<=betaSieveWildBootST_SY_GiRegidMin,1);
        betaSWBMinCov_SY = 1- mean(betaSWBMinCov_SY<1);
        if betaSWBMinCov_SY >= 1-siglevel
            betaCov1 = betaSWBMinCov_SY;
            alphaS_SWB_SY = alphaPmin + 1/B;
            while alphaS_SWB_SY <= alphaPmax 
                betaQuanalphaS = quantile(abs(betaDiffSieveWildBootGiRegid),1-alphaS_SWB_SY,2);
                betaCov2 = mean(abs(betaDiffSieveWildBootGiRegid)<=betaQuanalphaS,1);
                betaCov2 = 1- mean(betaCov2<1);
                if abs(betaCov2 - TheoCov) <= abs(betaCov1 - TheoCov)
                    alphaS_SWB_SY = alphaS_SWB_SY + 1/B;
                    betaCov1 = betaCov2;
                else
                    break
                end
            end
            alphaS_SWB_SY = alphaS_SWB_SY - 1/B;
        else
            alphaS_SWB_SY = alphaPmin;
        end
        % over full time span t = 1,...,T
        betaSWB_FullSpan_SY = quantile(abs(mbetaHatSieveWildBootG{3}(:,:,regid)-mbetatilde(:,regid)),1-alphaS_SWB_SY,2);
        betaSTIntSWBFullSpan{regid,2,Gid} = mbetahat(:,regid) + [-betaSWB_FullSpan_SY betaSWB_FullSpan_SY];
        
        % over G only
        betaSWB_G_SY = quantile(abs(betaDiffSieveWildBootGiRegid),1-alphaS_SWB_SY,2);
        betaSTIntSWBGset{regid,2,Gid} = mbetahatAllGset{Gid}(:,regid) + [-betaSWB_G_SY betaSWB_G_SY];
                
    end

end

% For each layer = 1, 2, 3, the left half part is for sieve bootstrap,
% while the right half part is for sieve wild bootstrap. The first column 
% in each part is for equal-tailed constructions. while the second column 
% in each part is for symmetric constructions.
betaSTIntFullSpan = [betaSTIntSBFullSpan betaSTIntSWBFullSpan];
betaSTIntGset = [betaSTIntSBGset betaSTIntSWBGset];

end


function [mhListDataDriven, mLoss] = BandwidthSelection_OptLocal(vy,mX,kx,gridBW,lList)

warning('off','all')

n = length(vy);

% parameter specifications
NlList = length(lList);
NSelectMethods = 2+NlList;

mLoss = zeros(length(gridBW),2);                % Optimal Global: AICmod | GCV 
mLossOptLocal = zeros(n,NlList,length(gridBW)); % Optimal Local: CV/LMCV(l=0) | LMCV(l=2) | LMCV(l=4) | LMCV(l=6)
for id = 1:length(gridBW)
    
    hid = gridBW(id);
    
    %====== Modified AIC & Generalized CV ======%
    mLoss(id,1:2) = AIC_GCV_Loss(vy,mX,kx,hid);
    
    %====== Modified CV: leave-(2l+1)-out  ======%    
    mLossOptLocal(:,:,id) = ModCVLoss_OptLocal(vy,mX,kx,lList,hid);
    
end

% Vary upper bounds: 0.2, hU
cutoff = 0.2:0.02:gridBW(end);
mhListDataDriven = zeros(length(cutoff),NSelectMethods+1);
for idcut = 1:length(cutoff)
    search_range = 1:find(abs(gridBW-cutoff(idcut))<1e-8);
    
    [~,optloc] = min(mLoss(search_range,:),[],1);
    mhListDataDriven(idcut,1:2) = gridBW(optloc);   
    
    mhListOptLocal = zeros(n,NlList);
    for lid = 1:NlList  
        for j = 1:n
            [~,optloc2] = min(squeeze(mLossOptLocal(j,lid,search_range))); 
            mhListOptLocal(j,lid) = gridBW(optloc2);
        end
    end
    mhListDataDriven(idcut,3:NSelectMethods) = min(mhListOptLocal,[],1);
    
    mhListDataDriven(idcut,NSelectMethods+1) = mean(mhListDataDriven(idcut,1:NSelectMethods));
end

end


function vLoss_AICmod_GCV = AIC_GCV_Loss(vy,mX,kx,h)

[n, dReg] = size(mX);
trend = (1:n)'./n;

AICmodx = @(s2,traceh,n) log(s2) + 2*(traceh+1)/(n-traceh-2);
GCVx = @(s2,traceh,n) s2/(1-traceh/n)^2;
ex = @(n,k) [zeros(k-1,1);1;zeros(n-k,1)];

vyhat = zeros(n,1);
traceh = 0;
for j = 1:n 
    tau = j/n;
    vKernelTauRoot = sqrt(kx((trend-tau)/h));
    mZtauh = vKernelTauRoot.*[mX mX.*(trend-tau)/h];
    vytauh = vKernelTauRoot.*vy;    
    vbetahath = mZtauh\vytauh;
    vyhat(j) = mX(j,:)*vbetahath(1:dReg);
    
    vHatMatSelect = mZtauh\(vKernelTauRoot.*ex(n,j));
    traceh = traceh + mX(j,:)*vHatMatSelect(1:dReg);
end

s2 = mean((vy-vyhat).^2);
vLoss_AICmod_GCV = [AICmodx(s2,traceh,n) GCVx(s2,traceh,n)];

end


function mLoss_ModCV_OptLocal = ModCVLoss_OptLocal(vy,mX,kx,lList,h)

[n, dReg] = size(mX);
trend = (1:n)'./n;
NlList = length(lList);

wx = @(x) normpdf(x,trend,sqrt(0.025));          % other variance: 0.0125
mResi2 = zeros(n,NlList); 
for j = 1:n 
    tau = j/n;
    vKernelTauRoot = sqrt(kx((trend-tau)/h));
    mZtauh = vKernelTauRoot.*[mX mX.*(trend-tau)/h];
    vytauh = vKernelTauRoot.*vy;  
    for lid = 1:NlList       
        l = lList(lid);
        id_leavelout = j-l:j+l;
        id_leavelout = id_leavelout(id_leavelout>=1 & id_leavelout<=n);
        mZtauh_leavelout = mZtauh;
        mZtauh_leavelout(id_leavelout,:) = [];
        vytauh_leavelout = vytauh;
        vytauh_leavelout(id_leavelout,:) = [];
        vthetahattau_leavelout = mZtauh_leavelout\vytauh_leavelout;
        mbetahat_leavelout = vthetahattau_leavelout(1:dReg)';     
        mResi2(j,lid) = (vy(j) - sum(mX(j,:).*mbetahat_leavelout,2))^2;
    end         
end

mLoss_ModCV_OptLocal = zeros(n,NlList); 
for j = 1:n
    tau = j/n;
    vwtau = wx(tau);
    mLoss_ModCV_OptLocal(j,:) = sum(mResi2.*vwtau,1)/n;
end

end

%########## Sieve Bootstrap ##########%
function mCriVals = SieveBootstrap_ParameterTests(B,mY,mX,h,htilde,kx,siglevel,GsetOption)

rng(123456)

[n,dReg] = size(mX);
NSeries = size(mY,2);

AllGsets = generateUniformSet(h,n);
switch upper(GsetOption)
    case upper('FullSample')
        Gset = AllGsets{3};
    case upper('Subsample')
        Gset = AllGsets{1};
    case upper('FullUnion')
        Gset = AllGsets{2};
end        
NGset = length(Gset);

%== Parameter specifications ==%
burnin = 20;

%== Dynamics of residuals ==%
mYHat = cell(1,NSeries);           % C | LS | RS | LL | RL | Smooth
mbetaTilde = cell(1,NSeries);
mbetaTildeGset = cell(1,NSeries);
mepsiTilde = cell(1,NSeries);
vpopt = zeros(1,NSeries);
mARlag = cell(1,NSeries);
for idseries = 1:NSeries
    vyObs = mY(:,idseries);
    [yhat,betatilde,betatildeGset,epsitilde,p_est,ARlag_est] = ResiDynamics(vyObs,mX,h,htilde,kx,GsetOption);
    mYHat{idseries} = yhat;
    mbetaTilde{idseries} = betatilde;
    mbetaTildeGset{idseries} = betatildeGset;
    mepsiTilde{idseries} = epsitilde;
    vpopt(idseries) = p_est;
    mARlag{idseries} = ARlag_est;
end

WaldHatBootAll = zeros(NGset,dReg,B,NSeries);
for Biter = 1:B
    
    % Report progress
    if (mod(Biter,1000)) == 0
        fprintf('Test for constancy: Bootstrap iteration %5d out of %5d \n',Biter,B);
    end
    
    for idseries = 1:NSeries
       popt = vpopt(idseries);
       vepsiTilde = mepsiTilde{idseries};
       vyhat = mYHat{idseries};
       betaGset = mbetaTildeGset{idseries};
       
       % Generate samples
       if popt>0     
           b =  mARlag{idseries};
           vepsiSieveBoot = datasample(vepsiTilde,n+burnin);
           vzSieveBoot = zeros(n+burnin,1);
           vzSieveBoot(1:popt) = vepsiSieveBoot(1:popt);            
           for j = popt+1:n+burnin  
               vzSieveBoot(j) = vzSieveBoot(j-1:-1:j-popt)'*b + vepsiSieveBoot(j);  
           end
           vzSieveBoot = vzSieveBoot(end-n+1:end);           
       else
           vzSieveBoot = datasample(vepsiTilde,n);           
       end
       vySieveBoot = vyhat + vzSieveBoot;
       
       % Construct Wald tests
       mbetahatBoot = local_linear(vySieveBoot,mX,h,kx,'GsetOption',GsetOption);
       WaldHatBootAll(:,:,Biter,idseries) = (mbetahatBoot - betaGset).^2;    
    end
end

alphaPmax = siglevel;
alphaPmin = 1/1000;
mCriVals = cell(1,NSeries);
for idseries = 1:NSeries
    WaldHatBoot = WaldHatBootAll(:,:,:,idseries);
    
    RejAlphaPmin = zeros(dReg,B);
    for idreg = 1:dReg
        WaldHatBooti = squeeze(WaldHatBoot(:,idreg,:));
        qhatAlphaPmini = quantile(WaldHatBooti,1-alphaPmin,2);      
        Reji = (WaldHatBooti >= qhatAlphaPmini);
        RejAlphaPmin(idreg,:) = (sum(Reji,1)>0);
    end
    RejRatioAlphaPmin = mean((sum(RejAlphaPmin,1)>0));

    if RejRatioAlphaPmin <= siglevel
        RejRatio1 = RejRatioAlphaPmin;
        alphaS = alphaPmin + 1/B;
        while alphaS <= alphaPmax 
            RejAlphaS = zeros(dReg,B);
            for idreg = 1:dReg
                WaldHatBooti = squeeze(WaldHatBoot(:,idreg,:));
                qhatAlphaSi = quantile(WaldHatBooti,1-alphaS,2);
                Reji = (WaldHatBooti >= qhatAlphaSi);
                RejAlphaS(idreg,:) = (sum(Reji,1)>0);
            end
            RejRatio2 = mean((sum(RejAlphaS,1)>0));
            if abs(RejRatio2 - siglevel) <= abs(RejRatio1 - siglevel)
                alphaS = alphaS + 1/B;
                RejRatio1 = RejRatio2;
            else
                break
            end
        end
        alphaS = alphaS - 1/B;
    else
        alphaS = alphaPmin;
    end
    
    qhatAlphaS = zeros(NGset,dReg);
    for idreg = 1:dReg
        WaldHatBooti = squeeze(WaldHatBoot(:,idreg,:));
        qhatAlphaS(:,idreg) = quantile(WaldHatBooti,1-alphaS,2);
    end
    
    mCriVals{idseries} = qhatAlphaS;
end

end


%########## Dynamics of Residuals for Sieve Bootstrap ##########%
function [vyhat,mbetatilde,mbetatildeGset,vepsiTilde,popt,b] = ResiDynamics(vy,mX,h,htilde,kx,GsetOption)


% Estimation under oversmoothing
mbetahathtildeAllGset = local_linear(vy,mX,h,kx,'GsetOption','AllGset',...
    'SmoothOption','Oversmooth','OversmoothBW',htilde);

mbetatilde = mbetahathtildeAllGset{3};    % n-dimensional
vyhat = sum(mX.*mbetatilde,2);

% Calculate residuals
vresi = vy - vyhat;

% Bootstrap residuals
popt = aic(vresi);                        % select optimal lags
vresilag = lagmatrix(vresi,0:popt);
vresilag(any(isnan(vresilag),2),:) = [];
if popt>0
    x = vresilag(:,2:1+popt);
    y = vresilag(:,1);
    b = x\y;
    vepsi = y-x*b; 
    vepsiTilde = vepsi - mean(vepsi);
else
    b = [];
    vepsi = vresi;
    vepsiTilde = vepsi - mean(vepsi);
end

switch upper(GsetOption)
    case upper('FullSample')
        mbetatildeGset = mbetahathtildeAllGset{3};
    case upper('Subsample')
        mbetatildeGset = mbetahathtildeAllGset{1};
    case upper('FullUnion')
        mbetatildeGset = mbetahathtildeAllGset{2};
end
        
end


function [] = MakeGraphs(mbetahat,PW,ST,datatime,datavars)

Nreg = size(mbetahat,2);

%======= Plots =======%

% Pointwise intervals & simultaneous bands (full sample)
for id = 1:Nreg
    fFullSample = figure('visible','off');
    hPW = plot(datatime,PW(:,:,id),':b','LineWidth',3);
    hold on
    hST = plot(datatime,ST{id,3},'--r','LineWidth',3);
    hold on
    plot(datatime,mbetahat(:,id),'-k','LineWidth',3)
    hold off
    grid on
    xtickangle(45)
    xlim([datatime(1)-0.5 datatime(end)+0.5])
    ax = gca;
    ax.FontSize = 20; 
    lg = legend([hPW(1),hST(1)],{'PW','ST'},'Interpreter','latex','Location','NE','FontSize',25);
    set(lg,'color','none','Box','off')
    title(datavars{id},'FontSize',45,'Interpreter','latex')
    set(fFullSample,'Position',[0,0,1000,800])
    saveas(fFullSample,['FullSample_',datavars{id},'.png'])
end


% Pointwise intervals & simultaneous bands (G)
for id = 1:Nreg
    fG = figure('visible','off');
    hPW = plot(datatime,PW(:,:,id),':b','LineWidth',3);
    hold on
    hST = plot(datatime,ST{id,2},'--r','LineWidth',3);
    hold on
    plot(datatime,mbetahat(:,id),'-k','LineWidth',3)
    hold off
    grid on
    xtickangle(45)
    xlim([datatime(1)-0.5 datatime(end)+0.5])
    ax = gca;
    ax.FontSize = 20; 
    lg = legend([hPW(1),hST(1)],{'PW','ST'},'Interpreter','latex','Location','NE','FontSize',25);
    set(lg,'color','none','Box','off')
    title(datavars{id},'FontSize',45,'Interpreter','latex')
    set(fG,'Position',[0,0,1000,800])
    saveas(fG,['G_',datavars{id},'.png'])
end

% Pointwise intervals & simultaneous bands (Gsub)
for id = 1:Nreg
    fGsub = figure('visible','off');
    hPW = plot(datatime,PW(:,:,id),':b','LineWidth',3);
    hold on
    hST = plot(datatime,ST{id,1},'--r','LineWidth',3);
    hold on
    plot(datatime,mbetahat(:,id),'-k','LineWidth',3)
    hold off
    grid on
    xtickangle(45)
    xlim([datatime(1)-0.5 datatime(end)+0.5])
    ax = gca;
    ax.FontSize = 20; 
    lg = legend([hPW(1),hST(1)],{'PW','ST'},'Interpreter','latex','Location','NE','FontSize',25);
    set(lg,'color','none','Box','off')
    title(datavars{id},'FontSize',45,'Interpreter','latex')
    set(fGsub,'Position',[0,0,1000,800])
    saveas(fGsub,['Gsub_',datavars{id},'.png'])
end


end