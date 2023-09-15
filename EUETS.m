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


clear;clc
set(findobj(0,'type','figure'),'visible','on')
close all

%======== Part 1. Plots of level data ========%
% Import data
euets = importdata('Data_weekly_final.csv');
datatime = euets.data(:,1);
vy = euets.data(:,2);
coal = euets.data(:,3);
gas = euets.data(:,4);

% First-differencing the log of stocks
stocks = euets.data(:,5);
stocks = diff(log(stocks));

% Make plots
figure(1)
yyaxis left
plot(datatime,vy,'LineWidth',2)
ylabel('EUA')
hold on 
yyaxis right
plot(datatime,coal,'LineWidth',2)
ylabel('Coal')
grid on
xtickangle(45)
xlim([datatime(1)-1 datatime(end)+1])
ax = gca;
ax.FontSize = 30;
set(gcf,'Position',[0,0,1000,800])

figure(2)
yyaxis left
plot(datatime,vy,'LineWidth',2)
ylabel('EUA')
hold on 
yyaxis right
plot(datatime,gas,'LineWidth',2)
ylabel('Gas')
grid on
xtickangle(45)
xlim([datatime(1)-1 datatime(end)+1])
ax = gca;
ax.FontSize = 30;
set(gcf,'Position',[0,0,1000,800])


%======== Part 2. OLS regression results ========%
% Import data
euets = importdata('Data_weekly_clean.csv');
datatime = euets.data(:,1);
vy = euets.data(:,3);
coal = euets.data(:,4);
gas = euets.data(:,5);
oil = euets.data(:,7);
tempRaw = euets.data(:,8);
tempDesea = importdata('Temp_weekly_deseas.csv');
tempDesea = tempDesea.data(2:end,2);

%--- Model (a): with oil ---%
mXoil = [coal gas oil tempDesea];
[~,seHACoil,coeffoil] = hac(mXoil,vy,'Bandwidth','AR1MLE'); 
tstatsoil = coeffoil./seHACoil;
vpvalsoil = zeros(length(tstatsoil),1);
for id = 1:length(tstatsoil)
    if tstatsoil(id) >=0 
        vpvalsoil(id) = (1-cdf('Normal',tstatsoil(id),0,1))*2;
    else
        vpvalsoil(id) = cdf('Normal',tstatsoil(id),0,1)*2;
    end
end

%--- Model (b): with stocks ---%
mXstocks = [coal gas tempDesea stocks];
[~,seHACstocks,coeffstocks] = hac(mXstocks,vy,'Bandwidth','AR1MLE'); 
tstatsstocks = coeffstocks./seHACstocks;
vpvalsstocks = zeros(length(tstatsstocks),1);
for id = 1:length(tstatsstocks)
    if tstatsstocks(id) >=0 
        vpvalsstocks(id) = (1-cdf('Normal',tstatsstocks(id),0,1))*2;
    else
        vpvalsstocks(id) = cdf('Normal',tstatsstocks(id),0,1)*2;
    end
end


%======== Part 3. Time-varying model ========%
% parameter specifications
hL = 0.06;
hU = 0.28;
gridBW = hL:0.005:hU;                        % search range for bandwidth selections
kx = @(x) 0.75.*(1-x.^2).*(abs(x) <= 1);     % Epanechnikov kernel


%--- Model with oil ---%
siglevel = [0.01 0.05];                      % significance levels; the first is 
                                             %  for parameter constancy
                                             %  test, the second is for bands
mXoilIntercept = [ones(length(vy),1) mXoil];
datavarsoil = {'Intercept', 'Coal', 'Gas', 'Oil', 'Temperature'};
[mbetahatoil,vhDataDrivenoil,PWoil,SToil] = TSConBands_DataDriven(vy,mXoilIntercept,kx,gridBW,...
                                              siglevel,datatime,datavarsoil);


%--- Model with stocks ---%
siglevel = [0.01 0.05];                      % significance levels; the first is 
                                             %  for parameter constancy
                                             %  test, the second is for bands
mXstocksIntercept = [ones(length(vy),1) mXstocks];
datavarsstocks = {'Intercept', 'Coal', 'Gas', 'Temperature', 'Stocks'};
[mbetahatstocks,vhDataDrivenstocks,PWstocks,STstocks] = TSConBands_DataDriven(vy,mXstocksIntercept,kx,gridBW,...
                                              siglevel,datatime,datavarsstocks);















