Using Multiple Linear Regression (MLR) to evaluate the trend of sea surface temperature and associated interaction with climate indexes.
==================================================================

Data description
-------------

Two major data are used in this tutorial. 1) Sea surface temperature anomalies during 1950 to 2018 from HadISST data (Titchner and Rayner, 2014). 2) Climate mode index, including MEI, AMO, PDO ,NPGO, and SAM.

Model description
-------------

The governing equation of the MLR used here is:

<a href="https://www.codecogs.com/eqnedit.php?latex=SSTA&space;=&space;b_0&space;&plus;&space;b_1&space;t&space;&plus;b_2&space;ENSO&space;&plus;&space;b_3&space;AMO&space;&plus;&space;b_4&space;PDO&space;&plus;&space;b_5&space;NPGO&space;&plus;&space;b_6&space;SAM" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SSTA&space;=&space;b_0&space;&plus;&space;b_1&space;t&space;&plus;b_2&space;ENSO&space;&plus;&space;b_3&space;AMO&space;&plus;&space;b_4&space;PDO&space;&plus;&space;b_5&space;NPGO&space;&plus;&space;b_6&space;SAM" title="SSTA = b_0 + b_1 t +b_2 ENSO + b_3 AMO + b_4 PDO + b_5 NPGO + b_6 SAM" /></a>

Where `t` is the time.

Step - by - Step Code
-------------

Firstly, we need to load the corresponding data. Firstly we load the SST anomalies. Since github does not allow too large dataset, I have to separate the whole data into files in every year. So we need to reconstruct them here.

```
%% Loading data
% SST anomalies
sst_anom=NaN(360,180,828);
for i=1950:2018
    data_here=['sst_anom_' num2str(i)];
    load(data_here);
    sst_anom(:,:,(1:12)+(i-1950)*12)=sst_here;
end
sst_anom=sst_anom(:,:,85:816);
```

Also load the climate mode indexes and restrict them into the corresponding periods.

```
% Climate Modes
load('pdo_line');%1950 - 2017
pdo_line=pdo_line(85:end,:);
load('sam_line');%1957 - 2018
sam_line=sam_line(1:732,:);
load('npgo_line');%1950 - 2018
npgo_line=npgo_line(85:816,:);
load('enso_line');%1950 - 2018
enso_line=enso_line(85:816,:);
load('amo_line');%1950 - 2018
amo_line=amo_line(85:816,:);
```


Then let's fit the model.

```
%% MLR

x=parpool(2);
x.IdleTimeout=6000;
% Here I execute parallel computation, using 2 CPU in my local machine. If you use super computer you could use more I think.


%% Fitting the Model

coef_t_fixed=NaN(360,180,2);
coef_enso_fixed=NaN(360,180,2);
coef_amo_fixed=NaN(360,180,2);
coef_pdo_fixed=NaN(360,180,2);
coef_npgo_fixed=NaN(360,180,2);
coef_sam_fixed=NaN(360,180,2);

parfor i=1:360;
    tic
    coef_t_fixed_here=NaN(180,2);
    coef_enso_fixed_here=NaN(180,2);
    coef_amo_fixed_here=NaN(180,2);
    coef_pdo_fixed_here=NaN(180,2);
    coef_npgo_fixed_here=NaN(180,2);
    coef_sam_fixed_here=NaN(180,2);
    for j=1:180;
        fixed_here=squeeze(sst_anom(i,j,:));
        
        if nansum(isnan(fixed_here))==0 && length(unique(fixed_here))~=1
            
%             data_used=[(1:size(enso_line,1))' enso_line(:,end) amo_line(:,end) pdo_line(:,end) npgo_line(:,end) sam_line(:,end)];
%             
%             
%             mdl_here = fitlm(data_used,fixed_here);
%             
%             coef_t_fixed_here(j,1)=mdl_here.Coefficients.Estimate(2);
%             coef_t_fixed_here(j,2)=mdl_here.Coefficients.pValue(2);
%             coef_enso_fixed_here(j,1)=mdl_here.Coefficients.Estimate(3);
%             coef_enso_fixed_here(j,2)=mdl_here.Coefficients.pValue(3);
%             coef_amo_fixed_here(j,1)=mdl_here.Coefficients.Estimate(4);
%             coef_amo_fixed_here(j,2)=mdl_here.Coefficients.pValue(4);
%             coef_pdo_fixed_here(j,1)=mdl_here.Coefficients.Estimate(5);
%             coef_pdo_fixed_here(j,2)=mdl_here.Coefficients.pValue(5);
%             coef_npgo_fixed_here(j,1)=mdl_here.Coefficients.Estimate(6);
%             coef_npgo_fixed_here(j,2)=mdl_here.Coefficients.pValue(6);
%             coef_sam_fixed_here(j,1)=mdl_here.Coefficients.Estimate(7);
%             coef_sam_fixed_here(j,2)=mdl_here.Coefficients.pValue(7);
           
           data_used=[ones(size(enso_line,1),1) (1:size(enso_line,1))' enso_line(:,end) amo_line(:,end) pdo_line(:,end) npgo_line(:,end) sam_line(:,end)];
            
           
           [b,bint,~,~,stats] = regress(fixed_here,data_used);
           
           coef_t_fixed_here(j,1)=b(2);
           coef_t_fixed_here(j,2)=double(bint(2,1).*bint(2,2)>0);
           coef_enso_fixed_here(j,1)=b(3);
           coef_enso_fixed_here(j,2)=double(bint(3,1).*bint(3,2)>0);
           coef_amo_fixed_here(j,1)=b(4);
           coef_amo_fixed_here(j,2)=double(bint(4,1).*bint(4,2)>0);
           coef_pdo_fixed_here(j,1)=b(5);
           coef_pdo_fixed_here(j,2)=double(bint(5,1).*bint(5,2)>0);
           coef_npgo_fixed_here(j,1)=b(6);
           coef_npgo_fixed_here(j,2)=double(bint(6,1).*bint(6,2)>0);
           coef_sam_fixed_here(j,1)=b(7);
           coef_sam_fixed_here(j,2)=double(bint(7,1).*bint(7,2)>0);
           
        end
    end
    
    coef_t_fixed(i,:,:)=coef_t_fixed_here;
    coef_enso_fixed(i,:,:)=coef_enso_fixed_here;
    coef_amo_fixed(i,:,:)=coef_amo_fixed_here;
    coef_pdo_fixed(i,:,:)=coef_pdo_fixed_here;
    coef_npgo_fixed(i,:,:)=coef_npgo_fixed_here;
    coef_sam_fixed(i,:,:)=coef_sam_fixed_here;
    
    toc
end

```

The fitted coefficients and its corresponding p - value are stored in `coef_*` variabiles. The dimension 1 and 2 of `coef_*` correspond to spatial scales, while the first layer of dimension 3 corresponds to fitted coefficients and the second layer corresponds to p - value.

Then we could plot the fitted results.

```
%% Drawing plots

load('lon_lat');
load('colormap_nature');

x0 = 0.1; y0 = 0.1; x1 = 0.03; y1 = 0.03; xx = 0.8/3; yy = 0.8/2;
for i = 1:3; for j = 1:2; pos(:,i,j) = [x0+(xx+x1)*(i-1) y0+(yy+y1)*(2-j) xx yy]; end; end;

lon=double(lon);lat=double(lat);
lon(lon<0)=180+lon(lon<0)+180;
m_proj('miller','lon',[nanmin(lon) nanmax(lon)],'lat',[nanmin(lat) nanmax(lat)]);
figure('pos',[10 10 1500 1500]);
h=tight_subplot(3,2,[0.05 0.01],[0.05 0.05],[0.01 0.01]);
subplot('position',pos(:,1,1));
data_here=coef_t_fixed(:,:,1)*12*10;
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-1,1,100),'linestyle','none');
colormap(colormap_nature);
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_t_fixed([181:end 1:180],:,2);
hold on
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
caxis([-0.7 0.7]);
m_text(50,-60,'a) T (F)','fontsize',16,'fontweight','bold');
s=colorbar('fontsize',12);
s.Label.String='^{o}C/decade';

subplot('position',pos(:,1,2));
data_here=coef_enso_fixed(:,:,1);
%m_pcolor(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))');
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-0.5,1.5,100),'linestyle','none');
colormap(colormap_nature);
hold on
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_enso_fixed([181:end 1:180],:,2);
hold on
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
m_text(50,-60,'b) ENSO (F)','fontsize',16,'fontweight','bold');
caxis([-1.3 1.3]);
s=colorbar('fontsize',12);
s.Label.String='^{o}C';

subplot('position',pos(:,2,1));
data_here=coef_amo_fixed(:,:,1);
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-3,3,100),'linestyle','none');
colormap(colormap_nature);
hold on
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_amo_fixed([181:end 1:180],:,2);
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
hold on
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
m_text(50,-60,'c) AMO (F)','fontsize',16,'fontweight','bold');
caxis([-3 3]);
s=colorbar('fontsize',12);
s.Label.String='^{o}C';

subplot('position',pos(:,2,2));
data_here=coef_pdo_fixed(:,:,1);
data_here(abs(data_here)>7)=nan;
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-0.7,0.7,100),'linestyle','none');
colormap(colormap_nature);
hold on
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_pdo_fixed([181:end 1:180],:,2);
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
hold on
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
m_text(50,-60,'d) PDO (F)','fontsize',16,'fontweight','bold');
caxis([-0.65 0.65]);
s=colorbar('fontsize',12);
s.Label.String='^{o}C';

subplot('position',pos(:,3,1));
data_here=coef_npgo_fixed(:,:,1);
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-0.4,0.4,100),'linestyle','none');
colormap(colormap_nature);
hold on
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_npgo_fixed([181:end 1:180],:,2);
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
hold on
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
m_text(50,-60,'e) NPGO (F)','fontsize',16,'fontweight','bold');
caxis([-0.4 0.4]);
s=colorbar('fontsize',12);
s.Label.String='^{o}C';

subplot('position',pos(:,3,2));
data_here=coef_sam_fixed(:,:,1);
m_contourf(lon([181:end 1:180]),lat,(data_here([181:end 1:180],:,1))',linspace(-0.1,0.1,100),'linestyle','none');
colormap(colormap_nature);
hold on
[lat_full,lon_full]=meshgrid(lat,lon);
lon_full=lon_full([181:end 1:180],:);
lat_full=lat_full([181:end 1:180],:);
p_here=coef_sam_fixed([181:end 1:180],:,2);
m_scatter(lon_full(p_here==1),lat_full(p_here==1),0.08,'k');
hold on
m_coast('patch',[0 0 0]);
m_grid('xtick',[],'ytick',[]);
m_text(50,-60,'f) SAM (F)','fontsize',16,'fontweight','bold');
caxis([-0.1 0.1]);
s=colorbar('fontsize',12);
s.Label.String='^{o}C';
```

![Image text](https://github.com/ZijieZhaoMMHW/MLR_SSTA/blob/master/example_trend.png)


