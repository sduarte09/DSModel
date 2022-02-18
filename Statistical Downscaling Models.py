# Santiago Duarte
# February 2022

# Importing Main libraries
import datetime as dt # Work with datetime format
import matplotlib.pyplot as plt
import netCDF4 as nc # Work with NetCDF format files
import numpy as np # Array processing for numbers, strings, records, and objects
import os # use OS functions
import pandas as pd # Work with Pandas DataFrames
import warnings # manage warning alerts
import sys
sys.path.append('G:\\Mi unidad\\PhD\\Research\\Machine Learning for Downscaling Comparison\\Scripts\\Machine Learning Models\\')
warnings.filterwarnings("ignore")

# Importing auxiliar functions
from DSMethods import multi_layer_perceptron # Import Linear regression Functions
from DSMethods import k_Nearest_Neighbors # Import k-NN Functions
from DSMethods import linear_regression # Import Linear regression Functions
from DSMethods import forest_trees # Import Forest Tree Decision Regression

def climatic_statistcal_downscaling(station_data,station_catalog,model_historic,model_rcp,output_folder,cal_date,val_date,app_date):
    # station_data: csv file with the climatic data of the stations
    # station_catalog: csv file with latitude and longitude of the stations
    # model_historic:  folder with the nc files with the historic experiment of the GCM
    # model_rcp: folder with the nc files with the RCP experiment of the GCM
    # output_folder: folder to save the results and plots
    # cal_date: last date of the calibration process
    # val_date: last date of the validation process
    # app_date: last date of the application process
    print("CLIMATIC STATISTICAL DOWNSCALING MODULE")
    print("Importing CSV Data")
    # The code works for daily values only right now
    station=pd.read_csv(station_data)
    dateparse = lambda x: pd.datetime.strptime(x,'%d/%m/%Y')
    station=pd.read_csv(station_data,index_col='Date',parse_dates=True, date_parser=dateparse)
    catalogo=pd.read_csv(station_catalog)
    if any(catalogo['Longitud']<0):
        catalogo['Longitud']=catalogo['Longitud']+360
    ns=station.shape[1]
    print("Loading NetCDF Data: Historic Data ")
    # Time model data
    time_his=[];time_rcp=[];    
    for filename in os.listdir(model_historic):
        his = nc.Dataset(model_historic+filename)
        if filename==os.listdir(model_historic)[0]:
            time_unit=his.variables['time'].units 
            time_cal=his.variables['time'].calendar
        time_valh=his.variables['time'][:] -0.5 
        time_his=np.concatenate((time_his,nc.num2date(time_valh,units=time_unit,calendar=time_cal)))    
    if time_his[0].year>=1900:
        time_his=[dt.datetime.strptime(k.strftime('%Y-%m-%d'),'%Y-%m-%d') for k in time_his]
    # Grid model data
    gcm_lat=his.variables['lat'][:]
    gcm_lon=his.variables['lon'][:]
    if  (gcm_lon<0).any():
        gcm_lon=gcm_lon+360   
    if len(gcm_lat.shape)==1:
        gcm_lat=np.array([gcm_lat,]* len(gcm_lon)).transpose()
        gcm_lon=np.array([gcm_lon,]* len(gcm_lat))  
    # Units Conversion
    var_type=his.variables.keys()[-1]
    if var_type=='pr':
        factor1=86400 # Conversion from kg/m^2/s to mm/day 
        factor2=0
        # units='(mm)'
    else: # Temperature: tas, tas_min, tas_max
        factor1=1
        factor2=-273.15 # Conversion from °K to ° C
        # units='(C)'     
    print("Loading NetCDF Data: RCP Data ")
    for filename in os.listdir(model_rcp):
        rcp = nc.Dataset(model_rcp+filename)  
        if filename==os.listdir(model_rcp)[0]:
            rcp_model=rcp.model_id
            rcp_esc=rcp.experiment_id
            time_unit1=rcp.variables['time'].units 
            time_cal1=rcp.variables['time'].calendar
        rcp = nc.Dataset(model_rcp+filename)
        time_valf=rcp.variables['time'][:]-0.5
        time_rcp=np.concatenate((time_rcp,nc.num2date(time_valf,units=time_unit1,calendar=time_cal1)))
    time_rcp=[dt.datetime.strptime(k.strftime('%Y-%m-%d'),'%Y-%m-%d') for k in time_rcp]
    time_gcm=np.concatenate((time_his, time_rcp))
    print("Get GCM cell and model information")
    gcm_cell=np.empty((ns,3))* np.nan 
    model=[None]*(ns-1)
    n=0
    start=station.index[0]
    knn_future=pd.DataFrame(index=pd.date_range(start,app_date))
    for_future=pd.DataFrame(index=pd.date_range(start,app_date))
    lin_future=pd.DataFrame(index=pd.date_range(start,app_date))
    for z in catalogo.index: #range(0,1): # 
        print("Station "+str(z+1) +": " +str(catalogo['ID'][z]))
        for cell_x in range(int(gcm_lat.shape[0])-1):
            for cell_y in range(int(gcm_lat.shape[1])-1):
                if (gcm_lat[cell_x-1,cell_y-1]+gcm_lat[cell_x,cell_y-1])/2<=catalogo['Latitud'][z] and catalogo['Latitud'][z]<=(gcm_lat[cell_x+1,cell_y-1]+gcm_lat[cell_x,cell_y-1])/2:    
                    if (gcm_lon[cell_x-1,cell_y-1]+gcm_lon[cell_x-1,cell_y])/2<=catalogo['Longitud'][z] and catalogo['Longitud'][z]<=(gcm_lon[cell_x-1,cell_y+1]+gcm_lon[cell_x-1,cell_y])/2:                   
                        gcm_cell[z,0]= cell_x    
                        gcm_cell[z,1]= cell_y
                    cell_y=cell_y+1
            cell_x=cell_x+1
        if any((np.array([int(gcm_cell[z,0]),int(gcm_cell[z,1])])== x).all() for x in np.delete(gcm_cell,z,0)[:,:2]):       
            cell_pos=np.where((gcm_cell[:,0:2]==[int(gcm_cell[z,0]),int(gcm_cell[z,1])]).all(axis=1))      
            gcm_cell[z,2]=gcm_cell[cell_pos[0][0],2]
        else:         
            data_his=[];data_rcp=[];
            for filename in os.listdir(model_historic):
                his = nc.Dataset(model_historic+'\\'+filename)    
                data_his=np.concatenate((data_his,his.variables[var_type][:,int(gcm_cell[z,0]),int(gcm_cell[z,1])]*factor1+factor2))    
            for filename in os.listdir(model_rcp):
                rcp = nc.Dataset(model_rcp+'\\'+filename)      
                data_rcp=np.concatenate((data_rcp,rcp.variables[var_type][:,int(gcm_cell[z,0]),int(gcm_cell[z,1])]*factor1+factor2))    
            model[n]=np.concatenate((data_his, data_rcp))
            gcm_cell[z,2]=n
            n=n+1
        his.close
        rcp.close
        # Relacionando datos para cada estación
        estacion=station[str(catalogo['ID'][z])]
        estacion=estacion.dropna()
        gcm=pd.DataFrame(model[int(gcm_cell[z,2])],index=time_gcm,columns=['GCM']) 
        # K-Nearest Neighbors
        print("    k-NN")
        knn_future[str(catalogo['ID'][z])]=k_Nearest_Neighbors.K_NN(estacion,gcm,cal_date,app_date)
        data_anually=knn_future[str(catalogo['ID'][z])].resample('A', how='sum')
        data_anually.plot()
        plt.savefig(output_folder+str(catalogo['ID'][z])+'_'+'knn'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.png')
        
        # Random Forest
        print("    Forest")
        for_future[str(catalogo['ID'][z])]=forest_trees.Decision_tree(estacion,gcm,cal_date,app_date)
        data_anually=for_future[str(catalogo['ID'][z])].resample('A', how='sum')
        data_anually.plot()
        plt.savefig(output_folder+str(catalogo['ID'][z])+'_'+'forest'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.png')

        # Linear regression
        print("    Linear")
        lin_future[str(catalogo['ID'][z])]=linear_regression.Lin_regr(estacion,gcm,cal_date,app_date)
        data_anually=lin_future[str(catalogo['ID'][z])].resample('A', how='sum')
        data_anually.plot()
        plt.savefig(output_folder+str(catalogo['ID'][z])+'_'+'forest'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.png')    

        # Multi Layer Perceptron (MLP)   
        print("MLP")
        
    knn_future.to_csv(output_folder+'knn'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.csv',index_label='Index')
    for_future.to_csv(output_folder+'forest'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.csv',index_label='Index')
    lin_future.to_csv(output_folder+'linear'+'_'+rcp_model+'_'+var_type+'_'+rcp_esc+'.csv',index_label='Index')
    #ann_future
    return [knn_future,for_future,lin_future]