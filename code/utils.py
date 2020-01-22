import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import pickle
import matplotlib.pyplot as plt

class time_model:
    def __init__(self,data):
        self.data = data
        
    ### model grid search ###
    def auto_arimga_grid(self):
        information = ['aic',"bic"]
        seasonal = [True,False]
        season_num = [2,4,12]
        #out_sample_table = {'1주':4,'2주':2,'3주':2,'4주':1}
        trend = ["c","t","ct"]
        Ds = [0,1]
        ds = [0,1]
        models = []
        for cre in information:
            for tr in trend:
                for d in ds:
                    for D in Ds:
                        for sea in seasonal:
                            for m in season_num:
                                #print("여까지됨",cre,tr,d,D,sea,m)
                                if sea == False:
                                    m = 1
                                    try:
                                        model = auto_arima(self.data,start_p=0,start_q=0,d=d,
                                                        max_p=10,max_q=10,
                                                        m=m, seasonal=sea,
                                                        start_Q=0,start_P=0,D=D,
                                                        max_Q=5,max_P=5,
                                                        information_criterion=cre,
                                                        trend=tr,stepwise=True,trace=False,
                                                        error_action='ignore',
                                                        suppress_warnings=True)
                                    except:
                                        continue
                                    models.append(model)
        return models
    
    ### metric####
    def MAE(self,y_true,y_pred):
        return np.mean(np.abs((y_true - y_pred)))
    
    def RMSE(self,y_true,y_pred):
        return np.sqrt(np.mean(np.power((y_true-y_pred),2)))
    
    ### model save load ###
    def save_model(self,model,save_path):
        pickle.dump(model, open(save_path, 'wb'))
        print("model save complete")
    
    def load_model(self,load_path):
        loaded_model = pickle.load(open(load_path, 'rb'))
        return loaded_model
    
    ### ploting forcast arimga model ###
    def plot_result(self,y_true,y_pred):
        y_true.plot()
        plt.label("Actual")
        y_red.plot()
        plt.label("Predict")
        plt.title("Forecaste ARIMA Model")
        plt.show()
        
        