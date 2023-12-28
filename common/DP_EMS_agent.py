from FCHEV_SOH import FCHEV_SOH
from Cell import CellModel_2


class DP_EMS_agent:
    def __init__(self, w_soc=100, gamma=0.9):
        self.FCHEV = FCHEV_SOH()
        self.Battery = CellModel_2()
        self.info={}
        self.done=False
        self.w_soc = w_soc
        self.gamma = gamma
        self.SOC_target = 0.5    # CS
        self.h2_fcs = 0
        self.dSOH_FCS = 0
        self.P_batt = 0
        self.dSOH_batt = 0
        
    def execute(self, action, car_spd, car_acc, soc):
        P_FCS = abs(action)*self.FCHEV.P_FC_max  # kW
        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(car_spd, car_acc)
        T_mot, W_mot, mot_eff, P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(P_FCS)  # kW
        self.dSOH_FCS, info_fcs_soh = self.FCHEV.run_FC_SOH(P_FCS)
        self.P_batt = P_mot-P_dcdc*1000  # W
        # update power battery
        OCV = self.Battery.ocv_func(soc)*13.87/168
        paras_list = [soc, 0.999999, 25, 25, 50, OCV, 1.56, 0.44]
        paras_list, self.dSOH_batt, I_batt, self.done, info_batt = \
            self.Battery.run_cell(self.P_batt, paras_list)
     
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': P_mot/1000})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        # SOC-NEW
        SOC_new = paras_list[0]
        return SOC_new
    
    def get_reward(self, soc_new):
        # equivalent hydrogen consumption
        if self.P_batt > 0:
            h2_batt = self.P_batt/1000*0.0164  # g in one second
        else:
            h2_batt = 0
        h2_equal = self.h2_fcs+h2_batt
        
        if soc_new >= 0.95 or soc_new <= 0.05:
            w_soc = self.w_soc*10  # 不可直接改变self.w_soc的值
        else:
            w_soc = self.w_soc
        soc_cost = w_soc*abs(soc_new-self.SOC_target)
    
        # money cost
        # hydrogen spent
        h2_price = 55/1000  # ￥ per g
        h2_money = h2_price*self.h2_fcs
        batt_price = 100000  # 108.14kWh * 1000 yuan/kWh
        batt_money = batt_price*self.dSOH_batt
        # money spent of FCE degradation
        FCE_price = 1
        fcs_money = FCE_price*self.dSOH_FCS
        # total money cost in one step
        money_cost = h2_money+batt_money+fcs_money
        money_cost_real = h2_money+self.dSOH_batt*100000+self.dSOH_FCS*300000
    
        reward = -(money_cost+soc_cost)
        reward = float(reward)
        self.info.update({'EMS_reward': reward, 'soc_cost': soc_cost,
                          'h2_equal': h2_equal, 'h2_batt': h2_batt,
                          'money_cost': money_cost, 'h2_money': h2_money,
                          'batt_money': batt_money, 'fcs_money': fcs_money,
                          'money_cost_real': money_cost_real})
        return reward, money_cost_real

    def get_info(self):
        return self.info

    def get_done(self):
        return self.done



