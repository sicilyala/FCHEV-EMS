import numpy as np
from FCHEV_SOH import FCHEV_SOH
from Cell import CellModel_2


class EMS:
    """ EMS with SOH """
    def __init__(self, w_soc, soc0, SOC_MODE, abs_spd_MAX, abs_acc_MAX):
        self.time_step = 1.0
        self.w_soc = w_soc
        self.done = False
        self.info = {}
        self.FCHEV = FCHEV_SOH()
        self.Battery = CellModel_2()
        self.obs_num = 7  # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        self.action_num = 1  # P_FCS
        # motor, unit in W
        self.P_mot_max = self.FCHEV.motor_max_power  # 200,000 W
        self.P_mot = 0
        # FCS, unit in kW
        self.h2_fcs = 0
        self.P_FCS = 0
        self.P_FCS_max = self.FCHEV.P_FC_max        # kW
        self.dSOH_FCS = 0
        self.SOH_FCS = 1.0
        # battery unit in W
        self.SOC_init = soc0        # 0.5
        if SOC_MODE == 'CD':        # charge-depletion
            self.SOC_target = self.SOC_init - 0.2
        else:                       # charge-sustaining
            self.SOC_target = self.SOC_init
        self.SOC = self.SOC_init
        self.OCV_initial = self.Battery.ocv_func(self.SOC_init)*13.87/168
        self.SOH_batt = 1.0
        self.Tep_a = 25
        self.P_batt = 0     # W
        self.P_batt_max = self.Battery.batt_maxpower    # in W
        self.SOC_delta = 0
        self.dSOH_batt = 0
        self.I_batt = 0
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a, Voc, V1, V2]
        self.paras_list = [self.SOC, self.SOH_batt, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.travle = 0
        self.car_spd = 0
        self.car_acc = 0
        self.abs_spd_MAX = abs_spd_MAX
        self.abs_acc_MAX = abs_acc_MAX
    
    def reset_obs(self):
        self.SOC = self.SOC_init
        self.SOH_batt = 1.0
        self.SOH_FCS = 1.0
        self.dSOH_FCS = 0
        self.Tep_a = 25
        self.P_mot = 0
        self.P_FCS = 0
        self.P_batt = 0
        self.paras_list = [self.SOC, self.SOH_batt, 25, 25, self.Tep_a, self.OCV_initial, 0, 0]
        self.done = False
        self.info = {}
        self.travle = 0
        self.car_spd = 0
        self.car_acc = 0
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        obs[0] = self.SOC
        obs[1] = self.SOH_batt
        obs[2] = self.SOH_FCS
        obs[3] = self.P_FCS / self.P_FCS_max        # in kW
        obs[4] = self.P_batt / self.P_batt_max   # in W
        obs[5] = self.car_spd / self.abs_spd_MAX
        obs[6] = self.car_acc / self.abs_acc_MAX
        return obs

    def execute(self, action, car_spd, car_acc):
        self.car_spd = car_spd
        self.car_acc = car_acc
        self.P_FCS = abs(action[0]) * self.FCHEV.P_FC_max     # kW
        T_axle, W_axle, P_axle = self.FCHEV.T_W_axle(self.car_spd, self.car_acc)
        T_mot, W_mot, mot_eff, self.P_mot = self.FCHEV.run_motor(T_axle, W_axle, P_axle)  # W
        P_dcdc, self.h2_fcs, info_fcs = self.FCHEV.run_fuel_cell(self.P_FCS)      # kW
        self.dSOH_FCS, info_fcs_soh = self.FCHEV.run_FC_SOH(self.P_FCS)
        self.SOH_FCS -= self.dSOH_FCS
        self.P_batt = self.P_mot - P_dcdc*1000        # W
        # update power battery
        self.paras_list, self.dSOH_batt, self.I_batt, self.done, info_batt = \
            self.Battery.run_cell(self.P_batt, self.paras_list)
        self.SOC = self.paras_list[0]
        self.SOH_batt = self.paras_list[1]
        self.Tep_a = self.paras_list[4]

        self.travle += self.car_spd*self.time_step
        self.info = {}
        self.info.update({'T_axle': T_axle, 'W_axle': W_axle, 'P_axle': P_axle/1000,
                          'T_mot': T_mot, 'W_mot': W_mot, 'mot_eff': mot_eff,
                          'P_mot': self.P_mot/1000, 'FCS_SOH': self.SOH_FCS,
                          'travel': self.travle})
        self.info.update(info_fcs)
        self.info.update(info_batt)
        self.info.update(info_fcs_soh)
        
        obs = np.zeros(self.obs_num, dtype=np.float32)  # np.array
        # soc, soh-batt, soh-fcs, P_FCS, P_batt, spd, acc
        obs[0] = self.SOC
        obs[1] = self.SOH_batt
        obs[2] = self.SOH_FCS
        obs[3] = self.P_FCS / self.P_FCS_max        # in kW
        obs[4] = self.P_batt / self.P_batt_max   # in W
        obs[5] = self.car_spd / self.abs_spd_MAX
        obs[6] = self.car_acc / self.abs_acc_MAX
        return obs
    
    def get_reward(self):
        # equivalent hydrogen consumption
        if self.P_batt > 0:
            h2_batt = self.P_batt / 1000 * 0.0164      # g in one second
            # 取FCS效率最高点(0.5622)计算, 该系数为0.0164
        else:
            h2_batt = 0
        h2_equal = self.h2_fcs + h2_batt
        
        # SOC cost
        # w_soc = 20.0  #
        # if 0.3 <= self.SOC <= 0.6:
        #     soc_cost = 0
        # else:
        #     soc_cost = w_soc*min(abs(self.SOC-0.6), abs(self.SOC-0.3))
        if self.SOC >= 0.95 or self.SOC <= 0.05:
            w_soc = self.w_soc * 10     # 不可直接改变self.w_soc的值
        else:
            w_soc = self.w_soc
        soc_cost = w_soc * abs(self.SOC - self.SOC_target)
        
        # money cost
        # hydrogen spent
        h2_price = 55/1000      # ￥ per g
        h2_money = h2_price * self.h2_fcs
        # money spent of power battery degradation
        # batt_price = 20000  # for 9.893 kWh * 2000 yuan/kWh = 19786 yuan
        # TODO  it should be 108.14kWh?
        batt_price = 100000     # 108.14kWh * 1000 yuan/kWh
        # batt_price = 0
        batt_money = batt_price * self.dSOH_batt
        # money spent of FCE degradation
        FCE_price = 300000
        # FCE_price = 0
        fcs_money = FCE_price * self.dSOH_FCS
        # total money cost in one step
        money_cost = h2_money + batt_money + fcs_money
        money_cost_real = h2_money + self.dSOH_batt * 100000 + self.dSOH_FCS * 300000
        
        reward = -(money_cost + soc_cost)
        reward = float(reward)
        self.info.update({'EMS_reward': reward, 'soc_cost': soc_cost,
                          'h2_equal': h2_equal, 'h2_batt': h2_batt,
                          'money_cost': money_cost, 'h2_money': h2_money,
                          'batt_money': batt_money, 'fcs_money': fcs_money,
                          'money_cost_real': money_cost_real})
        return reward

    def get_info(self):
        return self.info

    def get_done(self):
        return self.done
    