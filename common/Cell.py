"""
this scipt contains two LIB power battery models
an Equivalent circuit model
an electro-thermal-aging model
"""
import math
import pickle
from scipy.interpolate import interp1d


class CellModel_1:
    def __init__(self):
        self.timestep = 1
        self.I_max = 460  # Battery current limitation
        self.Q_batt = 25  # Ah
        # Nominal battery accumulate Ah-throughput
        self.nominal_life = pow(20/self.severity_factor_fun(0.35, 2.7, 41), 1/0.57)  # 48000 Ah
        self.accumulated_Ah = 0
        self._func_init()
    
    def _func_init(self):
        # SoC
        SOC_list = [0, 0.00132660038355303, 0.0574708841412090, 0.107025590196296, 0.156580329178506, 0.206134478930137,
                    0.255687515404994, 0.305240257178054, 0.354795683551491, 0.404476817165514, 0.454031523456938,
                    0.503585181935343, 0.553137334622058, 0.602688668725870, 0.652241410558679, 0.701796476729142,
                    0.751350724430672, 0.800904841303918, 0.850591981005177, 0.900297369762374, 0.950130539134393,
                    0.999900000000000, 1]
        # Discharging resistance
        R_dis = [0.313944125557612, 0.313944125557612, 0.313944125557612, 0.242065812972601, 0.230789057570884,
                 0.212882084646638, 0.196703117398054, 0.191970726117600, 0.182268278911152, 0.176602004689331,
                 0.173003239603114, 0.172594564949654, 0.177777197273542, 0.188139855710787, 0.212047123247474,
                 0.218402021824003, 0.211830447697718, 0.203822955457752, 0.197028250778952, 0.194459010902986,
                 0.218377885152249, 0.296468469580613, 0.296468469580613]  # ohm
        # Charging resistance
        R_chg = [0.278440842615743, 0.278440842615743, 0.278440842615743, 0.220685921062183, 0.219189603825182,
                 0.205347807991030, 0.191299532644795, 0.187277234013650, 0.178094888469773, 0.172984724665031,
                 0.169768580157996, 0.169487135972806, 0.174936544546881, 0.185495927766875, 0.208445003365321,
                 0.213703074103948, 0.206748591485580, 0.198442444892965, 0.191938639000188, 0.189976229191598,
                 0.214377935413427, 0.296464707815351, 0.296464707815351]
        # Open circuit voltage
        V_oc = [491.8387067, 491.8387067, 517.6726653, 521.6434164, 527.0477514, 531.5501691, 535.3424403,
                539.2562079, 541.5306725, 543.354122, 545.2170669, 547.3269911, 549.9472871, 553.641686,
                559.2529919, 564.8908875, 570.3695838, 576.2404755, 582.5029103, 589.1188686, 596.2324457,
                604.3783404, 604.3783404]
        V_oc = [(i+15)*360/570 for i in V_oc]
        self.R_dis_func = interp1d(SOC_list, R_dis, kind='linear', fill_value='extrapolate')
        self.R_chg_func = interp1d(SOC_list, R_chg, kind='linear', fill_value='extrapolate')
        self.V_func = interp1d(SOC_list, V_oc, kind='linear', fill_value='extrapolate')
    
    @staticmethod
    def severity_factor_fun(soc, I_c, T_batt):
        if soc > 0.45:
            # B = 2694.5*soc+6022.2
            B = 1385.5*soc+4193.2
        else:
            # B = 2896.6*soc+7411.2
            B = 1287.6*soc+6356.3
        E_a = 31700  # activation energy, J/mol
        R_g = 8.31  # Universal gas constant
        T_kelvin = 273.15+T_batt
        sigma = B*math.exp((163.3*I_c-E_a)/(R_g*T_kelvin))
        return sigma
    
    def run_cell(self, P_batt, SOC, Tep_c, Tep_s):
        fail = False
        # Battery efficiency
        e_batt = (P_batt > 0)+(P_batt <= 0)*0.98
        # Battery internal resistance, on package level
        resistance = (P_batt > 0)*self.R_dis_func(SOC)+(P_batt <= 0)*self.R_chg_func(SOC)
        # Battery voltage
        V_batt = self.V_func(SOC)  # OCV
        # Battery current
        if V_batt**2 < 4*resistance*P_batt:
            I_batt = e_batt*V_batt/(2*resistance)
        else:
            I_batt = e_batt*(V_batt-math.sqrt(V_batt**2-4*resistance*P_batt))/(2*resistance)
        if I_batt > self.I_max:
            I_batt = self.I_max
        I_c = abs(I_batt/self.Q_batt)  # c-rate
        SOC_new = -I_batt/self.Q_batt/3600+SOC
        if SOC_new > 1:
            SOC_new = 1.0
            fail = True
        if SOC_new < 0:
            SOC_new = 0.0
            fail = True
        # self.accumulated_Ah += abs(I_batt)*self.timestep/3600  # 1s , now is on package level, should on cell
        cell_nums = 110*4
        self.accumulated_Ah += abs(I_batt)*self.timestep/3600/cell_nums
        # Battery temperature, on cell or package?
        Tep_e = 25  # 环境的温度,摄氏度
        C_c = 62.7  # heat capacity of the core,J/K
        C_s = 4.5  # heat capacity of the casing,J/K
        R_c = 1.94  # heat conduction resistance,K/W
        # R_u = 15  # convection resistance,K/W       3.19?
        R_u = 3.19
        # I_cell = I_batt/4       # not good
        # H_gen = (I_cell**2)*resistance
        H_gen = (I_batt**2)*resistance
        delta_Tep_c = (Tep_s-Tep_c)/(R_c*C_c)+(H_gen/C_c)
        delta_Tep_s = ((Tep_e-Tep_s)/(R_u*C_s))-((Tep_s-Tep_c)/(R_c*C_c))
        Tep_c = Tep_c+delta_Tep_c
        Tep_s = Tep_s+delta_Tep_s
        if Tep_c > 55:
            Tep_c = 55
        if Tep_s > 55:
            Tep_s = 55
        Tep_a = (Tep_c+Tep_s)/2
        '''
        # battery aging 容量损失的经验模型 : on cell level
        # paper: Optimal energy management of hybrid electric vehicles including battery aging
        # severity_factor = self._severity_factor_fun(SOC_new, abs(I_c), Tep_a)
        # Ah_eff = severity_factor*abs(I_batt)
        # life_depleted_fraction = Ah_eff/self.nominal_life
        # estimated_life = pow(20/severity_factor, 1/0.57)
        '''
        # battery aging
        # paper: A control-oriented cycle-life model for hybrid electric vehicle lithium-ion batteries
        # paper: Energy Management Strategy for HEVs Including Battery Life Optimization
        sigma = self.severity_factor_fun(SOC_new, I_c, Tep_a)
        estimated_life = pow(20/sigma, 1/0.57)
        severity_factor = self.nominal_life/estimated_life
        Ah_eff = severity_factor*abs(I_batt)*self.timestep/3600
        life_depleted_fraction = Ah_eff/self.nominal_life  # 损失100%才报废
        
        out_info = {'SoC': SOC_new, 'battery_temperature': Tep_a, 'accumulated_Ah': self.accumulated_Ah,
                    'resistance': resistance, 'battery_OCV': V_batt, 'current': I_batt, 'current_rate': I_c,
                    'severity_factor': severity_factor, 'estimated_life': estimated_life,
                    'battery_power': P_batt/1000, 'life_depleted_fraction': life_depleted_fraction,
                    'Ah_eff': Ah_eff, 'sigma': sigma}
        done = fail
        return SOC_new, Tep_c, Tep_s, Tep_a, life_depleted_fraction, I_batt, done, out_info
   

class CellModel_2:
    def __init__(self):
        self.timestep = 1
        self.cell_n = 8       # n串, 6串对应最大输出功率是150kW
        self.batt_maxpower = self.cell_n * 25 * 1000  # W      [200kW, 9.893kWh]
        # used in electric model
        self.r0 = 0.0031  # Rs
        self.r1 = 0.0062
        self.r2 = 0.0054
        self.c1 = 8710.0
        self.c2 = 258.4211
        # used in heating model
        self.Cn = 2.422     # Ah
        self.Cc = 62.7
        self.Cs = 41
        self.Ru = 5.095
        self.Rc = 1.94
        # used in ageing model
        Ic_rate = [0.5, 2, 6, 10]
        Bc_data = [31630, 21681, 12934, 15512]
        self.Bc_func = interp1d(Ic_rate, Bc_data, kind='linear', fill_value='extrapolate')
        data_dir = "E:/SEU2/Program1/MADDPG-program/common/data/"
        self.ocv_func = pickle.load(open(data_dir+'ocv.pkl', 'rb'))

    def run_cell(self, P_batt, paras_list):
        # paras_list = [SOC, SOH, Tep_c, Tep_s, Tep_a, Voc, V1, V2]
        SOC = paras_list[0]
        SOH = paras_list[1]
        Tep_c = paras_list[2]
        Tep_s = paras_list[3]
        Tep_a = paras_list[4]
        Voc = paras_list[5]     # initial SOC 0.6: 3.237 V
        V1 = paras_list[6]
        V2 = paras_list[7]
        # battery power limit   # P_batt is in W
        if P_batt > self.batt_maxpower:
            P_batt = self.batt_maxpower
        if P_batt < -self.batt_maxpower:
            P_batt = -self.batt_maxpower
        # battery pack of 168*6 cells
        cell_num = 168 * self.cell_n
        P_cell = P_batt/cell_num  # in W
        # print('cell power: %.4f'%P_cell)
        V_3 = Voc+V1+V2
        delta = V_3**2-4*self.r0*P_cell
        if delta < 0:
            I_batt = V_3/(2*self.r0)
        else:
            I_batt = (V_3-math.sqrt(delta))/(2*self.r0)     # P>0 -> I>0 -> dsoc < 0
        Ic_rate = abs(I_batt/self.Cn)       # I_batt: I of one single cell
        cell_heat = I_batt*(V1+V2+self.r0*I_batt)  # H(t)
        soc_deriv = self.timestep*(I_batt/3600/self.Cn)
        v1_deriv = self.timestep*(-V1/self.r1/self.c1+I_batt/self.c1)
        v2_deriv = self.timestep*(-V2/self.r2/self.c2+I_batt/self.c2)
        tc_deriv = self.timestep*(((Tep_s-Tep_c)/self.Rc+cell_heat)/self.Cc)
        ts_deriv = self.timestep*(((Tep_c-Tep_s)/self.Rc+(Tep_a-Tep_s)/self.Ru)/self.Cs)
        # electric model
        SOC_new = SOC-soc_deriv
        if SOC_new >= 1:
            Voc_new = self.ocv_func(1.0)  # for a cell#
            fail = True
            # SOC_new = 1.0
        elif SOC_new <= 0.01:
            Voc_new = self.ocv_func(0.01)
            fail = True
            # SOC_new = 0.001
        else:
            Voc_new = self.ocv_func(SOC_new)  # for a cell
            fail = False
        # print('SOC: %.6f'%SOC_new)
        Voc_new = Voc_new*13.87/168
        V1_new = V1+v1_deriv
        V2_new = V2+v2_deriv
        # terminal voltage
        Vt_new = Voc_new+V1_new+V2_new+self.r0*I_batt
        power_out = Vt_new*I_batt
        # thermal model
        Tep_c_new = Tep_c+tc_deriv
        Tep_s_new = Tep_s+ts_deriv
        Tep_a_new = (Tep_c_new+Tep_s_new)/2
        if Tep_a_new > 60:
            Tep_a_new = 60
        # aging model
        Bc = self.Bc_func(Ic_rate)
        E = 31700-370.3*Ic_rate
        # T = Tep_a_new+273.15
        T = 25 + 273.15   # constant temperature assumption
        Ah = (20/Bc/math.exp(-E/8.31/T))**(1/0.55)  # z = 0.55, ideal_gas_constant = 8.31
        N1 = 3600*Ah/self.Cn
        dsoh = self.timestep*(abs(I_batt/2/N1/self.Cn))
        SOH_new = SOH-dsoh
    
        out_info = {'SOC': SOC_new, 'SOH': SOH_new, 'soc_deriv': soc_deriv,
                    'cell_OCV': Voc_new, 'cell_Vt': Vt_new, 'cell_V_3': V_3,
                    'cell_V1': V1_new, 'cell_V2': V2_new,
                    'I': I_batt, 'I_c': Ic_rate, 'cell_power_out': power_out,
                    'P_batt': P_batt/1000, 'tep_a': Tep_a_new, 'dsoh': dsoh}
        paras_list_new = [SOC_new, SOH_new, Tep_c_new, Tep_s_new, Tep_a_new, Voc_new, V1_new, V2_new]
        done = fail
        return paras_list_new, dsoh, I_batt, done, out_info
    