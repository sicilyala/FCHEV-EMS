from common.FCHEV_SOH import FCHEV_SOH
from common.utils import get_driving_cycle, get_acc_limit

if __name__=='__main__':
    Car=FCHEV_SOH()
    mass = 8400
    P_req_dict = {}
    spd_acc_max_list = {}
    scenario_names = ["Standard_ChinaCity", "FTP75_2", "Standard_HWFET", "Standard_IM240",
                      "Standard_JN1015", "Standard_UDDS", "Standard_WVUCITY", "Standard_WVUINTER",
                      "Standard_WVUSUB", "Standard_NEDC", "CLTC_P", "CYC_NYCC", "CYC_US06",
                      "CLTCP_WVUSUB_WVUINTER", "WVUCITY_HWFET", "CLTCP_WVUINTER"]
    for scenario_name in scenario_names:
        speed_list = get_driving_cycle(cycle_name=scenario_name)
        acc_list = get_acc_limit(speed_list, output_max_min=False)
        spdmax = max(speed_list)
        accmax = max(acc_list)
        accmin = min(acc_list)
        absspdmax = max(abs(speed_list))
        absaccmax = max(abs(accmin), abs(accmax))
        spd_acc_max_list.update({scenario_name: [absspdmax, absaccmax]})
        P_req_list = []
        timestep = len(speed_list)
        # print(timestep)
        for i in range(timestep):
            car_spd = speed_list[i]
            car_acc = acc_list[i]
            T_axle, W_axle, P_axle = Car.T_W_axle(car_spd, car_acc)  # in W
            T_mot, W_mot, mot_eff, P_mot = Car.run_motor(T_axle, W_axle, P_axle)
            P_req_list.append(P_mot/1000)
        P_req_max_min = (max(P_req_list), min(P_req_list))
        P_req_dict.update({scenario_name: [P_req_max_min, spdmax, accmax]})
    print('\n------P_motor_max_min------')
    for key in P_req_dict:
        print(key, P_req_dict[key])
    print('\n------spd_acc_max_list------')
    for key in spd_acc_max_list:
        print(key, spd_acc_max_list[key])
    
    