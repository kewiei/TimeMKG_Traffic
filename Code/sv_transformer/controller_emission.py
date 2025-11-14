from collections import deque
import logging
import os
import argparse
import configparser
import time
import gym
import numpy as np
import uuid
import collections
import sys, shutil

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import env.env_cam
from utilities.agent import DiscretePolicy
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sumo_control_5 as sc
import traci
import sumolib
from sumolib import checkBinary
from utilities.simulator_demand import source_demand, source_speed, source_demand_5, source_speed_5, source_demand_5_new
from utilities.ssm_candidate_e import Ssm_cane
from utilities.gear_seq import gear_seq
from utilities.type_calibrator import append_vtype_skip_unbusy_road_5min_agg

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/flatbushlarge5min.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file for controller: %s' % (args.config), flush=True)
config.read(args.config)
# data_config = config['Data']
# training_config = config['Training']
# connector_config = config['Connector']
controller_config = config['Controller']
simulator_config = config['Simulator']

source_data = pd.read_csv(controller_config['data_dict'])

simProjDir = simulator_config['simProjDir']
simMainFile = simulator_config['simMainFile']
randomSeed = int(simulator_config['randomSeed'])
acc_num = int(controller_config['acc_num'])

if controller_config.getboolean('connector_use'):
    from connector_1h_class import Connector_1h
    CT = Connector_1h(config, args.cuda)
    initial_row_prediction = CT.predict_main('prediction')
    # from connector_1h_TimeMKG_class import Connector_1h_TimeMKG
    # CT = Connector_1h_TimeMKG(config, args.cuda)
    # initial_row_prediction = CT.predict_main('prediction')

wz_flag = -1    

if "WZ_2_0" in controller_config['params_path']:
    wz_flag = 0
elif "WZ_2_1" in controller_config['params_path']:
    wz_flag = 1
elif "WZ_2_2" in controller_config['params_path']:
    wz_flag = 2

# initial add file purge and copy
# shutil.copy(simProjDir+'flatbush_game_large_o.add.xml',simProjDir+'flatbush_game_large.add.xml')
sc.Tools.init_addtionalfile(randomSeed, simProjDir)

print('Begin experiments index %s' % controller_config['params_path'], flush=True)
if not os.path.exists("./results/" + controller_config['params_path']):
    os.makedirs("./results/" + controller_config['params_path'])
with open("./results/" + controller_config['params_path']+"config.conf", 'w') as backup_file:
    config.write(backup_file)
    
# if not os.path.exists("./plots/" + controller_config['params_path']):
#     os.makedirs("./plots/" + controller_config['params_path'])


# All the edges ID
edges_pool = ["1-2","1-901","1-903","2-1","2-5","2-905","2-906","5-2","5-6","5-16","6-5","6-8","6-16","6-21","7-5","7-8","8-6","8-7","8-10","9-7","9-10",
                "10-8","10-9","10-12","11-9","11-12","12-10","12-11","12-14","12-23","13-11","13-14","14-12","14-24","14-909","16-6","16-21","16-907",
                "21-6","21-16","21-22","22-21","22-23","22-26","23-12","23-24","23-27","24-27","26-22","26-32","26-34","26-911","27-23","27-28","28-27","28-29","28-32","29-28","29-30",
                "30-29","30-31","30-912","31-29","31-30","31-33","32-26","32-28","32-33","33-31","33-32","33-915","34-26","34-35","35-34","35-36","35-37",
                "36-35","36-37","36-40","37-35","37-38","37-41","38-37",
                "40-36","40-42","41-37","41-40","41-44","42-40","42-45","43-41","43-44","44-41","44-43","44-45","44-46","45-44","45-46","45-917","46-44","46-918","46-919",
                "901-1","903-1","904-1","905-2","907-16","908-13","909-14","910-22","911-26","913-34","914-36","915-33","916-43","917-45","919-46"]
        
def update_policies_md(policies, prime, dual_em, md_lambda, rho):
    for agent, policy in enumerate(policies):
        policy_probs = policy.action_probs
        res_prime = np.array(prime[agent])
        # option 2: clip res_dual to 0
        em_res_dual = np.array(dual_em[agent])
        em_dual_expectation = np.sum(policy_probs * em_res_dual)
        # option 2: clip res_dual to 0
        # ped_res_dual = np.array(dual_ped[agent])
        # ped_dual_expectation = np.sum(policy_probs * ped_res_dual)

        # products = policy_probs * np.exp(res_prime-md_lambda[agent] * res_dual)
        products = policy_probs * np.exp(res_prime-\
                                         np.maximum(0, rho * em_dual_expectation + md_lambda[agent]) * em_res_dual
                                         )
        new_probs = products/np.sum(products)
        policy.update_distribution(new_probs)
        
        # lambda_products = np.sum(policy_probs * res_dual)
        # md_lambda[agent] = np.maximum( (1-ita*gamma) * md_lambda[agent] +  gamma*lambda_products, 0)
        new_dual_expectation = np.sum(new_probs * em_res_dual)
        md_lambda[agent] = np.maximum(0, md_lambda[agent] + rho * new_dual_expectation)
        # new_dual_expectation = np.sum(new_probs * ped_res_dual)
        # ped_md_lambda[agent] = np.maximum(0, ped_md_lambda[agent] + rho * new_dual_expectation)
    return md_lambda

def get_prediction(pre_input, pre_input2, step):
    # start from 11
    CT.predictor_input[step+11,:,0] = pre_input
    CT.predictor_input[step+11,:,1] = pre_input2
    CT.read_and_generate_dataset_encoder_decoder(save=False)
    # all the input has been normalized into range [-1,1] by MaxMin normalization
    CT._load_graphdata_normY_connector()
    
    return CT.predict_and_save_results('prediction')

def startSumo(simProjDir, simMainFile, simSeed, exp_dir, simMode="sumo", 
              isTrajSaved=False, trajFile=None, trajStartTime=None,
              isTripInfoSaved=False, tripInfoFile=None):
    """
    Initialize the simuation and return a connection with SUMO.
    """

    # Check whether output directory exists.
    if not os.path.exists(simProjDir+"Output"):
        os.mkdir(simProjDir+"Output")

    simSettings = [checkBinary(simMode), 
                   "-c", simProjDir+simMainFile, 
                   "--step-length", "1", 
                   "--seed", str(simSeed), 
                   "--no-warnings", "true", 
                   "--collision.action", "none",
                #    "--device.ssm.file", "ssm_all_{}.xml".format(simSeed),
                #    "--time-to-teleport", "-1",
                   "--log", "sumo-log-debug.txt",
                #    "--device.rerouting.threads", "2",
                #    "--routing-threads", "2",
                   "--ignore-junction-blocker", "30"]
    
    if isTrajSaved:
        simSettings.extend(["--fcd-output", simProjDir+"Output/"+trajFile])
        simSettings.extend(["--fcd-output.acceleration", "true"])
        simSettings.extend(["--device.fcd.begin", str(trajStartTime)])
        
    if isTripInfoSaved:
        simSettings.extend(["--tripinfo-output", simProjDir+"Output/"+tripInfoFile])
        
    if simMode == "sumo-gui":
        simSettings.extend(["--start", "true"])
        simSettings.extend(["--quit-on-end", "true"])
    session_id = uuid.uuid4().hex[:5]
    label_name = f"ce_{session_id}"
    # label_name = 'default'
    traci.start(simSettings, label=label_name,verbose=True)
    return traci.getConnection(label_name)
    # traci.start(simSettings)
    # return traci.getConnection()

def call_sumo_30min(exp_dir, flow = None, speed = None, hour = None, close_time = {}, open_time = {}, car_obs = None):
    
    if flow is not None:
        source_demand_5[hour] = flow[1]
        source_demand_5[hour-1] = flow[0]
        try:
            source_demand_5[hour+1] = flow[0]
        except:
            print("Should be last row!")
        # try:
        #     source_demand_5[hour+2] = flow[1]
        # except:
        #     print("Approch to the half end when gear 1!!")
        
    if speed is not None:
        source_speed_5[hour] = speed[1]
        source_speed_5[hour-1] = speed[0]
        try:
            source_speed_5[hour+1] = speed[0]
        except:
            print("Should be last row!")
        # try:
        #     source_speed_5[hour+2] = speed[1]
        # except:
        #     print("Approch to the half end when gear 1!!")
    
    sumo = startSumo(simProjDir, simMainFile, randomSeed, exp_dir, "sumo", 
                    isTrajSaved=False, trajFile="traj.xml", trajStartTime=0,
                    isTripInfoSaved=False, tripInfoFile="tripInfo.xml")
    
    
    # Default demand matrix for the 
    demand_origin = np.array(source_demand_5)
    # Update the speed matrix from your output
    speed_update = np.array(source_speed_5)
    # TODO: here
    # sc.Tools.generate_addtionalfile(randomSeed, exp_dir)
    sc.Tools.generate_routefile_30m(demand_origin, exp_dir)
    
    sumo = startSumo(simProjDir, simMainFile, randomSeed, exp_dir, "sumo", 
                    isTrajSaved=False, trajFile="traj.xml", trajStartTime=0,
                    isTripInfoSaved=False, tripInfoFile="tripInfo.xml")
    
    # Wrap up SUMO.
    env = sc.SimulationEnvironment(sumo)
    
    # options = get_options()
    # Set simulation time.
    totalTime = int(3600 * round(hour/2 + 3, 1)) # in unit of seconds.
    # set maximum simulation time because in current sumo version, there is a bug that will cause the sumulation to bug out after 86400s when there are calibrators
    totalTime = min(totalTime, 86400)

    startTimestamp = datetime.datetime.now()
    
    for _ in range(totalTime):
    
        # Simulate from t to t+1.
        env.update()
        
        #TODO id changing function
        
        #print(f"environment step: {env.step}")
        
    traci.close()
    time.sleep(30)
    
    endTimestamp = datetime.datetime.now()
    print("Simulation running time: {} seconds".format((endTimestamp-startTimestamp).seconds))
    
    # Save measurements from SUMO API.
    # outputDir = simProjDir + "Output"

    sumo_volume, sumo_speed, sumo_count = sc.Tools.output_convert(randomSeed, exp_dir)
    em_output = sc.Tools.em_convert(randomSeed, hour, exp_dir)
    
    return em_output, sumo_volume, sumo_speed, sumo_count, close_time, open_time

def calculate_hour_interval(time_in_seconds):
    return time_in_seconds // 300

def process_em(data, new_columns):
    """
    transpose the data and reorder the columns according to new_columns
    param data: the data to be processed
    new_columns: the new columns order
    return: the processed data
    """
    data_new = data.pivot(index='interval_begin', columns='edge_id', values='CO2_normed')
    data_new.reset_index(drop=True, inplace=True)

    # Create a new DataFrame with the desired columns, initializing with NaNs
    new_df = pd.DataFrame(columns=new_columns)

    # Align the existing pivoted DataFrame with the new DataFrame
    for column in data_new.columns:
        if column in new_df.columns:
            new_df[column] = data_new[column]
    # Fill the 'Interval' column with a sequence of numbers starting from 0
    new_df['Interval'] = range(len(new_df))
    # data_new = data_new.ffill()
    new_df = new_df.fillna(0)
    return new_df

def create_mask_list(data_list, index_list):
    # Initialize a list with zeros, same length as data_list
    mask_list = [0] * len(data_list)
    
    # Set positions specified in index_list to 1
    for index in index_list:
        if 0 <= index < len(data_list):
            mask_list[index] = 1
    
    return mask_list

def generate_difference_list(A, B, C):
    # Initialize the difference list D with the same length as A and B
    D = [0] * len(A)
    
    # Iterate over the elements of A and B
    for i in range(len(A)):
        # Calculate the absolute difference if the index is in C
        if i in C:
            D[i] = abs(A[i] - B[i])/max(B[i],0.01) if abs(A[i] - B[i])/max(B[i],0.01) > -1 else -1 #0.2 to be changed to 0.0
        # If the index is not in C, set the difference to 0
        else:
            D[i] = -1
            
    return D

def get_diff(volum, speed, v_t, s_t, cane, time, filling_bound_v, filling_bound_s, n_sumo):
    
    if cane == [] or time<=60:
        v_diff = [-1] * len(v_t)
        s_diff = [-1] * len(s_t)
    else:
        v_mean = np.mean(volum[time-n_sumo:time], axis=0).tolist()
        s_mean = np.mean(speed[time-n_sumo:time], axis=0).tolist()
        
        v_diff = generate_difference_list(v_mean, v_t, cane)
        
        s_diff = generate_difference_list(s_mean, s_t, cane)
    
    return v_diff, s_diff

def fill_na_sumo_data(data, filling_bound):
    data = data.ffill()
    data = data.fillna(filling_bound)
    return data.to_numpy(dtype=np.float32)

def camera_control():
    # Create the environment
    num_agents = int(controller_config['num_agents'])
    num_features = int(controller_config['num_features'])
    data_dict = controller_config['data_dict']
    data_dict2 = controller_config['data_dict2']
    # data_dict3 = controller_config['lane_dict']
    pedestrian_dict = controller_config['pedestrian_dict']
    # ssm_data_dict = controller_config['ssm_dict']
    # ssm_count_dict = controller_config['ssm_count_dict']
    em_dict = controller_config['em_dict']
    em_lst_path = controller_config['em_lst_path']
    car_type_dict = controller_config['car_type_dict']
    save_dict = "./results/"+controller_config['params_path']+"/saved_data.csv"
    save_dict2 = "./results/"+controller_config['params_path']+"/saved_data_s.csv"
    save_fu_dict = "./results/"+controller_config['params_path']+"/saved_fu_data.csv"
    save_fu_dict2 = "./results/"+controller_config['params_path']+"/saved_fu_data_s.csv"
    save_predict_dict = "./results/"+controller_config['params_path']+"/saved_pre_data.csv"
    save_predict_dict2 = "./results/"+controller_config['params_path']+"/saved_pre_data_s.csv"
    counted_em_dict = "./results/"+controller_config['params_path']+"/counted_em"
    counted_em_dict_original_true  = "./results/"+controller_config['params_path']+"/em_true.csv"
    counted_em_dict_true = "./results/"+controller_config['params_path']+"/counted_em_true.csv"
    save_cane_dict = "./results/"+controller_config['params_path']+"/cane_ssm.csv"
    
    re_save_dict = "./results/"+controller_config['params_path']+"/re_saved_data.json"
    re_save_dict_true = "./results/"+controller_config['params_path']+"/re_saved_data_true.json"
    em_re_save_dict = "./results/"+controller_config['params_path']+"/em_re_saved_data.json"
    action_list_dict = "./results/"+controller_config['params_path']+"/saved_action.npy"
    plot_dict = "./plots/"+controller_config['params_path']+"/changing_pattern_plot_sampled.png"
    plot_dict_true = "./plots/"+controller_config['params_path']+"/changing_pattern_plot_sampled_true.png"
    policy_dict = "./results/"+controller_config['params_path']+"/saved_policy_probs.json"
    delta_true_dict = "./results/"+controller_config['params_path']+"/saved_true_delta_state.npy"
    delta_dict = "./results/"+controller_config['params_path']+"/saved_delta_state.npy"
    md_lambda_list_dict = "./results/"+controller_config['params_path']+"/md_lambda.npy"
    # ped_md_lambda_list_dict = "./results/"+controller_config['params_path']+"/ped_md_lambda.npy"
    plot_delta_dict = "./plots/"+controller_config['params_path']+"/delta_plot.png"
    close_time_dict = "./results/"+controller_config['params_path']+"/close_time.json"
    open_time_dict = "./results/"+controller_config['params_path']+"/open_time_dict.json"
    
    # num_actions = int(controller_config['num_actions'])
    bar = int(controller_config['bar'])
    seed = int(controller_config['seed'])
    connector_use = controller_config.getboolean('connector_use')
    test_predictor = controller_config.getboolean('test_predictor')
    uniform = controller_config.getboolean('uniform')
    epsilon = float(controller_config['epsilon'])
    network_id = int(controller_config['network_id'])
    n_sumo = int(simulator_config['call_period'])
    beta = float(controller_config['beta'])
    ped_beta = float(controller_config['ped_beta'])
    hour = 0
    em_ratio_list = []
    em_ratio_list_dict = "./results/"+controller_config['params_path']+"/em_ratio.txt"
    ped_ratio_list = []
    ped_ratio_list_dict = "./results/"+controller_config['params_path']+"/ped_ratio.txt"
    true_em_ratio_list = []
    em_ratio_true_list_dict = "./results/"+controller_config['params_path']+"/em_ratio_true.txt"
    # gear_log_dict = "./results/"+controller_config['params_path']+"/gear_log.txt"
    # gear_log = []
    exp_dir = simulator_config['simProjDir']
    
    gamma = float(controller_config['gamma'])
    ita = float(controller_config['ita'])
    rho = float(controller_config['rho'])
    em_re_list = []
    ssm_cane = Ssm_cane()
    filling_bound_v = 1
    filling_bound_s = 0
    
    flow_gear = [[], []]
    speed_gear = [[], []]

    
    close_time = {}
    open_time = {}
    
    log_dict = "./results/"+controller_config['params_path']+"/my_log.log"
    
    logging.basicConfig(filename=log_dict, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    md_lambda = np.array([0.0 for _ in range(num_agents)])
    md_lambda_list = []
    # ped_md_lambda = np.array([0.0 for _ in range(num_agents)])
    # ped_md_lambda_list = []
    
    volume_diff = []
    speed_diff = []
    ssm_link = []
    volume_sumo_pre = []
    speed_sumo_pre = []

    agent_re_list = []
    patterns = []

    agent_re_list_true = []
    patterns_true = []
    delta_state_true = []
    action_list = []
    policy_probs_list = []
    delta_state = []
    step = 0
    
    prediction = initial_row_prediction[0,:,0,:]
    prediction_fu_list = []
    prediction_fu_list2 = []
    step_pre = 0
        
    #--------------ssm_requred---------------- 
    em_head_columns = pd.read_csv(data_dict).columns
    
    # em_df = sc.Tools.em_convert_tool(em_dict, counted_em_dict_original_true)
    em_df = pd.read_csv(em_lst_path)
    true_em = process_em(em_df, em_head_columns)  

    env = gym.make(controller_config['env_name'], data_dict = data_dict, num_features=num_features, num_agents = num_agents, network_id = network_id, filling_bound = filling_bound_v)
    
    env2 = gym.make(controller_config['env_name'], data_dict = data_dict2, num_features=num_features, num_agents = num_agents, network_id = network_id, filling_bound = filling_bound_s)
    
    # env3 = gym.make(controller_config['env_name'], data_dict = data_dict3, num_features=num_features, num_actions=num_actions, num_agents = num_agents, network_id = network_id, filling_bound = 0)
        
    env3 = gym.make(controller_config['env_name'], data_dict = pedestrian_dict, num_features=num_features, num_agents = num_agents, network_id = network_id, filling_bound = 0)
    
    # env3.clip_data()
    
    car_type_list = ['pas', 'pas2', 'bus', 'truck']
    car_envs = []
    car_data_paths = {}
    
    # TODO: here
    for car_type in car_type_list:
        car_data_paths[car_type] = car_type_dict + car_type + ".csv"
    car_env = gym.make('CustomEnvGroup-v0', data_paths = car_data_paths, num_features=num_features, num_agents = num_agents, network_id = network_id, filling_bound = 0)

        
    em_lookingtable = []
    
    column_names = env.save_file[1]
    vo_diff_dict = "./results/"+controller_config['params_path']+"/volume_difference.csv"
    speed_diff_dict = "./results/"+controller_config['params_path']+"/speed_difference.csv"
    
    vo_sumo_pre_dict = "./results/"+controller_config['params_path']+"/volume_sumo_pre.csv"
    speed_sumo_pre_dict = "./results/"+controller_config['params_path']+"/speed_sumo_pre.csv"
    def save_difference(vo_diff, sp_diff, ssm_link, vo_sumo_pre, speed_sumo_pre):
        
        volume_buffer = np.array(vo_diff)
        # assert self.obs_buffer.shape[1] == df.shape[1] - 1
        new_df = pd.DataFrame(volume_buffer, columns=['Interval'] + column_names)
        new_df.to_csv(vo_diff_dict, index=False)
        
        speed_buffer = np.array(sp_diff)
        # assert self.obs_buffer.shape[1] == df.shape[1] - 1
        new_df = pd.DataFrame(speed_buffer, columns=['Interval'] + column_names)
        new_df.to_csv(speed_diff_dict, index=False)
        
        ssm_buffer = np.array(ssm_link)
        # assert self.obs_buffer.shape[1] == df.shape[1] - 1
        new_df = pd.DataFrame(ssm_buffer, columns=['Interval'] + column_names)
        new_df.to_csv(counted_em_dict+f".csv", index=False)
        
        volum_sumo_buffer = np.array(vo_sumo_pre)
        # assert self.obs_buffer.shape[1] == df.shape[1] - 1
        new_df = pd.DataFrame(volum_sumo_buffer, columns=['Interval'] + column_names)
        new_df.to_csv(vo_sumo_pre_dict, index=False)
        
        speed_sumo_buffer = np.array(speed_sumo_pre)
        # assert self.obs_buffer.shape[1] == df.shape[1] - 1
        new_df = pd.DataFrame(speed_sumo_buffer, columns=['Interval'] + column_names)
        new_df.to_csv(speed_sumo_pre_dict, index=False)
    
    # initilization of simulator
    em_output, sumo_volume, sumo_speed, sumo_count, close_time, open_time = call_sumo_30min(exp_dir=exp_dir, hour = 0)
    initial_em = process_em(em_output, em_head_columns)
    sumo_volume = fill_na_sumo_data(sumo_volume, filling_bound_v)
    sumo_speed = fill_na_sumo_data(sumo_speed, filling_bound_s)
    volume_sumo_pre.append([step]+sumo_volume[step].tolist())
    speed_sumo_pre.append([step]+sumo_speed[step].tolist())
                
    agent_action_shape = env.action_space.nvec
    policies = [DiscretePolicy(num_choices, seed) for num_choices in agent_action_shape]
    # Test the reset function
    print("Reset Function:")
    
    # Sample random actions for each agent
    # initial_actions = env.action_space.sample()
    processed_em = initial_em.to_numpy(dtype=np.float32)[step,1:]
    ssm_link.append(initial_em.to_numpy(dtype=np.float32)[step])
    env.update_cone(np.array(create_mask_list(processed_em, ssm_cane.get_list())))
    
    initial_actions = [policy.sample_epsilon_action(epsilon) for policy in policies]
    md_lambda_list.append(md_lambda.tolist())
    # ped_md_lambda_list.append(ped_md_lambda.tolist())
    
    em_state = env.get_state_pre(processed_em)
    em_obs = env.get_obs(em_state[1], initial_actions)
    logging.info(f'Step {step}: Obs is {em_obs[0].tolist()}')
    logging.info(f'Step {step}: EM is {initial_em.to_numpy(dtype=np.float32)[step, 1:].tolist()}')
    em_ratio = np.sum(np.maximum(em_obs[0], 0))/np.maximum(np.sum(em_state[0]), 0.01)
    em_ratio_list.append(em_ratio.tolist())
    with open(em_ratio_list_dict, "w") as file:
        for item in em_ratio_list:
            file.write(str(item) + "\n")
    logging.info(f'Step {step}: Ratio is {em_ratio.tolist()}')

    # ped raion
    processed_ped = env3.loaded_data[env3.time]
    ped_state = env.get_state_pre(processed_ped)
    ped_obs = env.get_obs(ped_state[1], initial_actions)
    ped_ratio = np.sum(np.maximum(ped_obs[0], 0))/np.maximum(np.sum(ped_state[0]), 0.01)
    ped_ratio_list.append(ped_ratio.tolist())
    with open(ped_ratio_list_dict, "w") as file:
        for item in ped_ratio_list:
            file.write(str(item) + "\n")
    logging.info(f'Step {step}: Ratio is {ped_ratio.tolist()}')
    
    true_em_state = env.get_state_pre(true_em.to_numpy(dtype=np.float32)[step, 1:])
    true_em_obs = env.get_obs(true_em_state[1], initial_actions)
    true_em_ratio = np.sum(np.maximum(true_em_obs[0], 0))/np.maximum(np.sum(true_em_state[0]), 0.01)
    true_em_ratio_list.append(true_em_ratio.tolist())
    with open(em_ratio_true_list_dict, "w") as file:
        for item in true_em_ratio_list:
            file.write(str(item) + "\n")
        
    # initial_actions = [initial_actions[i] for i in range(env.num_agents)]
    observation = env.reset(actions = initial_actions)
    observation2 = env2.reset(actions = initial_actions)
    observation3 = env3.reset(actions = initial_actions)
    
    car_obs_list = []
    car_observation = car_env.reset(actions = initial_actions)

    car_obs_list.append(car_observation)
    
    env.update_fusion(observation[0])
    env.update_pre(observation[0])
    env2.update_fusion(observation2[0])
    env2.update_pre(observation2[0])
    env3.update_fusion(observation3[0])

    car_env.update_fusion(car_observation)
    car_env.update_pre(car_observation)

    em_lkt = [edges_pool[i] for i in range(len(observation[0])) if observation[0][i] != -1]
    em_lookingtable.append(em_lkt)
    done = env.done
    action_list.append(initial_actions)
    print("Actions:", initial_actions)
    # print(f"Initial Observation: {observation[0]}\n{observation[1]}\n{observation[2]}")

    agents_re = env.get_re_cd5_2(initial_actions)
    agent_re_list.append(agents_re)
    em_re_list.append([0])
    delta_state.append(env.get_state_pre((env.loaded_data[env.time]-env.loaded_data[env.time-1])/np.maximum(env.loaded_data[env.time-1],0.01))[0])
    
    agents_re_true = env.get_re_cd5_2(initial_actions)   
    agent_re_list_true.append(agents_re_true)
    delta_state_true.append(env.get_state_pre(env.loaded_data[env.time]-env.loaded_data[env.time-1])[0])


    re_sampled_list = []
    for i in range(num_agents):
        re_sampled_list.append(agents_re[i][initial_actions[i]].copy())
    patterns.append(re_sampled_list[0])


    re_sampled_list_true = []
    for i in range(num_agents):
        re_sampled_list_true.append(agents_re_true[i][initial_actions[i]].copy())
    patterns_true.append(re_sampled_list_true[0])
    
    policy_probs = []
    print("Action probs and re for agnets:")
    for agent, policy in enumerate(policies):
        policy_probs.append(policy.action_probs.tolist())
        print(f"Agent {agent}")
        print(f"Probs: {policy.action_probs.tolist()}")
        print(f"Re: {agents_re[agent]}")
    policy_probs_list.append(policy_probs)

    print("step", step)
    step += 1

    while step <= 59:
        start_time = time.time()
        
        processed_em = initial_em.to_numpy(dtype=np.float32)[step, 1:]
        ssm_link.append(initial_em.to_numpy(dtype=np.float32)[step])
        env.update_cone(np.array(create_mask_list(processed_em, ssm_cane.get_list())))
        
        actions = [policy.sample_epsilon_action(epsilon) for policy in policies]
        
        em_state = env.get_state_pre(initial_em.to_numpy(dtype=np.float32)[step, 1:])
        em_obs = env.get_obs(em_state[1], actions)
        logging.info(f'Step {step}: Obs is {em_obs[0].tolist()}')
        logging.info(f'Step {step}: EM is {initial_em.to_numpy(dtype=np.float32).tolist()}')
        #emission ratio
        em_ratio = np.sum(np.maximum(em_obs[0], 0))/np.maximum(np.sum(em_state[0]), 0.01)
        em_ratio_list.append(em_ratio.tolist())

        with open(em_ratio_list_dict, "w") as file:
            for item in em_ratio_list:
                file.write(str(item) + "\n")
        logging.info(f'Step {step}: Ratio is {em_ratio.tolist()}')

        # pedestrain ratio
        # ped_obs
        # ped_state
        # processed_ped has to be float 32 numpy
        processed_ped = env3.loaded_data[env3.time]
        ped_state = env.get_state_pre(processed_ped)
        ped_obs = env.get_obs(ped_state[1], actions)
        # 939-948 copy
        ped_ratio = np.sum(np.maximum(ped_obs[0], 0))/np.maximum(np.sum(ped_state[0]), 0.01)
        ped_ratio_list.append(ped_ratio.tolist())
        with open(ped_ratio_list_dict, "w") as file:
            for item in ped_ratio_list:
                file.write(str(item) + "\n")
        logging.info(f'Step {step}: Ratio is {ped_ratio.tolist()}')
        
        true_em_state = env.get_state_pre(true_em.to_numpy(dtype=np.float32)[step, 1:])
        true_em_obs = env.get_obs(true_em_state[1], actions)
        true_em_ratio = np.sum(np.maximum(true_em_obs[0], 0))/np.maximum(np.sum(true_em_state[0]), 0.01)
        true_em_ratio_list.append(true_em_ratio.tolist())
        with open(em_ratio_true_list_dict, "w") as file:
            for item in true_em_ratio_list:
                file.write(str(item) + "\n")
        
        next_observation, rewards, done, _ = env.step_f(actions)
        next_observation2, _, _, _ = env2.step_f(actions)
        next_observation3, _, _, _ = env3.step_f(actions)

        car_observation, _, _, _ = car_env.step_f(actions = actions)
        car_obs_list.append(car_observation)
            
        
        env.update_fusion(next_observation[0])
        env.update_pre(next_observation[0])
        env2.update_fusion(next_observation2[0])
        env2.update_pre(next_observation2[0])
        env3.update_fusion(next_observation3[0])
        car_env.update_fusion(car_observation)
        car_env.update_pre(car_observation) 
        
        em_lkt = [edges_pool[i] for i in range(len(next_observation[0])) if next_observation[0][i] != -1]
        em_lookingtable.append(em_lkt)
        action_list.append(actions)
        print("Actions:", actions)
        
        volume_sumo_pre.append([step]+sumo_volume[step].tolist())
        speed_sumo_pre.append([step]+sumo_speed[step].tolist())

        agents_re = env.get_re_cd5_2(actions)
        agent_re_list.append(agents_re)
        em_re_list.append([0])
        delta_state.append(env.get_state_pre((env.get_state_pre((env.loaded_data[env.time]-env.loaded_data[env.time-1])/np.maximum(env.loaded_data[env.time-1],0.01))[0]))[0])

        agents_re_true = env.get_re_cd5_2(actions)
        agent_re_list_true.append(agents_re_true)
        delta_state_true.append(env.get_state_pre(env.loaded_data[env.time]-env.loaded_data[env.time-1])[0])

        re_sampled_list = []
        for i in range(num_agents):
            re_sampled_list.append(agents_re[i][actions[i]].copy())
        patterns.append(re_sampled_list[0])

        re_sampled_list_true = []
        for i in range(num_agents):
            re_sampled_list_true.append(agents_re_true[i][actions[i]].copy())
        patterns_true.append(re_sampled_list_true[0])
        
        md_lambda_list.append(md_lambda.tolist())
        # ped_md_lambda_list.append(ped_md_lambda.tolist())

        policy_probs = []
        print("Action probs and re for agnets:")
        for agent, policy in enumerate(policies):
            policy_probs.append(policy.action_probs.tolist())
            print(f"Agent {agent}")
            print(f"Probs: {policy.action_probs.tolist()}")
            print(f"Re: {agents_re[agent]}")
        policy_probs_list.append(policy_probs)

        print("step", step)
        step += 1
        end_time = time.time()
        iteration_duration = end_time - start_time
        print(f"Time: {iteration_duration}")
    buffer_state = step +2
    volume_diff.append([hour]+[-1]*len(next_observation[0]))
    speed_diff.append([hour]+[-1]*len(next_observation[0]))
    hour += 1
    
    #-----------------calibrator debugging----------------
    
    car_update_dict = {}
    look_back_time = 60
    for car_type in car_type_list:
        car_update_matrix = [car_env.transform_state_matrix_to_lst(entry[car_type]) for entry in car_obs_list[-look_back_time:]]
        car_update_dict[car_type] = np.array(car_update_matrix)
    # skip all info about 10 entry edges
    append_vtype_skip_unbusy_road_5min_agg(file_path = simProjDir, car_obs_dict=car_update_dict, edge_names=list(em_head_columns)[1:], base_time=step-look_back_time)
    
    #-----------------calibrator debugging----------------

    em_output, sumo_volume, sumo_speed, sumo_count, close_time, open_time = call_sumo_30min(exp_dir=exp_dir, hour = 0)
    em = process_em(em_output, em_head_columns)
    sumo_volume = fill_na_sumo_data(sumo_volume, filling_bound_v)
    sumo_speed = fill_na_sumo_data(sumo_speed, filling_bound_s)
    
    
    while not done:
        start_time = time.time()
        # Step function
        print("\nTesting Step Function:")
        processed_em = em.to_numpy(dtype=np.float32)[step, 1:]
        ssm_link.append(em.to_numpy(dtype=np.float32)[step])
        ssm_cane.add_elements(np.where(processed_em >= bar)[0])
        env.update_cone(np.array(create_mask_list(processed_em, ssm_cane.get_list())))
        
        actions = [policy.sample_epsilon_action(epsilon) for policy in policies]
        
        em_state = env.get_state_pre(em.to_numpy(dtype=np.float32)[step, 1:])
        em_obs = env.get_obs(em_state[1], actions)
        logging.info(f'Step {step}: Obs is {em_obs[0].tolist()}')
        logging.info(f'Step {step}: EM is {em.to_numpy(dtype=np.float32)[step, 1:].tolist()}')
        em_ratio = np.sum(np.maximum(em_obs[0], 0))/np.maximum(np.sum(em_state[0]), 0.01)
        em_ratio_list.append(em_ratio.tolist())
        with open(em_ratio_list_dict, "w") as file:
            for item in em_ratio_list:
                file.write(str(item) + "\n")
        logging.info(f'Step {step}: Ratio is {em_ratio.tolist()}')

        # ped raion
        # processed_ped = env3.loaded_data[env3.time]
        ped_state = env.get_state_pre(processed_ped)
        ped_obs = env.get_obs(ped_state[1], actions)
        ped_ratio = np.sum(np.maximum(ped_obs[0], 0))/np.maximum(np.sum(ped_state[0]), 0.01)
        ped_ratio_list.append(ped_ratio.tolist())
        with open(ped_ratio_list_dict, "w") as file:
            for item in ped_ratio_list:
                file.write(str(item) + "\n")
        logging.info(f'Step {step}: Ratio is {ped_ratio.tolist()}')
        
        true_em_state = env.get_state_pre(true_em.to_numpy(dtype=np.float32)[step, 1:])
        true_em_obs = env.get_obs(true_em_state[1], actions)
        true_em_ratio = np.sum(np.maximum(true_em_obs[0], 0))/np.maximum(np.sum(true_em_state[0]), 0.01)
        true_em_ratio_list.append(true_em_ratio.tolist())
        with open(em_ratio_true_list_dict, "w") as file:
            for item in true_em_ratio_list:
                file.write(str(item) + "\n")
        
        # actions = [actions[i] for i in range(env.num_agents)]
        next_observation, rewards, done, _ = env.step_f(actions)
        next_observation2, _, _, _ = env2.step_f(actions)
        next_observation3, _, _, _ = env3.step_f(actions)
        
        # cur_obs_cam = next_observation[2]
        # print('>>> next_observation cur_obs_cam', cur_obs_cam)
        # check_symetric(cur_obs_cam)

        # car_observations = []
        # for car_env in car_envs:
        #     car_observations.append(car_env.step_f(actions = actions)[0])
        # car_obs_list.append(car_observations)
        
        car_observation, _, _, _ = car_env.step_f(actions = actions)
        car_obs_list.append(car_observation)
        
        em_lkt = [edges_pool[i] for i in range(len(next_observation[0])) if next_observation[0][i] != -1]
        em_lookingtable.append(em_lkt)
        action_list.append(actions)
        print("Actions:", actions)
        # print(f"Next Observation: {next_observation[0]}\n{next_observation[1]}\n{next_observation[2]}")
        volume_sumo_pre.append([step]+sumo_volume[step].tolist())
        speed_sumo_pre.append([step]+sumo_speed[step].tolist())

        # 5 min interval
        if step % 30 == 0 and step // 30 > 1:
            flow = env.mean_true(n_sumo)
            # ssm_flow = 
            flow_sumo = [
                flow[102], flow[103], flow[106], flow[107], flow[108],
                flow[109], flow[110], flow[112], flow[113], flow[116]
            ]
            speed_sumo = env2.mean_fu(n_sumo)
            
            flow_gear[0] = flow_gear[1]
            flow_gear[1] = flow_sumo
            speed_gear[0] = speed_gear[1]
            speed_gear[1] = speed_sumo
            if step % 60 == 0 and step // 60 > 1:
                car_update_dict = {}
                look_back_time = 60
                # look_back_time = 30
                for car_type in car_type_list:
                    car_update_matrix = [car_env.transform_state_matrix_to_lst(entry[car_type]) for entry in car_obs_list[-look_back_time:]]
                    car_update_dict[car_type] = np.array(car_update_matrix)
                append_vtype_skip_unbusy_road_5min_agg(file_path = simProjDir, car_obs_dict=car_update_dict, edge_names=list(em_head_columns)[1:], base_time=step-look_back_time)

                em_output, sumo_volume, sumo_speed, sumo_count, close_time, open_time =call_sumo_30min(exp_dir=exp_dir, flow=flow_gear, speed=speed_gear, hour=hour, close_time = close_time, open_time = open_time, car_obs = car_obs_list)
                em = process_em(em_output, em_head_columns)
                sumo_volume = fill_na_sumo_data(sumo_volume, filling_bound_v)
                sumo_speed = fill_na_sumo_data(sumo_speed, filling_bound_s)
            
            fu_flow = env.mean_fu(n_sumo)
            differences = get_diff(sumo_volume, sumo_speed, fu_flow, speed_sumo, ssm_cane.get_list(), len(env.cane_buffer), filling_bound_v, filling_bound_s, n_sumo)
            
            volume_diff.append([hour]+differences[0])
            speed_diff.append([hour]+differences[1])
            
            ssm_cane.empty_list()
            hour += 1
        if step % 5 == 0 and prediction_fu_list!=[] and step_pre < initial_row_prediction.shape[0]-1:
            step_pre+=1
            pre_input = np.mean(np.array(prediction_fu_list), axis=0)
            pre_input2 = np.mean(np.array(prediction_fu_list2), axis=0)
            pre_output = get_prediction(pre_input, pre_input2, step_pre)
            # TODO: lookahead step can be modified
            prediction = pre_output[step_pre,:,0,:]
            
            prediction_fu_list = []
            prediction_fu_list2 = []
            
        # TODO: simulation update and call neeeded!!!
        
        if test_predictor:
            prediction_fu = prediction[:,0]
            prediction_fu2 = prediction[:,1]
        else:
            prediction_fu = np.maximum(np.where(next_observation[0] != -1, next_observation[0], prediction[:,0]), 1)
            prediction_fu2 = np.maximum(np.where(next_observation2[0] != -1, next_observation2[0], prediction[:,1]), 0)
        
        env.update_fusion(prediction_fu)
        env.update_pre(prediction[:,0])
        prediction_fu_list.append(prediction_fu.tolist())
        env2.update_fusion(prediction_fu2)
        env2.update_pre(prediction[:,1])
        env3.update_fusion(next_observation3[0])
        car_env.update_fusion(car_observation)
        car_env.update_pre(car_observation) 
        prediction_fu_list2.append(prediction_fu2.tolist())

        if (step>=buffer_state):
            action_update = action_list[-5:]
            agents_re = env.get_re_pre_cd5_3(action_update, prediction[:,0])
            processed_ped = env3.loaded_data[env3.time]
            inhaled_em = np.multiply(processed_em, processed_ped)
            # print('>>> inhaled_em',inhaled_em.shape,inhaled_em)
            # em_re = env.get_re_pre_md(action_update, processed_em, beta)
            em_re = env.get_re_pre_md(action_update, inhaled_em, beta)
            # processed_ped = next_observation3[0]
            ped_re = env3.get_re_pre_md(action_update, processed_ped, ped_beta)
            
            agents_re_true = env.get_re_cd5_2(actions)
            agent_re_list.append(agents_re)
            agent_re_list_true.append(agents_re_true)
            em_re_list.append(em_re)
            delta_state.append((env.get_state_pre((np.maximum(prediction_fu, 0)-np.maximum(env.fusion_buffer[-2][1:], 0))/np.maximum(env.fusion_buffer[-2][1:], 0.01)))[0])
            delta_state_true.append((env.get_state_pre((env.loaded_data[env.time]-env.loaded_data[env.time-1])/np.maximum(env.loaded_data[env.time-1],0.01))[0]))
        elif (step>=buffer_state):
            agents_re = env.get_re_cd5_2(actions)
            agent_re_list.append(agents_re)
        else:
            agents_re = env.get_re_cd5_2(actions)
            agent_re_list.append(agents_re)
            delta_state.append(env.get_state_pre((env.get_state_pre((env.loaded_data[env.time]-env.loaded_data[env.time-1])/np.maximum(env.loaded_data[env.time-1],0.01))[0]))[0])
            
            agents_re_true = env.get_re_cd5_2(actions)
            agent_re_list_true.append(agents_re_true)
            delta_state_true.append(env.get_state_pre(env.loaded_data[env.time]-env.loaded_data[env.time-1])[0])

        if step % 5 == 0 and prediction_fu_list!=[] and step_pre < initial_row_prediction.shape[0]-1:
            if not(uniform) and (step>=buffer_state):
                # update_policies(policies, agents_re)
                md_lambda = update_policies_md(policies, agents_re, em_re, md_lambda, rho)
            
            
        md_lambda_list.append(md_lambda.tolist())
        # ped_md_lambda_list.append(ped_md_lambda.tolist())

        re_sampled_list = []
        for i in range(num_agents):
            re_sampled_list.append(agents_re[i][actions[i]])
        patterns.append(re_sampled_list)


        re_sampled_list_true = []
        for i in range(num_agents):
            re_sampled_list_true.append(agents_re_true[i][actions[i]])
        patterns_true.append(re_sampled_list_true)

        policy_probs = []
        print("Action probs and re for agnets:")
        for agent, policy in enumerate(policies):
            policy_probs.append(policy.action_probs.tolist())
            print(f"Agent {agent}")
            print(f"Probs: {policy.action_probs.tolist()}")
            print(f"Re: {agents_re[agent]}")
        policy_probs_list.append(policy_probs)
        # print("Rewards:", rewards)
        # print("Done Flag:", done)
        print("step", step)
        step += 1
        
        end_time = time.time()
        iteration_duration = end_time - start_time
        print(f"Time: {iteration_duration}")
    
        env.save_buffer(save_dict)
        env2.save_buffer(save_dict2)
        np.save(action_list_dict, action_list)
        
        env.save_buffer_fusion(save_fu_dict)
        env.save_buffer_predict(save_predict_dict)
        env.save_buffer_cane(save_cane_dict)
        env2.save_buffer_fusion(save_fu_dict2)
        env2.save_buffer_predict(save_predict_dict2)
        with open(re_save_dict_true, 'w') as json_file:
            json.dump(agent_re_list_true, json_file)
        with open(re_save_dict, 'w') as json_file:
            json.dump(agent_re_list, json_file)
        with open(em_re_save_dict, 'w') as json_file:
            json.dump(em_re_list, json_file)
        np.save(delta_true_dict, np.array(delta_state_true))
        np.save(delta_dict, np.array(delta_state))
        np.save(md_lambda_list_dict, np.array(md_lambda_list))
        # np.save(ped_md_lambda_list_dict, np.array(ped_md_lambda_list))
        em.to_csv(counted_em_dict+f"{hour}.csv")
        true_em.to_csv(counted_em_dict_true)
        with open(close_time_dict, 'w') as f:
            json.dump(close_time, f)
        with open(open_time_dict, 'w') as f:
            json.dump(open_time, f)
        
        save_difference(volume_diff, speed_diff, ssm_link, volume_sumo_pre, speed_sumo_pre)
    else:
        with open(re_save_dict, 'w') as json_file:
            json.dump(agent_re_list, json_file)
        np.save(delta_dict, np.array(delta_state))

    with open(policy_dict, 'w') as json_file:
        json.dump(policy_probs_list, json_file)

    # Close the environment when done
    env.close()
    env2.close()
    logging.shutdown()
    # print(lane_flags)

if __name__ == "__main__":
    start_time = time.time()  # Record start time
    camera_control()  # Run your function
    end_time = time.time()  # Record end time

    execution_time = end_time - start_time  # Compute execution time
    print(f"Execution time: {execution_time:.6f} seconds")