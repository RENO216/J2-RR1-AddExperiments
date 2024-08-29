from __init__ import np, sio, json, os, pywarraychannels

class commSys:
    def __init__(self, data_dir, Dataset_id):
        link = "up"             # Whether it's up-link or down-link
        index = 0               # Complementary to the dataset, you should be able to eliminate this variable if you're loading different data
            # Noise related
        T = 15                  # C
        k_B = 1.38064852e-23    # Boltzmanz's constant
            # Speed of light
        

        orientations_UE = [
            pywarraychannels.uncertainties.Static(tilt=-np.pi/2),
            pywarraychannels.uncertainties.Static(tilt=np.pi/2),
            pywarraychannels.uncertainties.Static(roll=np.pi/2),
            pywarraychannels.uncertainties.Static(roll=-np.pi/2)
            ]
        orientations_AP = [
            orientations_UE[3]
            ]
        Simple_U = False        # Wether to apply SVD reduction to the measurements
        N_est = 5

        p_t_dBm = 40
        f_c = 73                # GHz
        B = 1                   # GHz
        K = 64                  # 
            
        with open(f"{data_dir}/ds{Dataset_id}/AP_pos.txt" ) as f:
            AP_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
        self.AP_pos = np.squeeze(np.array(AP_pos_all))
        with open(f"{data_dir}/ds{Dataset_id}/UE_pos.txt" ) as f:
            self.UE_pos_all = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
        with open(f"{data_dir}/ds{Dataset_id}/AP_selected.txt" ) as f:
            AP_selected_all = [int(a) for a in f.read().split("\n")[1].split()]
        with open(f"{data_dir}/ds{Dataset_id}/Info_selected.txt" ) as f:
            self.Rays_all = [pywarraychannels.em.Geometric([[float(p) for p in line.split(' ')] for line in ue_block.split("\n")], bool_flip_RXTX=link=="up") for ue_block in f.read()[:-1].split('\n<ue>\n')]
        
        
        self.chan_ids = np.squeeze(sio.loadmat(f'../Dataset/StrongestChanID_Set{Dataset_id}.mat')['chan_ids'] - 1)
        self.lane_y = np.unique(np.array(self.UE_pos_all)[:, 1])
        
        
        
    def loadResults_MOMPdropDoA(self, data_dir, Dataset_id, dropDoA=1):
        N_est = 5
        c = 299792458                 # m/s
        p_t_dBm = 40
        if not dropDoA: 
            N_UE = 4                # Number of UE antennas in each dimension
            N_AP = 8                # Number of AP antennas in each dimension
            N_RF_UE = 2             # Number of UE RF-chains in total
            N_RF_AP = 4             # Number of AP RF-chains in total
            N_M_UE = 4              # Number of UE measurements in each dimension
            N_M_AP = 8              # Number of AP measurements in each dimension
            K_res_lr = 2
            K_res = 64
            Q = 32                 # Length of the training pilot
        elif dropDoA:
            N_UE = 8                # Number of UE antennas in each dimension
            N_AP = 16                # Number of AP antennas in each dimension
            N_RF_UE = 4             # Number of UE RF-chains in total
            N_RF_AP = 8             # Number of AP RF-chains in total
            N_M_UE = 8              # Number of UE measurements in each dimension
            N_M_AP = 16              # Number of AP measurements in each dimension
            K_res_lr = 4 
            K_res = 128  
            Q = 64 
        folder_to_load = 'paths-retDoA' if dropDoA else 'paths-all'
        ue_number = 2000 if not (Dataset_id in [1, 2]) else 1996
        method = "MOMP"         # Channel estimation method (MOMP or OMP)
        with open(os.getcwd()+ f'{data_dir}/ds{Dataset_id}/{folder_to_load}/single_{method}_{N_M_UE}_{N_M_AP}_{int(p_t_dBm)}dBm_{10*K_res}_{ue_number}ue.json', 'r') as f:
            estimation = json.loads(f.read())
        locs_true, all_ests_aoas, all_ests_aods, all_ests_toas, P_ests =np.zeros([1,3]), np.zeros([1,3]), np.zeros([1,3]), [], [] 
        for est_id in range(len(estimation)):
            estimation_cur = estimation[est_id]
            all_ests_aoas = np.vstack([all_ests_aoas, np.array(estimation_cur['DoA'])])
            all_ests_aods = np.vstack([all_ests_aods, np.array(estimation_cur['DoD'])])
            all_ests_toas += estimation_cur['DDoF']
            P_ests += estimation_cur['Power']
            locs_true = np.vstack([locs_true, np.tile(np.array(self.UE_pos_all[est_id]), [N_est, 1])])
        self.locs_true, self.all_ests_aoas, self.all_ests_aods =locs_true[1:], all_ests_aoas[1:], all_ests_aods[1:]
        self.all_ests_toas = np.reshape(np.array(all_ests_toas) / c, [-1, 1])
        self.P_ests = np.reshape(np.array(P_ests) - p_t_dBm, [-1, 1]) if dropDoA else np.reshape(np.array(P_ests), [-1, 1])
        self.Num_inters_all = np.array(sio.loadmat(f'{data_dir}/ds{Dataset_id}/Num_inters.mat')['Num_inters'])
    
    def relate_ested_paths(ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas, ang_thresh=5, toa_thresh=0.2e-8):
        # ang_thresh: degree threshold to link an estimated path to the true path
        # final_id: the estimated path could be linked to any of the true paths; 
        # pass_id: the estimated path accuracy is within some range of the true paths
        ested_angs = np.hstack([ested_aoas, ested_aods])
        true_angs = np.hstack([true_aoas, true_aods])

        ang_diff_mat = np.dot(ested_angs, true_angs.T)
        toa_diff_mat = np.abs(np.reshape(ested_toas, [-1, 1]) - np.reshape(true_toas, [1, -1]))

        ang_id = ang_diff_mat.argmax(axis=1)

        ang_pass_id = ang_diff_mat.max(axis=1) >= np.cos(np.deg2rad(ang_thresh)) * 2
        toa_id = toa_diff_mat[range(ested_angs.shape[0]), ang_id] <= toa_thresh
        pass_id = ang_pass_id & toa_id
        
        final_id = ang_id[pass_id]

        avg_ang_err = [] # in degree

        if np.any(pass_id):
            ested_aoas = ested_aoas[pass_id, :]
            ested_aods = ested_aods[pass_id, :]
            ested_toas = ested_toas[pass_id]
            ested_toas = np.reshape(ested_toas, [-1, 1])       

            true_aoas = true_aoas[final_id, :]
            true_aods = true_aods[final_id, :]
            # true_toas = true_toas[toa_id[final_id]]
            true_toas = true_toas[final_id]
            true_toas = np.reshape(true_toas, [-1, 1])

            est_err_aod = np.rad2deg(np.arccos(np.diag(np.dot(ested_aods, true_aods.T))))
            avg_ang_err.append(np.mean(est_err_aod))
            est_err_aoa = np.rad2deg(np.arccos(np.diag(np.dot(ested_aoas, true_aoas.T))))
            avg_ang_err.append(np.mean(est_err_aoa))
            
            avg_ang_err = np.array(avg_ang_err)


            return final_id, pass_id, ested_aoas, ested_aods, ested_toas, true_aoas, true_aods, true_toas, avg_ang_err
        else:
            return None
