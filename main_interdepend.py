from src import *

print(device_lib.list_local_devices())
device_name = device_lib.list_local_devices()[-1].name

# Data processing
#TODO: specify the datasets
check_set = 'test'
PathNetBest = f'./src/models/PathNet/best_model.tf'
model = load_model('./src/models/PathNet/best_model.h5')


with tf.device(device_name):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=False)
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss= loss_obj, optimizer=opt, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

if (check_set == 'train'):
    Dataset_id_set = [1,2,5]
elif (check_set == 'test'):
    Dataset_id_set = [3,4]
pi = np.pi

# Perfect channel dataset
X, y = [], []
for set_id in Dataset_id_set:
    DS_all = np.array(pd.read_csv(f'../Dataset/DS_path_order_AbsAng_set{set_id}.csv', header = 0, engine='python'))
    p = DS_all[:, 0].copy()
    p_norm = my_data_normalize(p, raw_data_name='power')
    toa = DS_all[:, 2].copy()
    toa_norm = my_data_normalize(toa, raw_data_name='toa', std_fact=4)
    
    X_temp = np.zeros([len(DS_all), 6])  # P    ToA   AoA_phi   AoA_the   AoD_phi   AoD_the
    X_temp[:, 0] = p_norm
    # X[:, 1] = toa / np.max(DS_all[:, 2]) # for v1-v8
    X_temp[:, 1] = toa_norm # for >= v9
    X_temp[:, 2:] = DS_all[:, 3:7].copy() # angles
    X_temp[:, 2:]  /= (2 * np.pi)

    y_temp = DS_all[:, -4].copy().astype(np.int32)
    y_temp[y_temp >= 2] = np.int32(2)
    X.append(X_temp)
    y.append(y_temp)
X, y = np.concatenate(X, 0), np.concatenate(y, 0)

# loss_result, acc_result = model.evaluate(X, y)
# print(f'Overall accuracy: {acc_result: .4f}')
# class_acc = []
# for i in range(3):
#     y_id = y == i
#     loss, acc = model.evaluate(X[y_id, :],  y[y_id])
#     class_acc.append(acc)
# for i in range(3):
#     print(str("Accuracy of class "+str(i)+f": {class_acc[i]: .4f}"))

#  Estimated chanel
X_, y_ = [], []
for set_id in Dataset_id_set:
    cs = CommSys.commSys(data_dir='./src/data/', Dataset_id=set_id)
    cs.loadResults_MOMPdropDoA(data_dir='./src/data/', Dataset_id=set_id, dropDoA=1)
    
    X_tp = np.zeros([len(cs.all_ests_aoas), 6])
    X_tp[:, 0] = cs.P_ests[:, 0].copy()
    X_tp[:, 1] = cs.all_ests_toas[:, 0].copy()
    X_tp[:, 2] = (np.angle(cs.all_ests_aods[:, 0] + 1j * cs.all_ests_aods[:, 1]) + 2*pi) % (2*pi)
    X_tp[:, 3] = np.arccos(cs.all_ests_aods[:, 2])
    X_tp[:, 4] = (np.angle(cs.all_ests_aoas[:, 0] + 1j * cs.all_ests_aoas[:, 1]) + 2*pi) % (2*pi)
    X_tp[:, 5] = np.arccos(cs.all_ests_aoas[:, 2])
    
    for chan_ii in range(len(cs.Rays_all)):
        ray_info = cs.Rays_all[chan_ii].ray_info
        tru_ests_aods = np.concatenate([np.cos(ray_info[:, -1])*np.cos(ray_info[:, -2]), np.cos(ray_info[:, -1])*np.sin(ray_info[:, -2]), np.sin(ray_info[:, -1])], 1)
        tru_ests_aoas = np.concatenate([np.cos(ray_info[:, -3])*np.cos(ray_info[:, -4]), np.cos(ray_info[:, -3])*np.sin(ray_info[:, -4]), np.sin(ray_info[:, -3])], 1)
        tru_toas = ray_info[:, 1:2]

    X_tp[:, 0] = (X_tp[:, 0] - (- 130) ) / (6 * 13) - (-0.5)
    X_tp[:, 1] = (X_tp[:, 1] - (2e-7) ) / (4 * 1.4e-7) - (-0.3)

    for col_id in range(2, 6):
        X_tp[:, col_id] /= (2*pi)
    y_tp = cs.Num_inters_all.copy()
    # TODO: find corresonding true paths using cs.ested_paths_v2()
    
    y_tp[y_tp>=2] = 2
    
    X_.append(X_tp)
    y_.append(y_tp)
X_, y_ = np.concatenate(X_, 0), np.concatenate(y_, 0)
    
print(f'--------------Path order classifying on {device_name} ---------------->>>>')
inter_pred_prob = model.predict(X, verbose=0)
num_inter_pred = np.argmax(inter_pred_prob, axis=1)
    

    
    
    
    





print()
    