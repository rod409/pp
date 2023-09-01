import pickle

with open('kitti_infos_train.pkl', 'rb') as f:
    kitti = pickle.load(f)

with open('kitti_dbinfos_train.pkl', 'rb') as f:
    dbkitti = pickle.load(f)

with open('waymo_infos_train.pkl', 'rb') as f:
    waymo = pickle.load(f)

with open('waymo_dbinfos_train.pkl', 'rb') as f:
    dbwaymo = pickle.load(f)

print('done')