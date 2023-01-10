import argparse

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, required=True)
    parser.add_argument("-gpu", "--gpu_id", type=int, default=0)
    # model
    parser.add_argument("-mn", "--model_name", type=str, required=True)
    parser.add_argument("-mv", "--model_version", type=str, required=True)
    parser.add_argument("-m", "--mask", default=None, type=str)
    parser.add_argument("-s", "--size", default=None, type=int)
    # data
    parser.add_argument("-dn", "--dataset_name", type=str, default=None)
    parser.add_argument("-tp", "--train_path", type=str, default=None)
    parser.add_argument('-cont', '--continue_train', action='store_true')
    # train
    parser.add_argument("-fsp", "--fix_step", type=int, default=None)
    parser.add_argument("-sf", "--save_freq", type=int, default=None)
    parser.add_argument("-eps", "--epochs", type=int, default=None)

    return parser.parse_args()

def get_test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-gpu", "--gpu_id", type=int, default=0)
    # model
    parser.add_argument("-mn", "--model_name", type=str)
    parser.add_argument("-mv", "--model_version", type=str, required=True)
    parser.add_argument("-m", "--mask", default=None, type=str)
    parser.add_argument("-s", "--size", default=None, type=int)
    # test
    parser.add_argument("-me", "--model_epoch", type=int, default=-1)
    parser.add_argument("-as", "--anomaly_score", type=str, default='MSE', help='MSE | Mask_MSE')
    parser.add_argument("-t", "--test_type", type=str, default='normal', help='normal | position')
    parser.add_argument("-pn", "--pos_normalized", action='store_true', help='Use for typecplus')
    parser.add_argument("-mm", "--minmax", action='store_true', help='Use for combine supervised')
    parser.add_argument("-ur", "--using_record", action='store_true', help='Using conf and score file to testing, save time')
    parser.add_argument("-ut", "--using_threshold", action='store_true', help='Using fix th to testing, blind testing')
    parser.add_argument("-nn", "--normal_num", type=int, default=None, help='test normal num')
    parser.add_argument("-sn", "--smura_num", type=int, default=None, help='test smura num')
    # data
    parser.add_argument("-dn", "--dataset_name", type=str, default=None)
    parser.add_argument("-dp", "--dataset_path", type=str, default=None)
    parser.add_argument("-cp", "--csv_path", type=str, default=None)
    parser.add_argument("-np", "--normal_path", type=str, default=None)
    parser.add_argument("-sp", "--smura_path", type=str, default=None)
   
    return parser.parse_args()
