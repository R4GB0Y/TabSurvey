import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    print("Loading dataset " + args.dataset + "...")

    if args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        # X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

    elif args.dataset == "KddCup99":  # Multi-class classification dataset with categorical data
        X, y = sklearn.datasets.fetch_kddcup99(return_X_y=True)
        X, y = X[:10000, :], y[:10000]  # only take 10000 samples from dataset

        # filter out all target classes, that occur less than 1%
        target_counts = np.unique(y, return_counts=True)
        smaller1 = int(X.shape[0] * 0.01)
        small_idx = np.where(target_counts[1] < smaller1)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)

        '''
        # filter out all target classes, that occur less than 100
        target_counts = np.unique(y, return_counts=True)
        small_idx = np.where(target_counts[1] < 100)
        small_tar = target_counts[0][small_idx]
        for tar in small_tar:
            idx = np.where(y == tar)
            y[idx] = b"others"

        # new_target_counts = np.unique(y, return_counts=True)
        # print(new_target_counts)
        '''
    elif args.dataset == "Adult" or args.dataset == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                                ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        X = df[features].to_numpy()
        y = df[label].to_numpy()

    elif args.dataset == "HIGGS":  # Binary classification dataset with one categorical feature
        path = "/opt/notebooks/data/HIGGS.csv.gz"
        df = pd.read_csv(path, header=None)
        df.columns = ['x' + str(i) for i in range(df.shape[1])]
        num_col = list(df.drop(['x0', 'x21'], 1).columns)
        cat_col = ['x21']
        label_col = 'x0'

        def fe(x):
            if x > 2:
                return 1
            elif x > 1:
                return 0
            else:
                return 2

        df.x21 = df.x21.apply(fe)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)

        X = df[num_col + cat_col].to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Heloc":  # Binary classification dataset without categorical data
        path = "heloc_cleaned.csv"  # Missing values already filtered
        df = pd.read_csv(path)
        label_col = 'RiskPerformance'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    # our dataset, first test on binary classification using only network data
    # network binary data
    elif args.dataset == "super_net_bin":  # Binary classification dataset with categorical data
        path_data = '/home/van0/Desktop/Benchmark/merg/ready_data/Linear_interpolation/super_net_bin.csv'

        features = ['sport', 'dport', 'flags', 'size', 'n_pkt_src', 'n_pkt_dst', 'modbus_response',
                    'ip_s_int', 'ip_d_int', 'mac_s_int', 'mac_d_int', 'modbus_function', 'protocol']
        label = "Label_n"
        #columns = features + [label]
        #df = pd.read_csv(path_data, names=columns)
        df = pd.read_csv(path_data)
        '''
        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                                ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        '''
        X = df[features].to_numpy()
        y = df[label].to_numpy()
    # physical binary data
    elif args.dataset == "super_phy_bin":  # Binary classification super dataset
        path_data = '/home/van0/Desktop/Benchmark/merg/ready_data/Linear_interpolation/super_phy_bin.csv'

        features = ['Tank_1', 'Tank_2', 'Tank_3', 'Tank_4', 'Tank_5', 'Tank_6', 'Tank_7', 'Tank_8',
                    'Pump_1', 'Pump_2', 'Pump_3', 'Pump_4', 'Pump_5', 'Pump_6', 'Flow_sensor_1', 'Flow_sensor_2',
                    'Flow_sensor_3', 'Flow_sensor_4', 'Valv_1', 'Valv_2', 'Valv_3', 'Valv_4', 'Valv_5', 'Valv_6',
                    'Valv_7', 'Valv_8', 'Valv_9', 'Valv_10', 'Valv_11', 'Valv_12', 'Valv_13', 'Valv_14', 'Valv_15',
                    'Valv_16', 'Valv_17', 'Valv_18', 'Valv_19', 'Valv_20', 'Valv_21', 'Valv_22']

        label = 'Label_n'

        df = pd.read_csv(path_data)

        X = df[features].to_numpy()
        y = df[label].to_numpy()
    # merged binary data
    elif args.dataset == "super_bin":  # Binary classification super dataset
        path_data = '/home/van0/Desktop/Benchmark/merg/ready_data/Linear_interpolation/super_bin.csv'

        features = ['sport', 'dport', 'flags', 'size', 'n_pkt_src', 'n_pkt_dst', 'modbus_response',
                    'ip_s_int', 'ip_d_int', 'mac_s_int', 'mac_d_int', 'modbus_function', 'protocol',
                    'Tank_1', 'Tank_2', 'Tank_3', 'Tank_4', 'Tank_5', 'Tank_6', 'Tank_7', 'Tank_8',
                    'Pump_1', 'Pump_2', 'Pump_3', 'Pump_4', 'Pump_5', 'Pump_6', 'Flow_sensor_1', 'Flow_sensor_2',
                    'Flow_sensor_3', 'Flow_sensor_4', 'Valv_1', 'Valv_2', 'Valv_3', 'Valv_4', 'Valv_5', 'Valv_6',
                    'Valv_7', 'Valv_8', 'Valv_9', 'Valv_10', 'Valv_11', 'Valv_12', 'Valv_13', 'Valv_14', 'Valv_15',
                    'Valv_16', 'Valv_17', 'Valv_18', 'Valv_19', 'Valv_20', 'Valv_21', 'Valv_22']

        label = 'Label_n'

        df = pd.read_csv(path_data)

        X = df[features].to_numpy()
        y = df[label].to_numpy()
    # merged super binary embedded data, no interpolation
    elif args.dataset == "super_embd":  # Binary classification super embedded dataset
        path_data = '/home/van0/Desktop/Work_dir/super_embd.csv'

        features = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                    '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43',
                    '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64',
                    '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85',
                    '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
                    '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123',
                    '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141',
                    '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                    '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177',
                    '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195',
                    '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213',
                    '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231',
                    '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249',
                    '250', '251', '252', '253', '254', '255', 'Tank_1', 'Tank_2', 'Tank_3', 'Tank_4', 'Tank_5', 'Tank_6', 'Tank_7', 'Tank_8',
                    'Pump_1', 'Pump_2', 'Pump_3', 'Pump_4', 'Pump_5', 'Pump_6', 'Flow_sensor_1', 'Flow_sensor_2', 'Flow_sensor_3', 'Flow_sensor_4',
                    'Valv_1', 'Valv_2', 'Valv_3', 'Valv_4', 'Valv_5', 'Valv_6', 'Valv_7', 'Valv_8', 'Valv_9', 'Valv_10', 'Valv_11', 'Valv_12',
                    'Valv_13', 'Valv_14', 'Valv_15', 'Valv_16', 'Valv_17', 'Valv_18', 'Valv_19', 'Valv_20', 'Valv_21', 'Valv_22']

        label = 'Label'

        df = pd.read_csv(path_data)

        X = df[features].to_numpy()
        y = df[label].to_numpy()

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print("Having", args.num_classes, "classes as target.")

    num_idx = []
    args.cat_dims = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)
        print("New Shape:", X.shape)

    return X, y


