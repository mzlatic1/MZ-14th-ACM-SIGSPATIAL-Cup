from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import zipfile
import optuna
import os

num_cpus = 8
os.environ['LOKY_MAX_CPU_COUNT'] = str(num_cpus) # prevents optuna from crashing

# prep folder variables
parent_folder = r'C:\Users\marko\Documents\SIGSPATIAL'
parquet_folder = os.path.join(parent_folder, 'parquet_outputs')
output_folder = os.path.join(parent_folder, 'prod_outputs')

# read parquet files via pyarrow-enabled pandas dataframes (these were previously converted, the original file types were .csv)
city_a = pd.read_parquet(os.path.join(parquet_folder, 'city_A_challengedata.parquet'), engine='pyarrow').drop(columns=['geometry'])
city_b = pd.read_parquet(os.path.join(parquet_folder, 'city_B_challengedata.parquet'), engine='pyarrow').drop(columns=['geometry'])
city_c = pd.read_parquet(os.path.join(parquet_folder, 'city_C_challengedata.parquet'), engine='pyarrow').drop(columns=['geometry'])
city_d = pd.read_parquet(os.path.join(parquet_folder, 'city_D_challengedata.parquet'), engine='pyarrow').drop(columns=['geometry'])

city_letters = ['A', 'B', 'C', 'D']
for idx, city_df in enumerate([city_a, city_b, city_c, city_d]):
    print('Processing City: ', city_letters[idx])

    # load standard scalar object
    scalar = StandardScaler()

    expl_cols = ['uid', 'd', 't']
    labels = ['x', 'y']

    # cluster inputs via KMeans clustering
    n_clusters = 14
    kmeans_cols = [f'col{num}' for num in range(n_clusters)]
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

    # split, subset, then apply transformations
    parent_city_train = city_df.query('x != 999 and y != 999').sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
    city_train_kmeans = pd.DataFrame(kmeans.fit_transform(parent_city_train[expl_cols]), columns=kmeans_cols)
    city_train = pd.concat([city_train_kmeans, parent_city_train[labels]], axis=1)
    city_train = pd.DataFrame(scalar.fit_transform(city_train), columns=kmeans_cols + labels)

    parent_city_test = city_df.query('x == 999 and y == 999').sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
    city_test_kmeans = pd.DataFrame(kmeans.transform(parent_city_test[expl_cols]), columns=kmeans_cols)
    city_test = pd.concat([city_test_kmeans, parent_city_test[labels]], axis=1)
    city_test_transform = pd.DataFrame(scalar.transform(city_test), columns=kmeans_cols + labels)

    # hyperparameter tuning
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 5, 25)
        weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
        algorithm = trial.suggest_categorical("algorithm", ['ball_tree', 'kd_tree', 'brute', 'auto'])
        leaf_size = trial.suggest_int("leaf_size", 1, 60)
        p = trial.suggest_int("p", 1, 2)

        reg = KNeighborsRegressor(n_jobs=num_cpus)
        reg.n_neighbors = n_neighbors
        reg.weights = weights
        reg.algorithm = algorithm
        reg.leaf_size = leaf_size
        reg.p = p
        reg.metric = 'l1' if p == 1 else 'l2'

        reg.fit(city_train[kmeans_cols], city_train[labels])
        preds = reg.predict(city_test_transform[kmeans_cols])
        return root_mean_squared_error(city_test_transform[labels], preds)


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    best_params['metric'] = 'l1' if best_params['p'] == 1 else 'l2'
    print(best_params)

    # Run model with best parameters
    kneighbors_model = KNeighborsRegressor(**best_params)
    kneighbors_model.fit(city_train[kmeans_cols], city_train[labels])
    predicted_results = kneighbors_model.predict(city_test_transform[kmeans_cols])

    print('Exporting results...')
    # inverse transformations on predicted fields, then export results based on competition rules
    inverse_results = pd.DataFrame(scalar.inverse_transform(pd.concat([city_test_transform[kmeans_cols], pd.DataFrame(predicted_results, columns=labels)], axis=1), copy=True), columns=kmeans_cols + labels)
    rejoin = pd.concat([parent_city_test[expl_cols], inverse_results[labels]], axis=1)
    for col in expl_cols:
        rejoin[col] = rejoin[col].apply(lambda num: round(num)).astype(int)

    out_csv = os.path.join(output_folder, f'071_spat-alpha_city{city_letters[idx]}_humob25.csv')
    rejoin.to_csv(out_csv, index=False)
    with zipfile.ZipFile(out_csv + '.gz', 'w', zipfile.ZIP_DEFLATED) as zfile:
        zfile.write(out_csv, os.path.split(out_csv)[1])
