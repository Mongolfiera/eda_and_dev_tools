import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def main():

    RANDOM_STATE = 42
    TEST_SIZE = 0.25
    url = "https://raw.githubusercontent.com/aiedu-courses/eda_and_dev_tools/refs/heads/main/datasets/online_shoppers_intention.csv"

    df = pd.read_csv(url)
    df.drop_duplicates(inplace=True)

    df.loc[df["Month"] == "aug", "Month"] = "Aug"
    # Заменим типы у булевых переменных
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)

    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def fill_missing_with_median(df, group_col, target_col):
        # Группируем по количеству просмотренных страниц и вычисляем медиану
        median_values = X_train[(X_train[group_col] > 0) & (X_train[target_col].notnull())].groupby(group_col)[target_col].median()

        # Заполняем пропуски в целевом столбце медианными значениями по группам
        for group_value in median_values.index:
            df.loc[(df[group_col] == group_value) & (df[target_col].isna()), target_col] = median_values[group_value]

        # Для нулевых значений посещения страниц время на страницах заполним нулями
        df.loc[(df[group_col] == 0) & (df[target_col].isna()), target_col] = 0

        # Оставшиеся пропуски заполним медианными значениями по всем данным
        median_value = X_train[target_col].median()
        df.loc[df[target_col].isna(), target_col] = median_value

    # Заполнение пропусков для Informational_Duration
    fill_missing_with_median(X_train, 'Informational', 'Informational_Duration')
    fill_missing_with_median(X_test, 'Informational', 'Informational_Duration')

    # Заполнение пропусков для ProductRelated_Duration
    fill_missing_with_median(X_train, 'ProductRelated', 'ProductRelated_Duration')
    fill_missing_with_median(X_test, 'ProductRelated', 'ProductRelated_Duration')

    # пропуски в ExitRates заполним медианой
    ExitRates_median = X_train[X_train["ExitRates"].notnull()]["ExitRates"].median()

    X_train.loc[(X_train["ExitRates"].isna()), "ExitRates"] = ExitRates_median
    X_test.loc[(X_test["ExitRates"].isna()), "ExitRates"] = ExitRates_median

    categorical = [
        'SpecialDay', 'Month', 'OperatingSystems', 'Browser',
        'Region', 'TrafficType', 'VisitorType', 'Weekend'
    ]
    numeric_features = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues'
    ]

    ct = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical),
        ('scaling', MinMaxScaler(), numeric_features)
    ])

    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed = ct.transform(X_test)
    # X_transformed = ct.transform(X)

    new_features = list(ct.named_transformers_['ohe'].get_feature_names_out())
    new_features = [el.replace(".", "_") for el in new_features]
    new_features.extend(numeric_features)

    # new_features
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=new_features)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=new_features)
    # X_transformed = pd.DataFrame(X_transformed, columns=new_features)

    X_train_transformed.to_csv('data/X_train.csv', index=False)
    X_test_transformed.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


if __name__ == '__main__':
    main()
