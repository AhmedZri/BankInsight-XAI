from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X, numeric_strategy='median', categorical_strategy='most_frequent', scale=True):
    nums = list(X.select_dtypes(include='number').columns)
    cats = [c for c in X if c not in nums]
    numeric = [('imputer', SimpleImputer(strategy=numeric_strategy, keep_empty_features=True))]
    if scale:
        numeric.append(('scaler', StandardScaler()))
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value='manquant', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([
        ('numeric', Pipeline(numeric), nums), ('categorical', categorical, cats)],
        verbose_feature_names_out=True)


def original_feature_mapping(preprocessor):
    """Correspondance positionnelle exacte : aucune découpe ambiguë de noms."""
    nums = list(preprocessor.transformers_[0][2])
    cats = list(preprocessor.transformers_[1][2])
    if not cats:
        return nums
    encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
    return nums + [name for name, values in zip(cats, encoder.categories_) for _ in values]
