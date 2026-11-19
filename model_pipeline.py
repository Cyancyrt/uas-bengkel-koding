"""Pipeline portabel dari notebook Optimasi Terkontrol (tanpa objek __main__)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if

TARGET = 'ingin_pindah_kerja'
ID = 'id_peserta'
DROP_FEATURES = [ID, 'kode_departemen_asal', 'sudah_ajukan_resign', 'akses_lamaran_linkedin']
EXPECTED_NUMERIC = ['indeks_pembangunan_kota', 'jam_pelatihan', 'skor_tes_awal',
                    'jumlah_modul_selesai', 'skor_kepuasan_kerja', 'persentase_kenaikan_gaji',
                    'jarak_rumah_ke_kantor_km', 'estimasi_gaji_bulanan_juta']
PLACEHOLDER = {'?', '-', 'tidak diisi', '', 'n/a', 'na', 'null', 'nan'}
THRESHOLD = 0.5233544005173285


def normalize_text(series):
    s = series.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    return s.mask(s.isin(PLACEHOLDER))


def clean_features(frame):
    if TARGET in frame.columns:
        raise ValueError('Target tidak boleh masuk clean_features.')
    a = frame.copy().drop(columns=DROP_FEATURES, errors='ignore')
    for col in a.select_dtypes(exclude='number').columns:
        s = normalize_text(a[col])
        a[col] = s.astype(object).where(s.notna(), np.nan)
    a['ukuran_perusahaan'] = a['ukuran_perusahaan'].replace({'oct-49': '10-49'})
    exp = a['lama_pengalaman_thn'].astype('string').str.replace(r'\s*tahun$', '', regex=True).str.strip()
    a['pengalaman_kurang_1'] = exp.eq('<1').fillna(False).astype(float)
    a['pengalaman_lebih_20'] = exp.eq('>20').fillna(False).astype(float)
    a['lama_pengalaman_thn'] = pd.to_numeric(exp.replace({'<1': '0.5', '>20': '21'}), errors='coerce').astype(float)
    a['lama_pengalaman_thn'] = a['lama_pengalaman_thn'].where(a['lama_pengalaman_thn'] >= 0)
    dates = pd.to_datetime(a.pop('tanggal_daftar'), format='%m/%d/%Y', errors='coerce')
    a['tahun_daftar'] = dates.dt.year.astype(float)
    a['bulan_daftar'] = dates.dt.month.astype(float)
    for col in EXPECTED_NUMERIC:
        a[col] = pd.to_numeric(a[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for col, lo, hi in [('indeks_pembangunan_kota', 0, 1), ('skor_tes_awal', 0, 100), ('skor_kepuasan_kerja', 0, 100)]:
        a[col] = a[col].where(a[col].between(lo, hi))
    for col in ['jam_pelatihan', 'jumlah_modul_selesai', 'jarak_rumah_ke_kantor_km', 'estimasi_gaji_bulanan_juta']:
        a[col] = a[col].where(a[col] >= 0)
    return a


class UpperQuantileClipper(TransformerMixin, BaseEstimator):
    def __init__(self, quantile=.99): self.quantile = quantile
    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.upper_ = np.array([np.quantile(c[np.isfinite(c)], self.quantile)
                                if self.quantile is not None and np.isfinite(c).any() else np.inf for c in arr.T])
        return self
    def transform(self, X):
        check_is_fitted(self, 'upper_')
        return np.minimum(np.asarray(X, dtype=float), self.upper_)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [f'x{i}' for i in range(self.n_features_in_)], dtype=object)


def other_numeric(frame):
    return [c for c in frame.select_dtypes(include='number') if c != 'jam_pelatihan']

def categorical_columns(frame):
    return frame.select_dtypes(exclude='number').columns.tolist()

def all_numeric(frame):
    return frame.select_dtypes(include='number').columns.tolist()


def can_predict_proba(wrapper):
    return hasattr(getattr(wrapper, 'estimator_', wrapper.estimator), 'predict_proba')


class BalancedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, random_state=42, ratio=1.0, threshold=None):
        self.estimator = estimator
        self.random_state = random_state
        self.ratio = ratio
        self.threshold = threshold
    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        labels, counts = np.unique(y, return_counts=True)
        if self.ratio is not None and not 0 < self.ratio <= 1:
            raise ValueError('ratio harus (0,1] atau None')
        target_count = int(np.ceil(counts.max()*self.ratio)) if self.ratio is not None else 0
        indices = []
        for label in labels:
            idx = np.flatnonzero(y == label)
            extra = rng.choice(idx, size=max(0, target_count-len(idx)), replace=True)
            indices.append(np.concatenate([idx, extra]))
        indices = rng.permutation(np.concatenate(indices))
        self.estimator_ = clone(self.estimator).fit(X[indices], y[indices])
        self.classes_ = self.estimator_.classes_
        self.resampled_counts_ = {int(k): int(v) for k, v in zip(*np.unique(y[indices], return_counts=True))}
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        if self.threshold is not None and hasattr(self.estimator_, 'predict_proba'):
            return (self.predict_proba(X)[:, list(self.classes_).index(1)] >= self.threshold).astype(int)
        return self.estimator_.predict(X)
    @available_if(can_predict_proba)
    def predict_proba(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(X)


def make_pipeline():
    preprocessor = ColumnTransformer([
        ('hours', Pipeline([('clip', UpperQuantileClipper()),
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())]), ['jam_pelatihan']),
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler())]), other_numeric),
        ('missing_flags', MissingIndicator(features='missing-only', error_on_new=False), all_numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, sparse_output=False))]), categorical_columns)
    ], remainder='drop')
    classifier = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                                        max_features='sqrt', class_weight=None, random_state=42, n_jobs=2)
    return Pipeline([('clean', FunctionTransformer(clean_features, validate=False)),
                     ('preprocess', preprocessor),
                     ('model', BalancedClassifier(classifier, random_state=42, ratio=1.0, threshold=THRESHOLD))])


def make_input(values, features):
    """Form harus menggunakan tipe data yang sama dengan CSV, termasuk jika kosong."""
    frame = pd.DataFrame([{k: values.get(k, np.nan) for k in features}], columns=features)
    for name in features:
        if name in EXPECTED_NUMERIC:
            frame[name] = pd.to_numeric(frame[name], errors='coerce').astype(float)
        else:
            frame[name] = frame[name].astype(object)
    return frame
