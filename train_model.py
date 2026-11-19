"""Reproduksi model final yang telah dipilih di notebook, tidak mengulang tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from model_pipeline import TARGET, ID, DROP_FEATURES, EXPECTED_NUMERIC, THRESHOLD, normalize_text, clean_features, make_pipeline, make_input

ROOT = Path(__file__).resolve().parent


def train(csv_path):
    df = pd.read_csv(csv_path)
    assert set(df[TARGET]) == {0, 1} and df[ID].is_unique and df[TARGET].notna().all()
    clean = df.drop_duplicates(subset=[c for c in df if c != ID], keep='first')
    X, y = clean.drop(columns=TARGET), clean[TARGET].astype(int)
    groups = pd.util.hash_pandas_object(clean_features(X), index=False)
    if groups.is_unique:
        tr, te = train_test_split(X.index, test_size=.20, random_state=42, stratify=y)
    else:
        ta, va = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
        tr, te = X.index[ta], X.index[va]
    assert not set(groups.loc[tr]) & set(groups.loc[te])
    X_train, X_test = X.loc[tr], X.loc[te]
    y_train, y_test = y.loc[tr], y.loc[te]
    model = make_pipeline().fit(X_train, y_train)
    pred, score = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    features = [c for c in X.columns if c not in DROP_FEATURES]
    assert np.array_equal(pred, model.predict(X_test[features]))
    tn, fp, fn, tp = map(int, confusion_matrix(y_test, pred, labels=[0, 1]).ravel())
    metrics = dict(accuracy=accuracy_score(y_test,pred), precision=precision_score(y_test,pred),
                   recall=recall_score(y_test,pred), f1=f1_score(y_test,pred),
                   roc_auc=roc_auc_score(y_test,score), average_precision=average_precision_score(y_test,score),
                   tn=tn,fp=fp,fn=fn,tp=tp)
    options, defaults = {}, {}
    cleaned = clean_features(X_train)
    for name in features:
        if name in EXPECTED_NUMERIC:
            defaults[name] = round(float(cleaned[name].median()), 3)
        elif name == 'tanggal_daftar':
            dates = pd.to_datetime(X_train[name], format='%m/%d/%Y', errors='coerce').dropna()
            defaults[name] = dates.sort_values().iloc[len(dates)//2].strftime('%m/%d/%Y')
        else:
            vals = normalize_text(X_train[name])
            if name == 'ukuran_perusahaan': vals = vals.replace({'oct-49': '10-49'})
            if name == 'lama_pengalaman_thn': vals = vals.str.replace(r'\s*tahun$', '', regex=True)
            choices = list(vals.value_counts().index)
            options[name] = choices
            defaults[name] = choices[0]
    # Profil contoh bersifat sintetis, bukan data pribadi atau hasil ekstraksi baris mentah.
    examples = {
        'Umum': defaults,
        'Karier stabil': {**defaults, 'jenis_kelamin':'laki-laki','status_kuliah':'tidak kuliah',
                          'indeks_pembangunan_kota':.92, 'lama_pengalaman_thn':'15', 'skor_kepuasan_kerja':88.,
                          'persentase_kenaikan_gaji':17.,'ukuran_perusahaan':'1000-4999','tipe_perusahaan':'swasta (pt)'},
        'Eksplorasi karier': {**defaults, 'indeks_pembangunan_kota':.58,'lama_pengalaman_thn':'2',
                             'status_kuliah':'kuliah penuh waktu','pengalaman_relevan':'tidak ada',
                             'skor_kepuasan_kerja':36., 'persentase_kenaikan_gaji':1.0,
                             'ukuran_perusahaan':None,'tipe_perusahaan':None}
    }
    for name, profile in examples.items():
        probe = make_input(profile, features)
        assert np.isfinite(model.predict_proba(probe)).all()
        print(name, 'score', model.predict_proba(probe)[0,1], 'kelas', model.predict(probe)[0])
    out = ROOT/'artifacts'; out.mkdir(exist_ok=True)
    meta = {'model':'Random Forest', 'threshold':THRESHOLD, 'source_notebook':'UAS_Bengkel_Koding_Optimasi_Terkontrol (1)(1).ipynb',
            'train_rows':len(tr),'test_rows':len(te),'raw_rows':len(df),'unique_rows':len(clean),
            'feature_count':len(features),'features':features,'numeric':EXPECTED_NUMERIC,'options':options,
            'defaults':defaults,'examples':examples,'metrics':metrics,'train_counts':{str(k):int(v) for k,v in y_train.value_counts().items()},
            'resampled_counts':model.named_steps['model'].resampled_counts_, 'sklearn_version':sklearn.__version__,
            'dataset_sha256':hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()}
    joblib.dump(model, out/'model.joblib', compress=3)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2))
    # Verifikasi file yang akan dipakai aplikasi sesudah serialisasi.
    restored = joblib.load(out/'model.joblib')
    assert np.array_equal(pred, restored.predict(X_test[features]))
    return meta


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    train(parser.parse_args().csv)
