"""Smoke test Streamlit dan dua kelas prediksi dari profil simulasi."""
import json
from pathlib import Path
import joblib
import numpy as np
from streamlit.testing.v1 import AppTest
from model_pipeline import make_input

ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'artifacts'/'metadata.json').read_text())
model=joblib.load(ROOT/'artifacts'/'model.joblib')
assert meta['feature_count']==22
assert (meta['metrics']['tn'],meta['metrics']['fp'],meta['metrics']['fn'],meta['metrics']['tp'])==(2432,445,257,698)

at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
assert not at.exception, at.exception
assert len(at.selectbox)==13 and len(at.number_input)==8 and len(at.date_input)==1
for preset,expected in [('Karier stabil',0),('Eksplorasi karier',1)]:
    at.button(key='example_'+preset).click().run()
    assert not at.exception,at.exception
    submit=next(b for b in at.button if b.label=='Jalankan prediksi')
    submit.click().run()
    assert not at.exception,at.exception
    result=at.session_state['prediction']
    ref=make_input(meta['examples'][preset],meta['features'])
    assert result['label']==expected==int(model.predict(ref)[0])
    assert np.isclose(result['score'],model.predict_proba(ref)[0,1])
    print(preset, result)

# Missing, termasuk angka kosong, harus diproses tanpa merusak klasifikasi.
at.number_input(key='field_skor_tes_awal').set_value(None)
at.selectbox(key='field_jenis_kelamin').select(None)
at.date_input(key='field_tanggal_daftar').set_value(None)
next(b for b in at.button if b.label=='Jalankan prediksi').click().run()
assert not at.exception,at.exception
assert np.isfinite(at.session_state['prediction']['score'])
print('PASS: form lengkap, kedua kelas, nilai kosong dan model portabel.')
