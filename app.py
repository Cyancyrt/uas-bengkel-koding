from datetime import datetime, date
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from model_pipeline import make_input

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CareerSignal | Prediksi Pindah Kerja', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
:root{--ink:#12243c;--muted:#64748b;--accent:#087f8c;--border:#dce5ee;}
.stApp{background:#f1f5f9;color:var(--ink)}
[data-testid="stMainBlockContainer"]{max-width:1450px;padding-top:2rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{background:#10253e;color:#f0f6fc;border-right:1px solid #27425b}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#e1ebf6}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small{color:#9db2c6}
[data-testid="stSidebar"] hr{border-color:#35506a}
h1,h2,h3,.hero-title{font-family:Manrope,Arial,sans-serif;letter-spacing:-.035em}
body,p,button,label{font-family:'DM Sans',Arial,sans-serif}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--border);border-radius:14px;padding:14px 16px;box-shadow:0 3px 10px #162d4510}
[data-testid="stMetricValue"]{font-weight:750;color:#10253e;letter-spacing:-.04em}
[data-testid="stMetricLabel"]{color:#64748b}
[data-testid="stForm"]{background:#fff;border:1px solid var(--border);border-radius:18px;padding:1.15rem 1.3rem;box-shadow:0 6px 22px #183b5610}
[data-testid="stForm"] label{font-weight:600;color:#344963}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{border-radius:9px!important;background:#f8fafc!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:1rem;border-bottom:1px solid #d8e2ec}
[data-testid="stTabs"] button{font-weight:700}
.hero{position:relative;overflow:hidden;border-radius:22px;background:linear-gradient(120deg,#10253e 0%,#173b58 68%,#087f8c 100%);padding:2rem 2.3rem;color:white;margin-bottom:1.1rem;box-shadow:0 12px 30px #1b425b22}
.hero:after{content:'';position:absolute;width:265px;height:265px;border:1px solid #ffffff25;border-radius:50%;right:-40px;top:-75px;box-shadow:0 0 0 38px #ffffff0b,0 0 0 76px #ffffff08;pointer-events:none}
.hero-eyebrow{display:inline-block;padding:.35rem .7rem;border:1px solid #8be0df6a;background:#84dfdc18;color:#a8eeeb;border-radius:99px;font-size:.72rem;font-weight:700;letter-spacing:.11em}
.hero-title{font-size:clamp(1.7rem,3vw,2.55rem);font-weight:800;margin:.8rem 0 .45rem;line-height:1.17;color:white;max-width:740px}
.hero-sub{font-size:1rem;color:#dae7f1;max-width:760px;line-height:1.55;margin:0}
.section-title{font-size:.9rem;font-weight:800;letter-spacing:.045em;color:#087f8c;text-transform:uppercase;margin:1rem 0 .55rem;padding-bottom:.45rem;border-bottom:1px solid #e1e8f0}
.panel{background:#fff;border:1px solid var(--border);border-radius:16px;padding:1.2rem 1.15rem;box-shadow:0 4px 15px #183b560a;margin-bottom:.8rem;overflow:hidden}
.panel-kicker{font-size:.73rem;color:#64748b;font-weight:800;letter-spacing:.105em;text-transform:uppercase;margin-bottom:.55rem}
.panel-title{font-size:1.3rem;font-weight:800;color:#12243c;margin:0 0 .55rem;line-height:1.25;letter-spacing:-.02em}
.panel-body{font-size:.9rem;color:#64748b;line-height:1.6;margin:0}
.logo-block{display:flex;align-items:center;gap:.7rem;margin:.1rem 0 .9rem}
.logo-mark{display:grid;place-items:center;background:#087f8c;border:1px solid #5dc8c2;border-radius:12px;width:43px;height:43px;color:white;font-size:18px;font-weight:800}
.logo-text{font-family:Manrope,sans-serif;font-size:1.23rem;font-weight:800;line-height:1.2;color:#fff}
.logo-caption{font-size:.74rem;color:#9db2c6;letter-spacing:.025em}
.side-stat{border:1px solid #34506a;border-radius:12px;padding:.9rem;margin-bottom:.6rem;background:#ffffff07}
.side-stat strong{color:#fff;font-size:1.23rem;font-weight:800;display:block}
.side-stat small{color:#a9bdce;font-size:.76rem}
.result-score{font-size:2.9rem;line-height:1.05;letter-spacing:-.065em;font-weight:850;margin:.8rem 0 .2rem;color:#12243c}
.score-track{height:13px;background:#e5edf3;border-radius:99px;position:relative;margin:1.25rem 0 .55rem;overflow:visible}
.score-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#087f8c,#40bfb2)}
.threshold-mark{position:absolute;top:-4px;height:21px;width:3px;border-radius:2px;background:#172c44}
.badge{display:inline-block;border-radius:99px;padding:.4rem .7rem;font-size:.76rem;font-weight:800;margin-bottom:.4rem}
.badge-positive{background:#fff1df;color:#a54b12}.badge-negative{background:#e4f6f0;color:#087756}
.footer-note{color:#64748b;font-size:.77rem;text-align:center;margin-top:1.6rem}
@media(max-width:700px){.hero{padding:1.4rem}.hero-sub{font-size:.92rem}[data-testid="stMainBlockContainer"]{padding:1rem .7rem}.result-score{font-size:2.3rem}}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ROOT/'artifacts'/'model.joblib')

@st.cache_data
def load_metadata():
    return json.loads((ROOT/'artifacts'/'metadata.json').read_text())

try:
    meta = load_metadata()
    model = load_model()
except (FileNotFoundError, ImportError, ValueError) as exc:
    st.error('Model tidak dapat dimuat. Pastikan artifacts/model.joblib, metadata.json, dan dependensi tersedia.')
    st.stop()

m = meta['metrics']; threshold = meta['threshold']
st.sidebar.markdown('''<div class="logo-block"><div class="logo-mark">CS</div><div><div class="logo-text">CareerSignal</div><div class="logo-caption">HR ANALYTICS / BENGKEL KODING</div></div></div>''', unsafe_allow_html=True)
st.sidebar.caption('Sistem prediksi keinginan pindah kerja peserta pelatihan. Proyek UAS Data Science.')
st.sidebar.markdown('---')
st.sidebar.markdown(f'''<div class="side-stat"><small>MODEL TERPILIH</small><strong>Random Forest</strong><small>100 pohon · depth 14</small></div>
<div class="side-stat"><small>DATA SETELAH DEDUPLIKASI</small><strong>{meta['unique_rows']:,}</strong><small>Profil unik</small></div>
<div class="side-stat"><small>AMBANG KEPUTUSAN</small><strong>{threshold:.4f}</strong><small>Ditentukan dari CV training</small></div>''', unsafe_allow_html=True)
st.sidebar.markdown('---')
st.sidebar.caption('Model untuk pembelajaran, bukan dasar tunggal pengambilan keputusan rekrutmen. Tidak menyimpan input pengguna.')

st.markdown('''<div class="hero"><span class="hero-eyebrow">DATA SCIENCE CAPSTONE · UAS 2026/2027</span><div class="hero-title">Prediksi Keinginan Pindah Kerja</div><p class="hero-sub">Kenali kecenderungan peserta pelatihan dengan model machine learning yang telah dievaluasi secara terkontrol, dari profil karier hingga aktivitas pelatihan.</p></div>''', unsafe_allow_html=True)

cards = st.columns(4)
for col, label, value in zip(cards, ['Accuracy', 'Precision kelas 1', 'Recall kelas 1', 'F1-score kelas 1'], [m['accuracy'],m['precision'],m['recall'],m['f1']]):
    col.metric(label, f'{value:.2%}')
st.caption(f'Metrik pada {meta["test_rows"]:,} data uji historis yang tidak digunakan dalam pelatihan. Kelas 1 = ingin pindah kerja.')
tab_pred, tab_eval, tab_method = st.tabs(['Prediksi peserta', 'Evaluasi model', 'Metodologi'])

LABELS = {
    'kode_kota':('Kode kota','Kode kota anonim dari dataset; bukan nama kota.'),
    'jenis_kelamin':('Jenis kelamin','Dapat dikosongkan jika tidak diketahui.'),
    'pendidikan':('Pendidikan terakhir','Jenjang pendidikan peserta.'),
    'jurusan':('Jurusan','Bidang studi utama.'),
    'status_kuliah':('Status kuliah','Status perkuliahan saat ini.'),
    'indeks_pembangunan_kota':('Indeks pembangunan kota','Skala 0–1; kosong jika tidak diketahui.'),
    'pengalaman_relevan':('Pengalaman relevan','Pengalaman terkait bidang data science.'),
    'lama_pengalaman_thn':('Lama pengalaman kerja','Dalam tahun; <1 dan >20 tersedia.'),
    'ukuran_perusahaan':('Ukuran perusahaan','Jumlah karyawan; jika tidak diketahui, pilih kosong.'),
    'tipe_perusahaan':('Tipe perusahaan','Jenis tempat bekerja saat ini.'),
    'jeda_pindah_kerja_thn':('Jeda pindah kerja','Tahun sejak perpindahan kerja terakhir.'),
    'skor_kepuasan_kerja':('Skor kepuasan kerja','Skala 0–100.'),
    'persentase_kenaikan_gaji':('Kenaikan gaji (%)','Persentase kenaikan gaji.'),
    'jarak_rumah_ke_kantor_km':('Jarak rumah–kantor (km)','Jarak dari tempat tinggal ke kantor.'),
    'estimasi_gaji_bulanan_juta':('Estimasi gaji (juta Rp/bulan)','Contoh: 8,5 berarti Rp8.500.000.'),
    'tanggal_daftar':('Tanggal pendaftaran','Rentang data pelatihan 2023–2024.'),
    'sumber_info':('Sumber informasi','Sumber mengetahui pelatihan.'),
    'skor_tes_awal':('Skor tes awal','Skala 0–100.'),
    'jam_pelatihan':('Jam pelatihan','Jumlah jam yang telah diselesaikan.'),
    'jumlah_modul_selesai':('Modul selesai','Jumlah modul yang telah diselesaikan.'),
    'kegiatan_ekstrakurikuler':('Kegiatan ekstrakurikuler','Aktivitas tambahan peserta.'),
    'sertifikasi_it':('Sertifikasi IT','Sertifikasi atau pelatihan profesional yang dimiliki.')
}
RANGES = {
    'indeks_pembangunan_kota':(0.,1.,.001,'%.3f'), 'skor_tes_awal':(0.,100.,.1,'%.1f'),
    'skor_kepuasan_kerja':(0.,100.,.1,'%.1f'), 'persentase_kenaikan_gaji':(0.,100.,.1,'%.2f'),
    'jarak_rumah_ke_kantor_km':(0.,None,1.,'%.1f'), 'estimasi_gaji_bulanan_juta':(0.,None,.1,'%.2f'),
    'jam_pelatihan':(0.,None,1.,'%.0f'), 'jumlah_modul_selesai':(0.,None,1.,'%.0f')
}

def to_date(value):
    return datetime.strptime(value,'%m/%d/%Y').date() if value else None

def apply_example(name):
    for k,v in meta['examples'][name].items():
        st.session_state['field_'+k] = to_date(v) if k=='tanggal_daftar' else v
    st.session_state.pop('prediction', None)

def display_option(value):
    if value is None: return 'Tidak diketahui / kosong'
    if value.upper() in {'S1','S2','S3','SD','SMA','STEM','LSM'} or value.startswith('k') and value[1:].isdigit(): return value.upper()
    if value=='<1':return 'Kurang dari 1 tahun'
    if value=='>20':return 'Lebih dari 20 tahun'
    if value=='>4':return 'Lebih dari 4 tahun'
    return value[:1].upper()+value[1:]

def field(name):
    key='field_'+name; label, help_text=LABELS[name]
    if key not in st.session_state:
        v=meta['defaults'][name]; st.session_state[key]=to_date(v) if name=='tanggal_daftar' else v
    if name=='tanggal_daftar':
        v=st.date_input(label, min_value=date(2000,1,1),max_value=date(2100,12,31), format='DD/MM/YYYY',key=key,help=help_text)
        return v.strftime('%m/%d/%Y') if v else None
    if name in RANGES:
        lo,hi,step,fmt=RANGES[name]
        return st.number_input(label,min_value=lo,max_value=hi,step=step,format=fmt,key=key,help=help_text)
    options=[None]+meta['options'][name]
    return st.selectbox(label,options,format_func=display_option,key=key,help=help_text)

def fields_section(title, items, values):
    st.markdown(f'<div class="section-title">{title}</div>',unsafe_allow_html=True)
    for i in range(0,len(items),2):
        row=st.columns(2)
        for col,name in zip(row,items[i:i+2]):
            with col: values[name]=field(name)

def show_result(result):
    if result is None:
        st.markdown('''<div class="panel"><div class="panel-kicker">HASIL PREDIKSI</div><div class="panel-title">Siap untuk dianalisis</div><p class="panel-body">Lengkapi profil di samping, lalu tekan <b>Jalankan prediksi</b>. Hasil klasifikasi, skor model, dan ambang keputusan akan tampil di sini.</p></div>''',unsafe_allow_html=True)
        return
    score, label = result['score'], result['label']; positive=bool(label)
    badge_class='badge-positive' if positive else 'badge-negative'
    result_text='Ingin pindah kerja' if positive else 'Tidak ingin pindah kerja'
    note='Skor model mencapai ambang keputusan kelas 1.' if positive else 'Skor model berada di bawah ambang keputusan kelas 1.'
    st.markdown(f'''<div class="panel"><div class="panel-kicker">HASIL PREDIKSI</div><span class="badge {badge_class}">KELAS {label}</span><div class="panel-title">{result_text}</div><div class="result-score">{score:.1%}</div><p class="panel-body">Skor model untuk kelas ingin pindah</p><div class="score-track"><div class="score-fill" style="width:{score*100:.4f}%"></div><div class="threshold-mark" style="left:{threshold*100:.4f}%"></div></div><p class="panel-body">Ambang keputusan: {threshold:.2%}</p><hr style="border:0;border-top:1px solid #e0e9f0;margin:1rem 0"><p class="panel-body">{note}</p></div>''',unsafe_allow_html=True)
    st.caption('Hasil dari profil terakhir yang dikirim. Skor bukan probabilitas risiko terkalibrasi. Oversampling mengubah distribusi data latih; hasil tidak menjamin keputusan seseorang.')
    if result.get('date_outside'):
        st.warning('Tanggal di luar rentang dataset 2023–2024. Interpretasikan hasil dengan hati-hati.')

with tab_pred:
    st.subheader('Masukkan profil peserta')
    st.caption('Gunakan data nyata atau pilih profil simulasi untuk demonstrasi. Input kosong tetap diproses pipeline.')
    cols=st.columns(3)
    for col,name in zip(cols,meta['examples']):
        col.button(name,key='example_'+name,on_click=apply_example,args=(name,),use_container_width=True,help='Isi form dengan profil simulasi, bukan data peserta asli.')
    left,right=st.columns([2.1,1.0],gap='large')
    with left:
        with st.form('participant_form'):
            values={}
            fields_section('01  ·  Profil dan pendidikan',['kode_kota','indeks_pembangunan_kota','jenis_kelamin','pendidikan','jurusan','status_kuliah'],values)
            fields_section('02  ·  Pengalaman dan pekerjaan',['pengalaman_relevan','lama_pengalaman_thn','ukuran_perusahaan','tipe_perusahaan','jeda_pindah_kerja_thn','skor_kepuasan_kerja','persentase_kenaikan_gaji','estimasi_gaji_bulanan_juta','jarak_rumah_ke_kantor_km'],values)
            fields_section('03  ·  Pelatihan dan aktivitas',['tanggal_daftar','sumber_info','skor_tes_awal','jam_pelatihan','jumlah_modul_selesai','kegiatan_ekstrakurikuler','sertifikasi_it'],values)
            submitted=st.form_submit_button('Jalankan prediksi',type='primary',use_container_width=True)
        if submitted:
            frame=make_input(values,meta['features'])
            score=float(model.predict_proba(frame)[0,list(model.classes_).index(1)])
            label=int(model.predict(frame)[0]);assert label==int(score>=threshold)
            dt=to_date(values['tanggal_daftar'])
            st.session_state.prediction={'score':score,'label':label,'date_outside': bool(dt and not date(2023,1,1)<=dt<=date(2024,11,30))}
            st.success(f'Prediksi selesai: {"Ingin pindah kerja" if label else "Tidak ingin pindah kerja"} · skor kelas 1: {score:.1%}. Lihat panel hasil untuk detail.')
    with right:
        show_result(st.session_state.get('prediction'))
        st.markdown('''<div class="panel"><div class="panel-kicker">CARA KERJA</div><div class="panel-title">Input → Pipeline → Prediksi</div><p class="panel-body">1. Isi 22 fitur mentah.<br>2. Pipeline membersihkan, mengisi missing, melakukan encoding dan scaling.<br>3. Random Forest menghitung skor kelas 1.<br>4. Skor dibandingkan dengan threshold hasil CV.</p></div>''',unsafe_allow_html=True)
        st.markdown('''<div class="panel"><div class="panel-kicker">AUDIT LEAKAGE</div><p class="panel-body">ID peserta, kode departemen, status resign dan aktivitas lamaran LinkedIn tidak dipakai model akhir. Input tersebut sengaja tidak ditampilkan.</p></div>''',unsafe_allow_html=True)

with tab_eval:
    st.subheader('Kinerja pada data uji')
    c1,c2=st.columns([1,1],gap='large')
    with c1:
        st.markdown('**Confusion matrix**')
        rows=pd.DataFrame([{'Aktual':'0 · Tidak pindah','Prediksi':'0 · Tidak pindah','Jumlah':m['tn']},{'Aktual':'0 · Tidak pindah','Prediksi':'1 · Ingin pindah','Jumlah':m['fp']},{'Aktual':'1 · Ingin pindah','Prediksi':'0 · Tidak pindah','Jumlah':m['fn']},{'Aktual':'1 · Ingin pindah','Prediksi':'1 · Ingin pindah','Jumlah':m['tp']}])
        base=alt.Chart(rows).encode(x=alt.X('Prediksi:N',sort=['0 · Tidak pindah','1 · Ingin pindah'],axis=alt.Axis(labelAngle=0)),y=alt.Y('Aktual:N',sort=['0 · Tidak pindah','1 · Ingin pindah']),tooltip=['Aktual','Prediksi','Jumlah'])
        heat=base.mark_rect(cornerRadius=6).encode(color=alt.Color('Jumlah:Q',scale=alt.Scale(range=['#e0f1f0','#087f8c']),legend=None))
        txt=base.mark_text(fontSize=23,fontWeight='bold').encode(text='Jumlah:Q',color=alt.condition('datum.Jumlah > 1100',alt.value('white'),alt.value('#10253e')))
        st.altair_chart((heat+txt).properties(height=260),use_container_width=True)
        st.caption(f'Terdeteksi {m["tp"]} dari {m["tp"]+m["fn"]} peserta yang sebenarnya ingin pindah. Sebanyak {m["fn"]} belum terdeteksi (false negative).')
    with c2:
        st.markdown('**Performa model akhir**')
        chart_data=pd.DataFrame({'Metrik':['Accuracy','Precision','Recall','F1-score','ROC-AUC','Average precision'],'Skor':[m['accuracy'],m['precision'],m['recall'],m['f1'],m['roc_auc'],m['average_precision']]})
        bar=alt.Chart(chart_data).mark_bar(cornerRadiusEnd=5,color='#087f8c').encode(y=alt.Y('Metrik:N',sort=chart_data['Metrik'].tolist()),x=alt.X('Skor:Q',scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format='%')),tooltip=[alt.Tooltip('Metrik:N'),alt.Tooltip('Skor:Q',format='.2%')])
        text=alt.Chart(chart_data).mark_text(align='right',dx=-6,color='white',fontWeight='bold').encode(y=alt.Y('Metrik:N',sort=chart_data['Metrik'].tolist()),x='Skor:Q',text=alt.Text('Skor:Q',format='.1%'))
        st.altair_chart((bar+text).properties(height=260),use_container_width=True)
        st.caption('F1 dipakai untuk menyeimbangkan precision dan recall kelas minoritas.')
    st.markdown('**Perbandingan 9 eksperimen dari notebook**')
    compare_path=ROOT/'artifacts'/'comparison.csv'
    if compare_path.exists():
        table=pd.read_csv(compare_path)[['Model','Skenario','Accuracy','Precision','Recall','F1-score']]
        st.dataframe(table.style.format({c:'{:.2%}' for c in ['Accuracy','Precision','Recall','F1-score']}),hide_index=True,use_container_width=True)
        st.caption('Direct memakai fitur mentah termasuk proxy berisiko leakage; tidak dijadikan pilihan deployment. Skenario tuning memakai data aman.')
    importance_path=ROOT/'artifacts'/'importance.csv'
    if importance_path.exists():
        st.markdown('**Fitur penting menurut evaluasi internal**')
        imp=pd.read_csv(importance_path).head(8)
        st.altair_chart(alt.Chart(imp).mark_bar(color='#376a9a',cornerRadiusEnd=4).encode(y=alt.Y('fitur:N',sort='-x',title='Fitur'),x=alt.X('mean_delta_f1:Q',title='Penurunan F1 saat fitur diacak'),tooltip=['fitur',alt.Tooltip('mean_delta_f1:Q',format='.4f')]).properties(height=250),use_container_width=True)
        st.caption('Permutation importance di validasi internal notebook. Asosiasi prediktif, bukan hubungan sebab-akibat.')

with tab_method:
    st.subheader('Metodologi dan batasan')
    st.markdown('''
    1. **EDA:** 25.000 baris, audit missing value, anomali, kategori, duplikasi dan distribusi target.
    2. **Deduplikasi:** 19.158 profil unik; train/test 80:20 dengan stratifikasi dan tanpa overlap profil.
    3. **Preprocessing:** normalisasi teks, validasi rentang, imputasi, clipping jam pelatihan, encoding dan scaling.
    4. **Balancing:** Random Oversampling 100% hanya pada training. Jumlah train per kelas menjadi 11.504.
    5. **Tuning:** 5-fold stratified CV, 44 konfigurasi tiga model, pemilihan F1 kelas 1 dan threshold OOF.
    6. **Deployment:** Pipeline yang sudah fit dimuat dari file joblib; input pengguna tidak melatih ulang model.
    ''')
    st.info('Model akhir dipilih dari CV training: Random Forest, 100 pohon, max_depth 14, min_samples_leaf 5, max_features sqrt, threshold 0,52335.')
    st.markdown('**Keterbatasan:** distribusi kelas tidak seimbang; skor model tidak terkalibrasi; pola missing pada ukuran perusahaan membawa sinyal prediktif; hasil belum diuji pada populasi baru. Model ini bukan alat keputusan otomatis terhadap individu.')
    st.markdown('**Sumber:** Dataset UAS Prediksi Pindah Kerja dan notebook `UAS_Bengkel_Koding_Optimasi_Terkontrol`.')

st.markdown('<div class="footer-note">CareerSignal · Bengkel Koding Data Science · Capstone Project 2026/2027</div>',unsafe_allow_html=True)
