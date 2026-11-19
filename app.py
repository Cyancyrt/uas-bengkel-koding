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
st.set_page_config(page_title='CareerSignal | Prediksi Pindah Kerja', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''<style>
:root{--ink:#12243c;--muted:#64748b;--accent:#087f8c;--border:#dce5ee;}
.stApp{background:#f1f5f9;color:var(--ink)}
[data-testid="stMainBlockContainer"]{max-width:1120px;padding-top:2rem;padding-bottom:2.5rem}
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
.step-strip{display:flex;gap:8px;margin:1rem 0 1.25rem;flex-wrap:wrap}
.step-chip{flex:1;min-width:110px;background:#e7edf4;color:#60738a;padding:.75rem .8rem;border-radius:10px;font-weight:700;font-size:.85rem;line-height:1.35}
.step-chip.active{background:#087f8c;color:#fff}.step-chip.done{background:#d9f0eb;color:#126e65}
.step-chip small{display:block;opacity:.8;font-size:.7rem;font-weight:600}
.field-hint{font-size:.82rem;color:#52677d;line-height:1.5;margin:-.4rem 0 .75rem;min-height:2.45rem}
.result-box{border-radius:20px;padding:1.6rem;border:1px solid #d0e7e4;background:linear-gradient(135deg,#f0fbf7,#fff);margin:1rem 0}
.result-box.positive{background:linear-gradient(135deg,#fff4e7,#fff);border-color:#ecd2b9}
.result-heading{font-size:clamp(1.5rem,3vw,2.15rem);font-weight:800;letter-spacing:-.04em;color:#12243c;margin:.3rem 0 .6rem;line-height:1.18}
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
st.sidebar.caption('Proyek UAS Data Science. Detail teknis model tersedia di tab Kinerja model.')
st.sidebar.markdown('---')
st.sidebar.markdown(f'''<div class="side-stat"><small>MODEL TERPILIH</small><strong>Random Forest</strong><small>100 pohon · depth 14</small></div>
<div class="side-stat"><small>DATA SETELAH DEDUPLIKASI</small><strong>{meta['unique_rows']:,}</strong><small>Profil unik</small></div>
<div class="side-stat"><small>AMBANG KEPUTUSAN</small><strong>{threshold:.4f}</strong><small>Ditentukan dari CV training</small></div>''', unsafe_allow_html=True)
st.sidebar.markdown('---')
st.sidebar.caption('Model untuk pembelajaran, bukan dasar tunggal pengambilan keputusan rekrutmen. Tidak menyimpan input pengguna.')

st.markdown('''<div class="hero"><span class="hero-eyebrow">CAREERSIGNAL · UAS 2026/2027</span><div class="hero-title">Apakah peserta cenderung ingin pindah kerja?</div><p class="hero-sub">Jawab pertanyaan tentang profil, pekerjaan, dan pelatihan dalam tiga langkah. Tidak perlu memahami machine learning untuk menggunakannya.</p></div>''', unsafe_allow_html=True)
tab_pred, tab_eval, tab_method = st.tabs(['Mulai prediksi', 'Kinerja model', 'Cara kerja'])

LABELS = {
    'kode_kota':('Kode kota tempat tinggal','Contoh K103. Ini kode anonim pada dataset, bukan nama kota; jangan menebak jika tidak tahu.'),
    'jenis_kelamin':('Jenis kelamin','Pilih sesuai data peserta; boleh tidak diketahui.'),
    'pendidikan':('Pendidikan terakhir','Contoh SMA, S1 (sarjana), atau S2 (magister).'),
    'jurusan':('Bidang studi / jurusan','STEM berarti sains, teknologi, teknik, dan matematika; humaniora mencakup ilmu budaya.'),
    'status_kuliah':('Apakah sedang kuliah?','Pilih tidak kuliah, penuh waktu, atau paruh waktu.'),
    'indeks_pembangunan_kota':('Indeks pembangunan kota','Angka 0–1 dari dataset. Contoh 0,920; bukan IPM skala 0–100. Kosongkan jika tidak tahu.'),
    'pengalaman_relevan':('Pernah bekerja di bidang terkait?','Pilih Ada bila memiliki pengalaman terkait bidang data science.'),
    'lama_pengalaman_thn':('Sudah berapa lama bekerja?','Total pengalaman kerja, dalam tahun.'),
    'ukuran_perusahaan':('Berapa jumlah karyawan perusahaan?','Pilih kisaran jumlah seluruh karyawan, bukan hanya tim Anda.'),
    'tipe_perusahaan':('Jenis perusahaan saat ini','Contoh swasta (PT), startup, pemerintah, atau LSM.'),
    'jeda_pindah_kerja_thn':('Kapan terakhir berganti pekerjaan?','Dalam tahun; pilih belum pernah jika belum pernah berganti pekerjaan.'),
    'skor_kepuasan_kerja':('Seberapa puas dengan pekerjaan?','Skor 0–100. Contoh 20 = kurang puas, 80 = sangat puas.'),
    'persentase_kenaikan_gaji':('Berapa persen kenaikan gaji?','Contoh gaji naik 10%: masukkan 10. Jika tidak naik, masukkan 0.'),
    'jarak_rumah_ke_kantor_km':('Jarak rumah ke kantor (km)','Masukkan jarak satu arah, misalnya 12 km.'),
    'estimasi_gaji_bulanan_juta':('Estimasi gaji (juta Rp/bulan)','Contoh: 8,5 berarti Rp8.500.000.'),
    'tanggal_daftar':('Tanggal mendaftar pelatihan','Contoh 14/12/2023. Data pelatihan model berasal dari tahun 2023–2024.'),
    'sumber_info':('Tahu program pelatihan dari mana?','Contoh Instagram, LinkedIn, teman, atau website kampus.'),
    'skor_tes_awal':('Nilai tes sebelum pelatihan','Masukkan nilai 0–100, jika tersedia.'),
    'jam_pelatihan':('Berapa jam pelatihan yang sudah diikuti?','Total jam, bukan jumlah hari; contoh 48 jam.'),
    'jumlah_modul_selesai':('Berapa modul yang sudah selesai?','Jumlah materi yang telah diselesaikan, bukan total materi.'),
    'kegiatan_ekstrakurikuler':('Kegiatan di luar pekerjaan / kuliah','Contoh seni, olahraga atau relawan. Pilih Tidak ada bila memang tidak mengikuti kegiatan.'),
    'sertifikasi_it':('Sertifikasi IT yang dimiliki','Pilih sertifikasi yang sesuai. Tidak ada berbeda dari tidak diketahui.')
}
RANGES = {
    'indeks_pembangunan_kota':(0.,1.,.001,'%.3f'), 'skor_tes_awal':(0.,100.,.1,'%.1f'),
    'skor_kepuasan_kerja':(0.,100.,.1,'%.1f'), 'persentase_kenaikan_gaji':(0.,100.,.1,'%.2f'),
    'jarak_rumah_ke_kantor_km':(0.,None,1.,'%.1f'), 'estimasi_gaji_bulanan_juta':(0.,None,.1,'%.2f'),
    'jam_pelatihan':(0.,None,1.,'%.0f'), 'jumlah_modul_selesai':(0.,None,1.,'%.0f')
}

def to_date(value):
    return datetime.strptime(value,'%m/%d/%Y').date() if value else None

STAGES = [
    ('Profil & pendidikan', 'Ceritakan latar belakang peserta.', ['kode_kota','indeks_pembangunan_kota','jenis_kelamin','pendidikan','jurusan','status_kuliah']),
    ('Pekerjaan & penghasilan', 'Ceritakan kondisi pekerjaan saat ini.', ['pengalaman_relevan','lama_pengalaman_thn','ukuran_perusahaan','tipe_perusahaan','jeda_pindah_kerja_thn','skor_kepuasan_kerja','persentase_kenaikan_gaji','estimasi_gaji_bulanan_juta','jarak_rumah_ke_kantor_km']),
    ('Pelatihan & aktivitas', 'Ceritakan pengalaman mengikuti pelatihan.', ['tanggal_daftar','sumber_info','skor_tes_awal','jam_pelatihan','jumlah_modul_selesai','kegiatan_ekstrakurikuler','sertifikasi_it'])
]
assert sorted(k for _,_,items in STAGES for k in items)==sorted(meta['features']), 'Fitur aplikasi berbeda dengan model.'
if 'draft' not in st.session_state:
    st.session_state.draft={k:None for k in meta['features']}
if 'step' not in st.session_state:
    st.session_state.step=0

def clear_widgets():
    for k in meta['features']:
        st.session_state.pop('field_'+k,None)

def reset_form():
    clear_widgets()
    st.session_state.draft={k:None for k in meta['features']}
    st.session_state.step=0
    st.session_state.pop('prediction',None)
    st.session_state.pop('example_name',None)
    st.session_state.pop('form_error',None)

def apply_example(name):
    clear_widgets()
    st.session_state.draft=dict(meta['examples'][name])
    st.session_state.example_name=name
    st.session_state.step=3
    predict_draft()

def predict_draft():
    values=st.session_state.draft
    frame=make_input(values,meta['features'])
    score=float(model.predict_proba(frame)[0,list(model.classes_).index(1)])
    label=int(model.predict(frame)[0]); assert label==int(score>=threshold)
    dt=to_date(values.get('tanggal_daftar'))
    st.session_state.prediction={'score':score,'label':label,'date_outside':bool(dt and not date(2023,1,1)<=dt<=date(2024,11,30)),
        'missing':sum(v is None for v in values.values()), 'input':dict(values)}

def edit_form():
    clear_widgets()
    st.session_state.step=0
    st.session_state.pop('prediction',None)

def move_step(back=False):
    current=st.session_state.step
    for key in STAGES[current][2]:
        value=st.session_state.get('field_'+key)
        if key=='tanggal_daftar' and value is not None:
            value=value.strftime('%m/%d/%Y')
        st.session_state.draft[key]=value
    st.session_state.pop('form_error',None)
    if back:
        st.session_state.step=max(0,current-1)
    elif current<2:
        st.session_state.step=current+1
    elif all(v is None for v in st.session_state.draft.values()):
        st.session_state.form_error='Belum ada informasi yang diisi. Isi setidaknya satu jawaban atau coba profil simulasi.'
    else:
        predict_draft()
        st.session_state.step=3

def display_option(value, name=None):
    if value is None: return 'Tidak tahu / belum tersedia'
    if name=='pendidikan':return {'s1':'S1 (Sarjana)','s2':'S2 (Magister)','s3':'S3 (Doktor)','sma':'SMA / sederajat','sd':'SD'}.get(value,value.upper())
    if name=='jurusan' and value=='stem': return 'STEM — Sains, teknologi, teknik, matematika'
    if name=='ukuran_perusahaan':return ('Kurang dari 10' if value=='<10' else '10.000 atau lebih' if value=='10000+' else value.replace('-', '–'))+' karyawan'
    if name in ['lama_pengalaman_thn','jeda_pindah_kerja_thn'] and value.isdigit():return value+' tahun'
    if value.upper() in {'S1','S2','S3','SD','SMA','STEM','LSM'} or value.startswith('k') and value[1:].isdigit(): return value.upper()
    if value=='<1':return 'Kurang dari 1 tahun'
    if value=='>20':return 'Lebih dari 20 tahun'
    if value=='>4':return 'Lebih dari 4 tahun'
    return value[:1].upper()+value[1:]

def field(name):
    key='field_'+name; label, help_text=LABELS[name]
    if key not in st.session_state:
        v=st.session_state.draft[name]; st.session_state[key]=to_date(v) if name=='tanggal_daftar' else v
    if name=='tanggal_daftar':
        v=st.date_input(label, value=None, min_value=date(2000,1,1),max_value=date(2100,12,31), format='DD/MM/YYYY',key=key)
        return v.strftime('%m/%d/%Y') if v else None
    if name in RANGES:
        lo,hi,step,fmt=RANGES[name]
        return st.number_input(label,min_value=lo,max_value=hi,step=step,format=fmt,value=None,key=key,placeholder='Isi angka jika diketahui')
    options=[None]+meta['options'][name]
    if name in ['lama_pengalaman_thn','jeda_pindah_kerja_thn']:
        options=[None]+sorted(options[1:],key=lambda v:0 if v in ('<1','belum pernah') else float(v[1:])+1 if v.startswith('>') else int(v))
    return st.selectbox(label,options,format_func=lambda v:display_option(v,name),key=key)

def fields_section(title, items, values):
    st.markdown(f'<div class="section-title">{title}</div>',unsafe_allow_html=True)
    for i in range(0,len(items),2):
        row=st.columns(2)
        for col,name in zip(row,items[i:i+2]):
            with col:
                values[name]=field(name)
                st.markdown(f'<div class="field-hint">{LABELS[name][1]}</div>',unsafe_allow_html=True)

def show_result(result):
    if result is None:
        st.markdown('''<div class="panel"><div class="panel-kicker">HASIL PREDIKSI</div><div class="panel-title">Siap untuk dianalisis</div><p class="panel-body">Lengkapi profil di samping, lalu tekan <b>Jalankan prediksi</b>. Hasil klasifikasi, skor model, dan ambang keputusan akan tampil di sini.</p></div>''',unsafe_allow_html=True)
        return
    score, label = result['score'], result['label']; positive=bool(label)
    badge_class='badge-positive' if positive else 'badge-negative'
    result_text='Peserta diprediksi cenderung ingin pindah kerja' if positive else 'Peserta diprediksi cenderung tidak ingin pindah kerja'
    note='Nilai lebih tinggi dari atau sama dengan batas keputusan model.' if positive else 'Nilai lebih rendah dari batas keputusan model.'
    st.markdown(f'''<div class="result-box {'positive' if positive else ''}"><div class="panel-kicker">HASIL PREDIKSI PESERTA</div><span class="badge {badge_class}">{'CENDERUNG PINDAH' if positive else 'CENDERUNG BERTAHAN'}</span><div class="result-heading">{result_text}</div><div class="result-score">{score:.1%}</div><p class="panel-body">Skor kecenderungan pindah menurut model (0–100%)</p><div class="score-track"><div class="score-fill" style="width:{score*100:.4f}%"></div><div class="threshold-mark" style="left:{threshold*100:.4f}%"></div></div><p class="panel-body">Garis pembatas: {threshold:.2%}. {note}</p></div>''',unsafe_allow_html=True)
    st.caption('Skor adalah penilaian internal model, bukan peluang pasti seseorang akan pindah kerja. Hasil ini hanya prediksi, bukan keputusan nyata.')
    if result.get('missing',0):
        st.info(f'{result["missing"]} dari 22 isian tidak diketahui. Model menangani data kosong secara otomatis. Lengkapi informasi jika memungkinkan; pola data kosong dapat memengaruhi prediksi.')
    if result.get('date_outside'):
        st.warning('Tanggal di luar rentang dataset 2023–2024. Interpretasikan hasil dengan hati-hati.')

with tab_pred:
    step=st.session_state.step
    if step==0:
        with st.container(border=True):
            st.markdown('**Belum punya data? Coba contoh langsung.**')
            st.caption('Dua contoh profil fiktif yang bisa langsung menampilkan hasil. Kamu juga bisa mengubah jawabannya setelah itu.')
            a,b=st.columns(2)
            a.button('Contoh: karier stabil',key='example_Karier stabil',on_click=apply_example,args=('Karier stabil',),use_container_width=True)
            b.button('Contoh: eksplorasi karier',key='example_Eksplorasi karier',on_click=apply_example,args=('Eksplorasi karier',),use_container_width=True)
    strip=''
    for i,title in enumerate([x[0] for x in STAGES]+['Hasil prediksi']):
        kind='active' if i==step else 'done' if i<step else ''
        strip+=f'<div class="step-chip {kind}"><small>{"LANGKAH "+str(i+1) if i<3 else "SELESAI"}</small>{title}</div>'
    st.markdown(f'<div class="step-strip">{strip}</div>',unsafe_allow_html=True)
    if step<3:
        title,desc,items=STAGES[step]
        st.subheader(f'Langkah {step+1} dari 3 — {title}')
        st.caption(desc+' Isi informasi yang diketahui saja. Pilih Tidak tahu atau biarkan angka/tanggal kosong jika tidak tersedia. Jangan menebak.')
        if st.session_state.get('example_name'):
            st.info('Sedang mengubah profil contoh: '+st.session_state.example_name)
        with st.form('participant_step_'+str(step)):
            values={}
            fields_section(f'{len(items)} pertanyaan',items,values)
            back,forward=st.columns([1,2])
            with back: st.form_submit_button('Kembali',disabled=step==0,use_container_width=True,on_click=move_step,args=(True,))
            with forward: st.form_submit_button('Lanjutkan' if step<2 else 'Lihat hasil prediksi',type='primary',use_container_width=True,on_click=move_step)
        if st.session_state.get('form_error'):
            st.warning(st.session_state.form_error)
        st.button('Mulai dari awal (hapus isian)',key='reset_draft',on_click=reset_form)
    else:
        if st.session_state.get('example_name'): st.caption('Profil simulasi: '+st.session_state.example_name+' (data fiktif)')
        result=st.session_state.get('prediction')
        show_result(result)
        if result:
            st.markdown('**Apa yang saya lakukan dengan hasil ini?**')
            st.write('Gunakan sebagai bahan pertimbangan untuk memahami kebutuhan peserta. Jangan menganggap prediksi sebagai bukti seseorang pasti akan pindah atau bertahan.')
            with st.expander('Lihat 22 jawaban yang dipakai model'):
                for title,desc,items in STAGES:
                    st.markdown('**'+title+'**')
                    rows=[{'Pertanyaan':LABELS[k][0], 'Jawaban':('Tidak diketahui' if result['input'][k] is None else display_option(str(result['input'][k]),k) if k not in RANGES and k!='tanggal_daftar' else str(result['input'][k]))} for k in items]
                    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            with st.expander('Detail teknis hasil (untuk UAS)'):
                st.write(f'Kelas: {result["label"]} · Random Forest · threshold OOF {threshold:.5f} · 22 fitur mentah.')
                st.caption('Skor tidak terkalibrasi karena model dilatih menggunakan oversampling. Data kosong diimputasi oleh pipeline yang dipelajari dari data latih.')
        a,b=st.columns(2)
        a.button('Ubah jawaban',on_click=edit_form,use_container_width=True)
        b.button('Prediksi peserta baru',type='primary',on_click=reset_form,use_container_width=True)

with tab_eval:
    st.subheader('Kinerja pada data uji')
    cards = st.columns(4)
    for col,label,value in zip(cards,['Ketepatan keseluruhan','Ketepatan prediksi pindah','Deteksi peserta ingin pindah','F1-score'],[m['accuracy'],m['precision'],m['recall'],m['f1']]):
        col.metric(label,f'{value:.2%}')
    st.caption(f'Diuji pada {meta["test_rows"]:,} data historis yang tidak dipakai pelatihan. Accuracy = ketepatan keseluruhan, precision = ketepatan prediksi ingin pindah, recall = bagian peserta yang benar-benar ingin pindah dan berhasil terdeteksi.')
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
