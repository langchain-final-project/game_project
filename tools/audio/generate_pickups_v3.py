"""외부 음원 없이 NumPy·SciPy로 수집·회복·보호·속도·자석 효과음을 합성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR=22050
SEED=2026101903
RNG=np.random.default_rng(SEED)
C4,E4,G4=261.625565,329.627557,391.995436
C5,E5,A5,C6,E6=523.251131,659.255114,880.,1046.502261,1318.510228
def tm(seconds): return np.arange(round(seconds*SR))/SR
def canvas(seconds): return np.zeros((len(tm(seconds)),2))


def filt(x,low=None,high=None):
    if low and high: spec=[low,high];kind='bandpass'
    elif low: spec=low;kind='highpass'
    else: spec=high;kind='lowpass'
    return signal.sosfilt(signal.butter(4,spec,fs=SR,btype=kind,output='sos'),x,axis=0)


def fade(x,attack=.003,release=.022):
    x=x.copy();a=min(len(x),round(attack*SR));r=min(len(x),round(release*SR))
    shape=(-1,1) if x.ndim==2 else (-1,)
    if a: x[:a]*=(np.sin(np.linspace(0,np.pi/2,a))**2).reshape(shape)
    if r: x[-r:]*=(np.sin(np.linspace(np.pi/2,0,r))**2).reshape(shape)
    return x


def stereo(x,pan=0):
    if x.ndim==2:return x
    return np.column_stack((x*np.sqrt(1-pan),x*np.sqrt(1+pan)))


def put(dst,src,sec=0,gain=1,pan=0):
    i=round(sec*SR);src=stereo(src,pan)*gain;n=min(len(src),len(dst)-i)
    if n>0:dst[i:i+n]+=src[:n]
def norm(x): return x/max(float(np.sqrt(np.mean(x*x))),1e-12)


def noise(seconds,low,high):
    return norm(filt(RNG.normal(size=len(tm(seconds))),low,high))


def env(t,attack=.005,decay=.065):
    return (1-np.exp(-t/attack))**2*np.exp(-t/decay)


def pluck(freq,seconds,decay=.055):
    t=tm(seconds);x=np.zeros(len(t))
    for ratio,amp,d in [(1,1,1),(2.001,.38,.7),(3.006,.15,.44),(4.01,.05,.28)]:
        x+=amp*np.sin(2*np.pi*freq*ratio*t)*env(t,.0045,decay*d)

    x+=.035*noise(seconds,630,1900)*env(t,.002,.010)
    return fade(filt(x,high=2400),.0015,.025)


def tine(freq,seconds=.32):
    t=tm(seconds)
    x=np.sin(2*np.pi*freq*t)*env(t,.0038,.095)
    x+=.13*np.sin(2*np.pi*freq*2.006*t)*env(t,.0038,.039)
    x+=.019*np.sin(2*np.pi*freq*2.71*t)*env(t,.004,.024)

    x+=.035*np.sin(2*np.pi*freq*.9991*t)*env(t,.007,.12)
    return fade(filt(x,high=3450),.0015,.05)


def pulse(freqs,seconds=.13):
    t=tm(seconds);x=np.zeros(len(t))
    for i,f in enumerate(freqs):
        x+=(np.sin(2*np.pi*f*t)+.24*np.sin(2*np.pi*f*2.002*t))*env(t,.004,.044)/(1+i*.25)
    return fade(filt(x,high=2850),.0015,.024)


def seat_tick(seconds=.050):
    t=tm(seconds)
    x=.65*np.sin(2*np.pi*812*t)*env(t,.002,.009)
    x+=.22*np.sin(2*np.pi*1670*t)*env(t,.002,.006)
    x+=noise(seconds,650,2600)*env(t,.0018,.007)*.18
    return fade(x,.001,.012)


def material(name):
    if name in ['fabric','paper','leaf','wood']:
        duration={'fabric':.17,'paper':.15,'leaf':.18,'wood':.11}[name]
        t=tm(duration);p=t/duration
        lo,hi={'fabric':(430,1650),'paper':(1100,3600),'leaf':(1650,3400),'wood':(600,2050)}[name]
        x=noise(duration,lo,hi)*np.sin(np.pi*p)**1.5
        if name=='paper': x*=.5+.5*np.sin(3*np.pi*p)**2
        if name=='leaf': x*=.55+.45*np.sin(5*np.pi*p)**2
        if name=='wood':
            x=.24*x+1.1*np.sin(2*np.pi*698*t)*env(t,.004,.022)
        return fade(x,.008,.025)
    duration=.19;t=tm(duration)
    if name=='metal':
        x=(np.sin(2*np.pi*1174.66*t)+.34*np.sin(2*np.pi*2157*t))*env(t,.003,.051)
    elif name=='ceramic':
        x=(np.sin(2*np.pi*913*t)+.24*np.sin(2*np.pi*1529*t))*env(t,.0027,.021)
    else:
        x=pluck(E4,duration,.039)

        second=pluck(G4,.115,.033);i=round(.055*SR)
        x[i:i+len(second)]+=.57*second[:len(x)-i]
    return fade(x,.0015,.025)


def formant_swoop(seconds,descending=True,variant=0):
    t=tm(seconds);p=t/seconds
    centers=np.array([640,1040,1690,2650.])*(1+variant*.035)
    white=RNG.normal(size=len(t))
    bands=np.array([norm(filt(white,float(c*.69),float(c*1.32))) for c in centers])
    q=p if descending else 1-p
    target=2550*(650/2550)**(q**.68)*(1+variant*.015)
    weights=np.exp(-.5*((np.log(centers[:,None])-np.log(target[None,:]))/.42)**2)
    weights/=np.sqrt(np.sum(weights*weights,axis=0,keepdims=True))
    x=np.sum(bands*weights,axis=0)
    shape=np.sin(np.pi*p)**1.15
    x*=shape
    # 좌우 효과가 사라지는 모노 재생에서도 주파수 이동으로 자석을 구분한다.
    pan=.5*np.sin(np.pi*p+variant*.4)*(1-p)
    out=np.column_stack((x*np.sqrt(1-pan),x*np.sqrt(1+pan)))
    return fade(out,.004,.018)


def mix_stems(tonal,noisy,noise_share):
    te=float(np.sum(tonal*tonal));ne=float(np.sum(noisy*noisy))
    noisy=noisy*np.sqrt(max(te,1e-15)/max(ne,1e-15)*noise_share/(1-noise_share))
    return tonal+noisy,noisy


def arrival(seconds=.080,freq=440):
    t=tm(seconds)
    x=(np.sin(2*np.pi*freq*t)+.24*np.sin(2*np.pi*freq*1.5*t))*env(t,.003,.018)
    return fade(filt(x,high=1900),.0015,.016)


def linear_master(x):
    x=filt(x,low=150,high=3900)
    x-=x.mean(axis=0,keepdims=True)
    return fade(x,.0015,.018)


def metrics(pcm):
    y=pcm.astype(float)/32768;m=y.mean(axis=1)
    f,p=signal.welch(m,SR,nperseg=min(2048,len(m)),noverlap=min(1024,len(m)//2))
    total=max(float(p.sum()),1e-20);cum=np.cumsum(p)/total
    ms=signal.convolve(np.mean(y*y,axis=1),np.ones(220)/220,mode='same')
    peaks,_=signal.find_peaks(ms,prominence=float(ms.max())*.15,distance=round(.055*SR))
    active=np.flatnonzero(ms>=float(ms.max())*.1)
    return {'duration_seconds':round(len(m)/SR,5),'peak_dbfs':round(20*np.log10(max(np.max(abs(y)),1e-12)),3),
            'rms_dbfs':round(10*np.log10(max(float(np.mean(y*y)),1e-12)),3),
            'spectral_centroid_hz':round(float(np.sum(f*p)/total),2),
            'rolloff_95_hz':round(float(f[min(np.searchsorted(cum,.95),len(f)-1)]),2),
            'energy_above_2000hz_percent':round(float(p[f>=2000].sum()/total*100),4),
            'energy_above_5000hz_percent':round(float(p[f>=5000].sum()/total*100),6),
            'energy_below_150hz_percent':round(float(p[f<150].sum()/total*100),5),
            'peak_envelope_times_seconds':[round(float(i/SR),4) for i in peaks],
            'active_10pct_span_seconds':round(float((active[-1]-active[0])/SR),4) if len(active) else 0,
            'clipped_samples':int(np.sum(np.abs(pcm.astype(np.int32))>=32767)),
            'first_last_max_integer_amplitude':int(np.max(abs(pcm[[0,-1]].astype(np.int32))))}


def build(outdir):
    global RNG
    RNG=np.random.default_rng(SEED);outdir.mkdir(parents=True,exist_ok=True);records=[]
    def write(name,x,peak,rationale,category,noise_stem=None):
        y=linear_master(x);gain=10**(peak/20)/max(np.max(abs(y)),1e-12)
        pcm=np.round(np.clip(y*gain,-.98,.98)*32767).astype(np.int16)
        path=outdir/name;wavfile.write(path,SR,pcm)
        item={'file':name,'category':category,'sample_rate':SR,'channels':2,'encoding':'PCM_16',
              'timbre_rationale':rationale,'metrics':metrics(pcm),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'play_once_composite':name not in ['sfx_warmth_small.wav','sfx_warmth_large.wav','sfx_memory_common.wav']}
        if noise_stem is not None:
            ns=linear_master(noise_stem);ts=linear_master(x-noise_stem)
            item['filtered_noise_stem_energy_share_percent']=round(float(np.sum(ns*ns)/(np.sum(ns*ns)+np.sum(ts*ts))*100),3)
        records.append(item)

    small=canvas(.20)
    put(small,pluck(C4,.20,.038),0,.9,-.04)
    put(small,pluck(E4,.137,.035),.063,.80,.04)
    write('sfx_warmth_small.wav',small,-10.2,
          'Dry low C4–E4 wooden/felt pluck, two definite rounded 5 ms attacks and clean short body. No pad, glass partials or sub-bass thud.','warmth')
    large=canvas(.36)
    for f,offset,gain,pan in [(C4,0,.83,-.06),(E4,.080,.78,0),(G4,.160,.72,.06)]:
        put(large,pluck(f,.36-offset,.055),offset,gain,pan)
    write('sfx_warmth_large.wav',large,-8.3,
          'Same dry lower wooden/felt instrument, C4–E4–G4 three overlapping plucks. Fuller rising triad rather than gain-only copy.','warmth')

    mem=canvas(.99)
    for f,offset,gain,pan in [(A5,0,.83,-.10),(C6,.300,.77,0),(E6,.600,.76,.10)]:
        put(mem,tine(f,.39),offset,gain,pan)
    write('sfx_memory_common.wav',mem,-8.1,
          'Clearly higher old music-box A5–C6–E6 at 0/300/600 ms. Three separate 4 ms tine attacks, decaying resonance, gentle higher partials; no blanket warm chord or continuous pad.','memory')
    for mat in ['fabric','paper','leaf','metal','music','ceramic','wood']:
        x=mem.copy();tex=material(mat)

        material_layer=canvas(.99);put(material_layer,tex,.115,1,-.12)
        scale=np.sqrt(np.sum(mem*mem)*.075/max(np.sum(material_layer*material_layer),1e-15))
        x+=material_layer*scale
        write(f'sfx_memory_{mat}.wav',x,-8.1,
              f'Same separated high A5–C6–E6 music-box motif, with an exposed {mat} texture between first and second notes. Material has about 7.5% of unprocessed motif energy: quiet but clearly present. Play composite alone.','memory')

    badge=canvas(.59)
    put(badge,seat_tick(),0,1.4)
    put(badge,pulse([G4,C5],.16),.105,.75,-.04)
    put(badge,pulse([440,587.329535],.20),.295,.82,.04)
    write('sfx_badge_collect.wav',badge,-7.7,
          'Audible tactile wooden/metal seating tick, then TWO short confident mid-register harmonic pulses at 105/295 ms. Discrete latch-and-pulse rhythm rather than the memory melody or a continuous bloom.','badge')

    heal_tone=canvas(.61);heal_air=canvas(.61)
    put(heal_tone,seat_tick(),0,.18)
    for onset,center,dur,pan in [(.065,510,.145,-.18),(.17,700,.16,0),(.29,920,.20,.18)]:
        t=tm(dur);f=center*(.85+.30*(1-np.exp(-t/.022)))
        phase=2*np.pi*np.cumsum(f)/SR
        droplet=(np.sin(phase)+.13*np.sin(2*phase))*env(t,.004,.043)
        put(heal_tone,fade(droplet,.0015,.03),onset,.65,pan)
        put(heal_air,noise(dur,900,2800)*env(t,.003,.030),onset,.12,pan)
    t=tm(.32);open_tail=(np.sin(2*np.pi*E5*t)+.35*np.sin(2*np.pi*G4*t))*np.sin(np.pi*t/.32)**2
    put(heal_tone,fade(open_tail,.01,.04),.27,.12)
    heal,heal_noise=mix_stems(heal_tone,heal_air,.22)
    write('sfx_badge_heal.wav',heal,-7.8,
          'Independent HEAL identity: three rounded rising watery droplets, audible bubbling air and a small final warm bloom. Substantial liquid pitch curves and splashes replace the generic badge pulses. Complete pickup+activation once.','badge_heal',heal_noise)

    shield_tone=canvas(.61);shield_air=canvas(.61)
    put(shield_tone,seat_tick(),0,.22)
    t=tm(.545);p=t/.545;shape=np.sin(np.pi*p)**.85*np.exp(-p*.75)
    dome=np.zeros((len(t),2))
    for side,detune in [(0,-.0018),(1,.0018)]:
        for f,amp in [(349.228,1),(622.254,.48),(1046.50,.20),(1396.91,.055)]:
            phase=2*np.pi*np.cumsum(f*(1+detune+.055*np.exp(-t/.045)))/SR
            dome[:,side]+=amp*np.sin(phase)*shape
    put(shield_tone,fade(dome,.010,.055),.045,.38)
    shell=noise(.49,550,1750)*np.sin(np.pi*tm(.49)/.49)**1.8
    put(shield_air,shell,.065,.10)
    shield,shield_noise=mix_stems(shield_tone,shield_air,.25)
    write('sfx_badge_shield.wav',shield,-7.8,
          'Independent SHIELD identity: one round sustained resonant dome with sympathetic upper modes, subtle settling pitch and enveloping air shell. No melody and no two-pulse badge chord. Complete pickup+activation once.','badge_shield',shield_noise)

    speed_tone=canvas(.58);speed_air=canvas(.58)
    put(speed_tone,seat_tick(),0,.19)
    for onset,pan,variant in [(.045,-.60,0),(.270,.60,2)]:
        whoosh=formant_swoop(.20,False,variant)

        whoosh*=np.array([np.sqrt(1-pan),np.sqrt(1+pan)])
        put(speed_air,whoosh,onset,.40)
    put(speed_tone,arrival(.080,659.255),.465,.20)
    speed,speed_noise=mix_stems(speed_tone,speed_air,.91)
    write('sfx_badge_speed.wav',speed,-8.0,
          'Independent SPEED identity: TWO quick directional upward-filtered airy whooshes, separated by a clear gap, then a tiny high rounded settle. Air is dominant, not a decorative layer on a badge chord. Complete pickup+activation once.','badge_speed',speed_noise)

    mag_tone=canvas(.60);mag_air=canvas(.60)
    put(mag_air,formant_swoop(.43,True),.015,.55)

    t=tm(.35);f=1130*(440/1130)**(t/.35)**.75
    spiral=np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*t/.35)**1.4
    put(mag_tone,spiral,.045,.20)
    put(mag_tone,arrival(.13,440),.415,.76)
    mag,mag_noise=mix_stems(mag_tone,mag_air,.68)
    write('sfx_magnet_activate.wav',mag,-7.8,
          'MAGNET activation: dominant inward air formant 2.5 kHz→650 Hz plus audible 1130→440 Hz falling pitch, narrowing into one centred rounded arrival. No generic badge chord, vacuum motor, hard hit or loop. Complete activation once.','magnet',mag_noise)
    for idx,(duration,freq,variant) in enumerate([(.20,440,0),(.22,493.883,1),(.21,523.251,2)],1):
        tonal=canvas(duration);air=canvas(duration)
        put(air,formant_swoop(.137+idx*.003,True,variant),0,.45)
        put(tonal,arrival(.080,freq),duration-.080,.80)

        t=tm(.115);f=(910+idx*65)*(510/(910+idx*65))**(t/.115)
        put(tonal,np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*t/.115)**2,.010,.13)
        absorb,abs_noise=mix_stems(tonal,air,.66)
        write(f'sfx_magnet_absorb_{idx:02}.wav',absorb,-11.1-idx*.18,
              'Short crisp-but-soft inward formant swish with clearly audible air, closing into a brief muted arrival. Same family in three pitch/formant variants; complete absorption+arrival replacing ordinary warmth, never stacked.','magnet_absorb',abs_noise)

    meta={'version':3,'seed':SEED,'source':'100% original deterministic modal, string, swept-formant and filtered-noise synthesis; no outside recordings or samples.',
          'format':'WAV PCM16 stereo 22050 Hz','human_listening_performed':False,
          'mixing':'Distinct register/rhythm/envelope/physical gesture per category. Above-5-kHz energy restricted by 3.9-kHz master low-pass. All composite variants play alone once.',
          'metrics_note':'Welch power centroid/rolloff and envelope peaks are objective proxies, not a human perception guarantee. Stem share uses separately filtered noise and tonal stem energies before shared normalization.',
          'files':records}
    (outdir/'pickup_v3_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return meta


def spectral_vector(pcm):
    mono=pcm.astype(float).mean(axis=1)/32768
    f,p=signal.welch(mono,SR,nperseg=min(2048,len(mono)))

    edges=np.geomspace(150,5000,45)
    v=np.array([p[(f>=a)&(f<b)].sum() for a,b in zip(edges[:-1],edges[1:])])
    return v/max(float(np.linalg.norm(v)),1e-20)


def qc(outdir,metadata):
    arrays={};checks=[]
    for row in metadata['files']:
        sr,x=wavfile.read(outdir/row['file']);arrays[row['file']]=x
        checks.append({'file':row['file'],'format_ok':bool(sr==SR and x.dtype==np.int16 and x.ndim==2 and x.shape[1]==2),
                       'headroom_ok':bool(row['metrics']['peak_dbfs']<=-6),'zero_endpoints':row['metrics']['first_last_max_integer_amplitude']==0,
                       'no_clipping':row['metrics']['clipped_samples']==0})
    representatives=['sfx_warmth_small.wav','sfx_memory_common.wav','sfx_badge_collect.wav','sfx_badge_heal.wav','sfx_badge_shield.wav','sfx_badge_speed.wav','sfx_magnet_activate.wav','sfx_magnet_absorb_01.wav']
    vectors={n:spectral_vector(arrays[n]) for n in representatives}
    comparisons=[]
    for i,left in enumerate(representatives):
        for right in representatives[i+1:]:
            comparisons.append({'left':left,'right':right,'logband_power_cosine_similarity':round(float(vectors[left]@vectors[right]),5)})
    with tempfile.TemporaryDirectory(prefix='repro_v3_',dir=outdir.parent) as tmp:
        second=build(Path(tmp))
    reproducible=all(a['sha256']==b['sha256'] for a,b in zip(metadata['files'],second['files']))
    stem_checks=[{'file':r['file'],'noise_stem_energy_share_percent':r['filtered_noise_stem_energy_share_percent'],
                  'at_least_20_percent_noise':r['filtered_noise_stem_energy_share_percent']>=20}
                 for r in metadata['files'] if r['category'] in ['magnet','magnet_absorb']]
    result={'all_format_headroom_endpoints_clipping_checks_pass':all(all(v for k,v in c.items() if k!='file') for c in checks),
            'deterministic_sha256_reproduction_pass':reproducible,'file_count':len(checks),'files':checks,
            'magnet_noise_energy_checks':stem_checks,'cross_category_spectral_cosines':comparisons,
            'interpretation':'Lower spectral cosine indicates different band-power distribution. Gesture/rhythm/register also differ. These proxies are not a human listening claim.',
            'human_listening_performed':False}
    (outdir/'pickup_v3_qc.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'rendered')
    args=parser.parse_args();meta=build(args.output);audit=qc(args.output,meta)
    print(json.dumps({'output':str(args.output),'wav_count':len(meta['files']),
                      'format_qc_pass':audit['all_format_headroom_endpoints_clipping_checks_pass'],
                      'deterministic':audit['deterministic_sha256_reproduction_pass']}))
