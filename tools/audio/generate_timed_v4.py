"""보호 3초·자석 4초 효과의 시작·반복·종료 음원을 정해진 길이로 합성한다."""
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
SEED=2026101904
RNG=np.random.default_rng(SEED)
def tm(seconds): return np.arange(round(seconds*SR))/SR
def canvas(seconds): return np.zeros((len(tm(seconds)),2))
def norm(x): return x/max(float(np.sqrt(np.mean(x*x))),1e-12)


def filt(x,low=None,high=None):
    if low and high:cutoff=[low,high];kind='bandpass'
    elif low:cutoff=low;kind='highpass'
    else:cutoff=high;kind='lowpass'
    return signal.sosfilt(signal.butter(4,cutoff,fs=SR,btype=kind,output='sos'),x,axis=0)


def fade(x,attack=.003,release=.025):
    x=x.copy();a=min(len(x),round(attack*SR));r=min(len(x),round(release*SR))
    shape=(-1,1) if x.ndim==2 else (-1,)
    if a:x[:a]*=(np.sin(np.linspace(0,np.pi/2,a))**2).reshape(shape)
    if r:x[-r:]*=(np.sin(np.linspace(np.pi/2,0,r))**2).reshape(shape)
    return x


def stereo(x,pan=0):
    if x.ndim==2:return x
    return np.column_stack((x*np.sqrt(1-pan),x*np.sqrt(1+pan)))


def put(dst,src,second=0,gain=1,pan=0):
    i=round(second*SR);src=stereo(src,pan)*gain;n=min(len(src),len(dst)-i)
    if n>0:dst[i:i+n]+=src[:n]


def noise(seconds,low,high):
    return norm(filt(RNG.normal(size=len(tm(seconds))),low,high))


def periodic_noise(low,high,seconds=1):
    n=round(seconds*SR);f=np.fft.rfftfreq(n,1/SR)
    weights=(f/low)**4/(1+(f/low)**4)
    weights/=np.sqrt(1+(f/high)**10)
    z=(RNG.normal(size=len(f))+1j*RNG.normal(size=len(f)))*weights
    z[0]=0
    return norm(np.fft.irfft(z,n=n))
def env(t,attack=.004,decay=.05):return (1-np.exp(-t/attack))**2*np.exp(-t/decay)
def level_peak(x,db):return x*10**(db/20)/max(float(np.max(abs(x))),1e-12)


def level_rms(x,db,peak_limit=-19):
    y=x*10**(db/20)/max(float(np.sqrt(np.mean(x*x))),1e-12)
    return y*min(1,10**(peak_limit/20)/max(float(np.max(abs(y))),1e-12))


def swoop(seconds,variant=0):
    t=tm(seconds);p=t/seconds;centers=np.array([640,1040,1690,2650.])*(1+variant*.025)
    white=RNG.normal(size=len(t))
    bands=np.array([norm(filt(white,float(c*.69),float(c*1.32))) for c in centers])
    target=2550*(650/2550)**(p**.68)
    weights=np.exp(-.5*((np.log(centers[:,None])-np.log(target[None,:]))/.42)**2)
    weights/=np.sqrt(np.sum(weights*weights,axis=0,keepdims=True))
    x=np.sum(bands*weights,axis=0)*np.sin(np.pi*p)**1.15
    pan=.42*np.sin(np.pi*p)*(1-p)
    out=np.column_stack((x*np.sqrt(1-pan),x*np.sqrt(1+pan)))
    return fade(out,.004,.025)


def arrival(seconds,freq=440):
    t=tm(seconds)
    x=(np.sin(2*np.pi*freq*t)+.24*np.sin(2*np.pi*freq*1.5*t))*env(t,.003,.023)
    return fade(filt(x,high=1900),.0015,.02)


def loop_body(family):
    """반복 경계가 이어지도록 정확히 1초 주기의 음색을 합성한다."""
    t=tm(1);p=t
    out=canvas(1)
    if family=='shield':
        for side in [0,1]:

            for f,amp,j in [(349,.64,0),(622,.31,1),(1047,.12,2)]:
                phase=2*np.pi*f*t+.13*np.sin(2*np.pi*t+j*.9+side*.4)
                breathing=.72+.17*np.sin(2*np.pi*t+j*1.8)+.08*np.sin(4*np.pi*t+j)
                out[:,side]+=amp*np.sin(phase)*breathing
        air=periodic_noise(560,1800)
        airy=stereo(air)*(.75+.15*np.sin(2*np.pi*t+.5))[:,None]
        out=norm(out)*.66+norm(airy)*.46
        return level_rms(out,-32.4,-22)
    centers=[690,1110,1820,2600]
    bands=np.array([periodic_noise(c*.70,c*1.28) for c in centers])


    target=1250*np.exp(.31*np.sin(2*np.pi*t+.7))
    weights=np.exp(-.5*((np.log(np.array(centers)[:,None])-np.log(target[None,:]))/.43)**2)
    weights/=np.sqrt(np.sum(weights*weights,axis=0,keepdims=True))
    air=np.sum(bands*weights,axis=0)*(.82+.12*np.sin(2*np.pi*t+.2))
    drift=periodic_noise(750,1600)*.13
    out[:,0]=air+drift;out[:,1]=air-drift
    return level_rms(out,-33.0,-22)


def start_parts():
    shield=canvas(.34);t=tm(.32);p=t/.32
    shape=np.sin(np.pi*p)**.85*np.exp(-p*.75)
    dome=np.zeros((len(t),2))
    for side,detune in [(0,-.0008),(1,.0008)]:
        for f,amp in [(349.228,1),(622.254,.48),(1046.5,.20),(1396.91,.055)]:
            phase=2*np.pi*np.cumsum(f*(1+detune+.055*np.exp(-t/.045)))/SR
            dome[:,side]+=amp*np.sin(phase)*shape
    put(shield,fade(dome,.010,.060),.015,.38)
    shell=noise(.30,550,1750)*np.sin(np.pi*tm(.30)/.30)**1.8
    put(shield,shell,.025,.13)
    tick_t=tm(.035)
    tick=np.sin(2*np.pi*812*tick_t)*env(tick_t,.002,.007)
    put(shield,fade(tick,.001,.01),0,.22)
    shield=level_peak(fade(filt(shield,low=190,high=3800),.0015,.035),-8.0)

    magnet=canvas(.44)
    put(magnet,swoop(.36),.005,.58)
    t=tm(.29);f=1130*(440/1130)**((t/.29)**.75)
    spiral=np.sin(2*np.pi*np.cumsum(f)/SR)*np.sin(np.pi*t/.29)**1.4
    put(magnet,spiral,.035,.18)
    put(magnet,arrival(.10,440),.325,.52)
    magnet=level_peak(fade(filt(magnet,low=190,high=3800),.0015,.035),-8.0)
    return shield,magnet


def end_parts():
    shield=canvas(.28)
    t=tm(.22)

    f=622.25*(349.23/622.25)**(t/.22)
    phase=2*np.pi*np.cumsum(f)/SR
    glassless=(np.sin(phase)+.16*np.sin(2*phase))*env(t,.009,.052)
    put(shield,glassless,.018,.70)
    haze=noise(.24,650,1850)*np.sin(np.pi*tm(.24)/.24)**2
    put(shield,haze,.018,.055)
    shield=level_peak(fade(filt(shield,low=210,high=3200),.004,.075),-17.5)

    magnet=canvas(.28)
    put(magnet,swoop(.205,1),0,.36)
    put(magnet,arrival(.085,523.251),.175,.45)
    magnet=level_peak(fade(filt(magnet,low=210,high=3600),.004,.035),-17.3)
    return shield,magnet


def compose(start,loop,end,duration,body_start,body_stop,end_start):
    full=canvas(duration);put(full,start)
    n=round((body_stop-body_start)*SR)
    repeat=np.tile(loop,(int(np.ceil(n/len(loop))),1))[:n]
    repeat=fade(repeat,.11,.14)
    put(full,repeat,body_start)
    put(full,end,end_start)
    # 종료음도 효과 지속 시간 안에 포함하고 양 끝의 진폭을 0으로 맞춘다.
    return fade(full,.0015,.015)


def measure(x,loop=False):
    y=x.astype(float)/32768;mono=y.mean(axis=1)
    f,p=signal.welch(mono,SR,nperseg=min(2048,len(mono)));total=max(float(p.sum()),1e-20)
    diff=np.abs(np.diff(y,axis=0));seam=float(np.max(abs(y[0]-y[-1])))
    metric={'frame_count':int(len(y)),'duration_seconds':round(len(y)/SR,8),
            'peak_dbfs':round(float(20*np.log10(max(float(np.max(abs(y))),1e-12))),3),
            'rms_dbfs':round(float(10*np.log10(max(float(np.mean(y*y)),1e-12))),3),
            'spectral_centroid_hz':round(float(np.sum(f*p)/total),2),
            'energy_below_150hz_percent':round(float(p[f<150].sum()/total*100),5),
            'energy_above_5000hz_percent':round(float(p[f>=5000].sum()/total*100),6),
            'clipped_samples':int(np.sum(abs(x.astype(np.int32))>=32767)),
            'first_last_max_integer_amplitude':int(np.max(abs(x[[0,-1]].astype(np.int32))))}
    if loop:
        metric.update({'boundary_delta':round(seam,8),'boundary_vs_adjacent_99pct':round(seam/max(float(np.quantile(diff,.99)),1e-12),5),
                       'construction':'exact integer-cycle modes and periodic FFT noise; no fade-to-silence loop dip'})
    return metric


def build(outdir):
    global RNG
    RNG=np.random.default_rng(SEED);outdir.mkdir(parents=True,exist_ok=True)
    shield_start,magnet_start=start_parts();shield_loop=loop_body('shield');magnet_loop=loop_body('magnet')
    shield_end,magnet_end=end_parts()
    shield=compose(shield_start,shield_loop,shield_end,3,.23,2.85,2.72)
    magnet=compose(magnet_start,magnet_loop,magnet_end,4,.32,3.86,3.72)
    definitions=[
        ('sfx_badge_shield.wav',shield,False,'Exact 3-second shield: v3 dome onset, quiet modulated resonant air body, soft descending shell expiry ending at 3.000s.'),
        ('sfx_magnet_activate.wav',magnet,False,'Exact 4-second magnet: v3 inward swoop onset, quiet moving-air field, softer inward closure ending at 4.000s.'),
        ('sfx_shield_start.wav',shield_start,False,'Derived shield onset stem; enveloping dome and small seating touch.'),
        ('sfx_shield_loop.wav',shield_loop,True,'Quiet modulated dome/air body; genuine seamless 1-second period, start/end stems supply the stronger identity.'),
        ('sfx_shield_end.wav',shield_end,False,'Soft .28s descending shell dissolution; schedule at expiry minus .28s.'),
        ('sfx_magnet_start.wav',magnet_start,False,'Derived magnet onset stem; strong inward filter/pitch movement and rounded centered arrival.'),
        ('sfx_magnet_loop.wav',magnet_loop,True,'Quiet inward air-field body; genuine seamless 1-second period, no motor drone.'),
        ('sfx_magnet_end.wav',magnet_end,False,'Soft .28s inward closure and muted settle; schedule at expiry minus .28s.'),
    ]
    records=[]
    for name,signal_,loop,reason in definitions:
        pcm=np.round(np.clip(signal_,-.98,.98)*32767).astype(np.int16)
        path=outdir/name;wavfile.write(path,SR,pcm)
        records.append({'file':name,'sample_rate':SR,'channels':2,'encoding':'PCM_16','loop':loop,
                        'rationale':reason,'metrics':measure(pcm,loop),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    def level_window(x,a,b):
        y=x[round(a*SR):round(b*SR)]
        return {'peak_dbfs':round(float(20*np.log10(max(float(np.max(abs(y))),1e-12))),3),
                'rms_dbfs':round(float(10*np.log10(max(float(np.mean(y*y)),1e-12))),3)}
    meta={'version':'timed_v4','seed':SEED,'source':'Original modal, swept-formant, and periodic noise synthesis; no external recordings, reused audio files, or time stretching.',
          'human_listening_performed':False,
          'timeline':{
              'shield':{'duration':3.0,'frames':66150,'onset':[0,.34],'body':[.23,2.85],'body_fade_in':.11,'body_fade_out':.14,'release':[2.72,3.0],
                        'levels':{'onset':level_window(shield,0,.34),'body_only':level_window(shield,.40,2.60),'release':level_window(shield,2.72,3)}},
              'magnet':{'duration':4.0,'frames':88200,'onset':[0,.44],'body':[.32,3.86],'body_fade_in':.11,'body_fade_out':.14,'release':[3.72,4.0],
                        'levels':{'onset':level_window(magnet,0,.44),'body_only':level_window(magnet,.50,3.60),'release':level_window(magnet,3.72,4)}}},
          'integration':'Primary full composites must play once. Optional stems replace the full composite for cancellation/refresh logic; do not stack them on top of it. Release begins duration minus .28s, finishes at expiry. Fade or stop body when ability is cancelled. Magnet absorb events remain separate pickups only while active.',
          'files':records}
    (outdir/'timed_v4_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return meta


def validate(outdir,meta):
    checks=[]
    for row in meta['files']:
        sr,x=wavfile.read(outdir/row['file']);m=row['metrics']
        expected={'sfx_badge_shield.wav':66150,'sfx_magnet_activate.wav':88200,
                  'sfx_shield_loop.wav':22050,'sfx_magnet_loop.wav':22050}.get(row['file'])
        checks.append({'file':row['file'],'format_ok':bool(sr==SR and x.dtype==np.int16 and x.ndim==2 and x.shape[1]==2),
                       'exact_frames_ok':True if expected is None else len(x)==expected,'headroom_ok':m['peak_dbfs']<-6,
                       'no_clipping':m['clipped_samples']==0,
                       'endpoints_ok':True if row['loop'] else m['first_last_max_integer_amplitude']==0,
                       'loop_boundary_ok':True if not row['loop'] else m['boundary_vs_adjacent_99pct']<2})
    with tempfile.TemporaryDirectory(prefix='timed_repro_',dir=outdir.parent) as tmp:repeat=build(Path(tmp))
    result={'all_pass':all(all(v for k,v in c.items() if k!='file') for c in checks),'checks':checks,
            'deterministic_sha256_reproduction_pass':all(a['sha256']==b['sha256'] for a,b in zip(meta['files'],repeat['files'])),
            'human_listening_performed':False}
    (outdir/'timed_v4_qc.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'rendered')
    a=p.parse_args();m=build(a.output);q=validate(a.output,m)
    print(json.dumps({'output':str(a.output),'wav_count':len(m['files']),'qc_pass':q['all_pass'],'deterministic':q['deterministic_sha256_reproduction_pass']}))
