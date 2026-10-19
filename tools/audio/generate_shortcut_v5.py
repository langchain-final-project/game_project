"""Original shortcut latch / remote barred-gate release, version 5.

No existing sound is stretched or reused; no samples or external recordings.
NumPy/SciPy synthesize dry latch, short granular bar travel, light open stop,
and low early-room reflections. No musical cue, hinge creak, clang or bass hit.
Run: python generate_shortcut_v5.py [--output rendered]
"""
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
SEED=2026101905
RNG=np.random.default_rng(SEED)
DURATION=.52


def tm(seconds):return np.arange(round(seconds*SR))/SR


def filter_(x,low=None,high=None):
    if low and high:freq=[low,high];kind='bandpass'
    elif low:freq=low;kind='highpass'
    else:freq=high;kind='lowpass'
    return signal.sosfilt(signal.butter(4,freq,fs=SR,btype=kind,output='sos'),x,axis=0)


def fade(x,attack=.0015,release=.012):
    x=x.copy();a=min(len(x),round(attack*SR));r=min(len(x),round(release*SR))
    shape=(-1,1) if x.ndim==2 else (-1,)
    if a:x[:a]*=(np.sin(np.linspace(0,np.pi/2,a))**2).reshape(shape)
    if r:x[-r:]*=(np.sin(np.linspace(np.pi/2,0,r))**2).reshape(shape)
    return x


def noise(seconds,low,high):
    x=filter_(RNG.normal(size=len(tm(seconds))),low,high)
    return x/max(float(np.sqrt(np.mean(x*x))),1e-12)


def env(t,attack,decay):return (1-np.exp(-t/attack))**2*np.exp(-t/decay)


def put(dst,x,start=0,gain=1,pan=0):
    i=round(start*SR)
    if x.ndim==1:x=np.column_stack((x*np.sqrt(1-pan),x*np.sqrt(1+pan)))
    n=min(len(x),len(dst)-i)
    if n>0:dst[i:i+n]+=x[:n]*gain


def latch(seconds=.060):
    t=tm(seconds)
    # A spring-loaded pawl releasing: very brief irregular friction and muted
    # inharmonic modes, not a tuned bell or chord.
    x=noise(seconds,780,3300)*env(t,.0009,.0058)*.82
    for freq,amp,decay in [(873,.37,.009),(1459,.21,.006),(2371,.055,.0038)]:
        x+=amp*np.sin(2*np.pi*freq*t)*env(t,.0015,decay)
    return fade(x,.001,.012)


def scrape(seconds=.232):
    t=tm(seconds);p=t/seconds
    grain=noise(seconds,620,2450)
    contour=np.sin(np.pi*p)**.75
    irregular=.58+.19*np.sin(2*np.pi*(19*t+9*t*t))+.09*np.sin(2*np.pi*47.3*t+.4)
    x=grain*contour*irregular*.22
    # Nonuniform little contacts convey short bar movement without a continuous
    # long sliding door or a regular ticking rhythm.
    for offset,duration,gain,center in [(.013,.025,.17,1120),(.061,.019,.11,1510),
                                        (.109,.029,.14,930),(.166,.023,.09,1370)]:
        tt=tm(duration)
        contact=noise(duration,center*.62,min(3200,center*1.65))
        contact*=np.sin(np.pi*tt/duration)**2
        i=round(offset*SR);n=min(len(contact),len(x)-i)
        x[i:i+n]+=contact[:n]*gain
    return fade(x,.008,.026)


def open_stop(seconds=.072):
    t=tm(seconds)
    # A small mechanism reaching the unlocked stop: higher and lighter than
    # the initial latch, with no low impact or long metal-ring tail.
    x=noise(seconds,950,2850)*env(t,.0015,.0065)*.46
    x+=.25*np.sin(2*np.pi*1217*t)*env(t,.002,.010)
    x+=.09*np.sin(2*np.pi*2039*t)*env(t,.002,.005)
    return fade(x,.0015,.017)


def metrics(pcm):
    y=pcm.astype(float)/32768;mono=y.mean(axis=1)
    f,p=signal.welch(mono,SR,nperseg=min(2048,len(mono)));total=max(float(p.sum()),1e-20)
    ma=signal.convolve(np.mean(y*y,axis=1),np.ones(110)/110,mode='same')
    maximum=float(ma.max());peaks,_=signal.find_peaks(ma,prominence=maximum*.04,distance=round(.025*SR))
    active=np.flatnonzero(ma>=maximum*.0001)
    def window(a,b):
        z=y[round(a*SR):round(b*SR)]
        return {'peak_dbfs':round(float(20*np.log10(max(float(np.max(abs(z))),1e-12))),3),
                'rms_dbfs':round(float(10*np.log10(max(float(np.mean(z*z)),1e-12))),3)}
    return {'frame_count':len(pcm),'duration_seconds':len(pcm)/SR,
            'peak_dbfs':round(float(20*np.log10(max(float(np.max(abs(y))),1e-12))),3),
            'rms_dbfs':round(float(10*np.log10(max(float(np.mean(y*y)),1e-12))),3),
            'centroid_hz':round(float(np.sum(f*p)/total),2),
            'rolloff95_hz':round(float(f[min(np.searchsorted(np.cumsum(p)/total,.95),len(f)-1)]),2),
            'energy_below_150hz_percent':round(float(p[f<150].sum()/total*100),6),
            'energy_above_5000hz_percent':round(float(p[f>=5000].sum()/total*100),6),
            'strong_envelope_peak_times_seconds':[round(float(i/SR),5) for i in peaks],
            'active_minus40db_envelope_span_seconds':[round(float(active[0]/SR),5),round(float(active[-1]/SR),5)],
            'sections':{'latch':window(0,.07),'bar_travel':window(.07,.30),'open_stop':window(.30,.40),'tail':window(.40,.52)},
            'clipped_samples':int(np.sum(abs(pcm.astype(np.int32))>=32767)),
            'first_last_max_integer_amplitude':int(np.max(abs(pcm[[0,-1]].astype(np.int32))))}


def build(outdir):
    global RNG
    RNG=np.random.default_rng(SEED);outdir.mkdir(parents=True,exist_ok=True)
    dry=np.zeros((round(DURATION*SR),2))
    put(dry,latch(),0,1,-.035)
    # Keep the latch unchanged; body and stop are independently raised so the
    # mechanism remains readable under music rather than only a tiny click.
    put(dry,scrape(),.064,.50*10**(7.5/20),.015)
    put(dry,open_stop(),.318,.62*10**(4.7/20),.035)
    # Early small-room reflection rather than a huge or supernatural reverb.
    out=dry.copy()
    for delay,gain,side in [(.017,.11,0),(.031,.075,1),(.052,.046,0),(.081,.027,1),(.112,.014,0)]:
        n=round(delay*SR)
        reflection=filter_(dry,high=2300)
        out[n:,side]+=reflection[:-n,side]*gain
        out[n:,1-side]+=reflection[:-n,side]*gain*.48
    out=filter_(out,low=320,high=3700)
    out-=out.mean(axis=0,keepdims=True)
    out=fade(out,.0015,.028)
    # Match the previous door cue peak: the change is physical identity/timing,
    # not making the new candidate artificially louder for comparison.
    out*=10**(-12.0/20)/max(float(np.max(abs(out))),1e-12)
    pcm=np.round(np.clip(out,-.98,.98)*32767).astype(np.int16)
    target=outdir/'sfx_door_open.wav';wavfile.write(target,SR,pcm)
    meta={'version':'shortcut_v5','file':'sfx_door_open.wav','seed':SEED,'sample_rate':SR,'channels':2,'encoding':'PCM_16',
          'source':'100% original filtered-noise and muted inharmonic modal synthesis. No external samples, reused audio, or time stretching.',
          'rationale':'Remote barred-gate shortcut release: immediate dry latch, short restrained granular mechanism movement, smaller seated-open stop, sparse short room tail. No hinged-door creak, door closing, slam, bass boom, large metallic clang, reward notes, sparkle or regular ticking.',
          'timeline':{'latch':[0,.060],'bar_movement':[.064,.296],'seated_open_stop':[.318,.390],'small_room_tail':[.335,.502],'complete':[0,.52]},
          'usage':'Play once for successful remote shortcut opening. Keep the sound tied to the existing E-trigger open event; do not change mechanics. Peak matched to prior door cue at -12 dBFS; runtime gain 1.0 before the existing SFX slider.',
          'metrics':metrics(pcm),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'human_listening_performed':False}
    (outdir/'shortcut_v5_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return meta


def validate(outdir,meta):
    sr,pcm=wavfile.read(outdir/'sfx_door_open.wav');m=meta['metrics']
    checks={'sample_rate_ok':sr==SR,'channels_ok':pcm.ndim==2 and pcm.shape[1]==2,'pcm16_ok':bool(pcm.dtype==np.int16),
            'exact_11466_frames_ok':len(pcm)==11466,'no_clipping':m['clipped_samples']==0,
            'headroom_below_minus6db_ok':m['peak_dbfs']<-6,'zero_first_last_samples':m['first_last_max_integer_amplitude']==0,
            'stop_lighter_than_latch':m['sections']['open_stop']['peak_dbfs']<m['sections']['latch']['peak_dbfs']-3,
            'negligible_bass_energy':m['energy_below_150hz_percent']<.1}
    with tempfile.TemporaryDirectory(prefix='shortcut_repro_',dir=outdir.parent) as tmp:repro=build(Path(tmp))
    result={'all_pass':all(checks.values()),'checks':checks,'sha256_reproduction_pass':meta['sha256']==repro['sha256'],
            'human_listening_performed':False,'limitation':'Physical identity, event semantics and spectral/envelope checks are design/measurement evidence, not a human listening claim.'}
    (outdir/'shortcut_v5_qc.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'rendered')
    a=p.parse_args();m=build(a.output);q=validate(a.output,m)
    print(json.dumps({'output':str(a.output),'duration':m['metrics']['duration_seconds'],'qc_pass':q['all_pass'],'reproducible':q['sha256_reproduction_pass']}))
