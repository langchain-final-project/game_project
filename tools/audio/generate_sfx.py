"""Original acoustic-model sound design for Kkamang Manor.

No recordings, downloaded samples, external music, or generative service are used.
Deterministic NumPy/SciPy synthesis. Runtime game needs only the rendered PCM files.
Run: python generate_sfx.py --output /desired/output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 22050
RNG = np.random.default_rng(20261016)
META = []


def db(amp):
    return 20 * np.log10(max(float(amp), 1e-12))


def time(duration):
    return np.arange(round(duration * SR)) / SR


def envelope(n, attack=.008, decay=.04, release=.04):
    """Smooth tactile attack, decay and genuine zero at tail."""
    t = np.arange(n) / SR
    out = (1 - np.exp(-t / max(attack, .0002))) ** 2
    out *= np.exp(-t / max(decay, .001))
    k = min(n, round(release * SR))
    if k:
        out[-k:] *= np.sin(np.linspace(np.pi / 2, 0, k)) ** 2
    return out


def edge_fade(x, attack=.01, release=.06):
    x = x.copy()
    a = min(len(x), max(1, round(attack * SR)))
    r = min(len(x), max(1, round(release * SR)))
    shape = (-1, 1) if x.ndim == 2 else (-1,)
    x[:a] *= np.sin(np.linspace(0, np.pi / 2, a)).reshape(shape) ** 2
    x[-r:] *= np.sin(np.linspace(np.pi / 2, 0, r)).reshape(shape) ** 2
    return x


def noise(n, low=100, high=5000):
    white = RNG.normal(size=n)
    if low and high:
        sos = signal.butter(3, [low, high], btype='bandpass', fs=SR, output='sos')
    elif high:
        sos = signal.butter(3, high, btype='lowpass', fs=SR, output='sos')
    else:
        sos = signal.butter(3, low, btype='highpass', fs=SR, output='sos')
    x = signal.sosfilt(sos, white)
    return x / (np.std(x) + 1e-12)


def periodic_noise(n, low, high):
    f = np.fft.rfftfreq(n, 1 / SR)
    mag = np.ones_like(f)
    mag *= (f / max(low, .001)) ** 2 / (1 + (f / max(low, .001)) ** 2)
    mag /= (1 + (f / high) ** 6) ** .5
    z = (RNG.normal(size=len(f)) + 1j * RNG.normal(size=len(f))) * mag
    z[0] = 0
    x = np.fft.irfft(z, n=n)
    return x / (np.std(x) + 1e-12)


def stereo(x, width=.18, delay=.005):
    if x.ndim == 2:
        return x
    d = round(delay * SR)
    late = np.pad(x, (d, 0))[:len(x)]
    return np.column_stack((x * (1 - width) + late * width,
                            x * (1 - width) - late * width * .35))


def room(x, wet=.14, tail=.12):
    """Sparse, diffused early room reflections: not an electronic echo."""
    x = stereo(x)
    out = np.pad(x, ((0, round(tail * SR)), (0, 0)))
    for seconds, gain, pan in [(0.017, .51, .20), (.031, .36, -.21),
                               (.049, .27, .23), (.073, .16, -.13),
                               (.101, .09, .12)]:
        delay = round(seconds * SR)
        stop = min(len(out), delay + len(x))
        out[delay:stop, 0] += x[:stop-delay, 0] * wet * gain * (1-pan)
        out[delay:stop, 1] += x[:stop-delay, 1] * wet * gain * (1+pan)
    return edge_fade(out, .002, .035)


def place(dst, src, sec, gain=1, pan=0, wrap=False):
    src = stereo(src)
    src = src * np.array([np.sqrt((1-pan)/2), np.sqrt((1+pan)/2)]) * gain * 1.4142
    start = round(sec * SR)
    if wrap:
        idx = (np.arange(len(src)) + start) % len(dst)
        np.add.at(dst, idx, src)
    else:
        end = min(len(dst), start + len(src))
        if end > start:
            dst[start:end] += src[:end-start]


def resonator(freq, duration, kind='felt', strength=1):
    """A few dispersive modes plus a filtered felt excitation."""
    t = time(duration)
    if kind == 'glass':
        modes = [(1, 1, .25), (2.005, .19, .13), (2.997, .12, .10),
                 (4.17, .026, .06)]
    elif kind == 'wood':
        modes = [(1, 1, .13), (2.68, .17, .07), (4.91, .06, .035)]
    elif kind == 'metal':
        modes = [(1, 1, .48), (2.76, .13, .21), (5.4, .045, .12)]
    else:
        modes = [(1, 1, .42), (2.001, .22, .23), (3.006, .07, .13), (4.01, .022, .08)]
    x = np.zeros(len(t))
    for ratio, amp, decay in modes:
        if freq * ratio < 7600:
            x += amp * np.sin(2*np.pi*freq*ratio*t) * envelope(len(t), .004, decay, .055)
    x += .012 * noise(len(t), 350, 2200) * envelope(len(t), .002, .016, .03)
    return strength * x


def chimes(notes, offsets, duration, kind='glass', gains=None):
    out = np.zeros((round(duration*SR), 2))
    if gains is None:
        gains = [1] * len(notes)
    for i, (freq, offset, gain) in enumerate(zip(notes, offsets, gains)):
        note = resonator(freq, max(.05, duration-offset), kind)
        place(out, note, offset, gain, (i-(len(notes)-1)/2)*.10)
    return edge_fade(out, .004, .045)


def soft_scrape(duration, material='wood'):
    n = round(duration * SR)
    t = np.arange(n) / SR
    ranges = {'wood': (220, 2300), 'fabric': (450, 2500), 'paper': (800, 4800),
              'leaf': (1000, 5300), 'stone': (380, 2100)}
    lo, hi = ranges[material]
    x = noise(n, lo, hi)
    motion = np.sin(np.pi * np.clip(t/duration, 0, 1)) ** 1.8
    modulation = .63 + .21*np.sin(2*np.pi*13*t + .8*np.sin(2*np.pi*3.1*t))
    if material == 'wood':
        # A slipping joint, never a low impact.
        f = 165 + 18*np.sin(2*np.pi*1.8*t) + 8*np.sin(2*np.pi*12.1*t)
        phase = 2*np.pi*np.cumsum(f)/SR
        x = .6*x + .20*np.sin(phase) + .04*np.sin(2.97*phase)
    return edge_fade(x * motion * modulation, .013, .055)


def save(outdir, name, x, peak_db, desc, loop=False):
    x = stereo(x)
    x -= x.mean(axis=0, keepdims=True)
    if not loop:
        x = edge_fade(x, .002, min(.025, len(x)/SR/5))
    peak = np.max(np.abs(x))
    if peak > 0:
        x *= 10**(peak_db/20) / peak
    pcm = np.round(np.clip(x, -.9999, .9999) * 32767).astype(np.int16)
    wavfile.write(outdir/name, SR, pcm)
    actual = pcm.astype(float) / 32768
    d = actual[0]-actual[-1]
    META.append({'file': name, 'duration_seconds': round(len(x)/SR, 4),
                 'sample_rate': SR, 'channels': 2, 'encoding': 'PCM_16',
                 'peak_dbfs': round(db(np.max(abs(actual))), 2),
                 'rms_dbfs': round(db(np.sqrt(np.mean(actual**2))), 2),
                 'loop': loop, 'loop_boundary_delta_dbfs': round(db(np.max(abs(d))), 2) if loop else None,
                 'notes': desc, 'method': 'original deterministic modal/filter/noise synthesis; no external samples'})


def create(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    C5, E5, G5, A4, D5 = 523.251, 659.255, 783.991, 440, 587.33
    small = chimes([C5, E5], [0, .078], .30, gains=[.85, .72])
    large = chimes([C5, E5, G5], [0, .087, .173], .49, gains=[.83,.75,.62])
    save(outdir, 'sfx_warmth_small.wav', small, -10.5, 'Small warmth: soft C5–E5 glass/felt ascending pair; short and non-piercing.')
    save(outdir, 'sfx_warmth_large.wav', large, -8.8, 'Large warmth: C5–E5–G5, richer warm rising three notes.')
    memory = chimes([A4, C5, E5], [0, .18, .36], .96, 'glass', [.6,.65,.7])
    mist = stereo(noise(len(memory), 1400, 3300)) * np.sin(np.linspace(0,np.pi,len(memory)))[:,None]**2 * .008
    memory += mist
    # Material variants share exactly the same A-C-E memory gesture.
    save(outdir, 'sfx_memory_common.wav', memory, -7.3, 'Memory: transparent, slower A4–C5–E5 gesture shared with the lullaby; restrained air and room.')
    for material in ['fabric', 'paper', 'leaf', 'metal', 'music', 'ceramic', 'wood']:
        variant = memory.copy()
        if material in ['fabric','paper','leaf','wood']:
            tex = soft_scrape(.36 if material != 'wood' else .28, material)
            gain = .038 if material == 'paper' else .030
        elif material == 'metal':
            tex = resonator(1174.66, .45, 'metal'); gain=.023
        elif material == 'ceramic':
            tex = resonator(978.1, .35, 'glass') + .22*resonator(1512.7,.35,'glass'); gain=.029
        else:
            tex = resonator(329.628,.48,'felt') + .32*resonator(659.255,.48,'felt'); gain=.038
        place(variant, tex, .39, gain, -.12)
        save(outdir, f'sfx_memory_{material}.wav', variant, -7.3,
             f'Common A-C-E memory gesture with a single quiet {material} texture; prerendered composite, do not double-play common effect.')
    n = round(.58*SR)
    t = np.arange(n)/SR
    gust = noise(n, 330, 3900)
    sweep = np.sin(np.pi*t/.58)**2.2 * (.8+.2*np.sin(2*np.pi*3*t))
    airy = gust*sweep + .035*np.sin(2*np.pi*(350*t-35*t*t))*sweep
    ghost = stereo(airy, .35, .009)
    ghost[:,0] *= .62+.38*t/.58
    ghost[:,1] *= 1-.38*t/.58
    save(outdir, 'sfx_ghost_contact.wav', ghost, -10.0, 'Ghost contact: one cool breath passing left to right, diffuse air tail; no hit, scream, bass impact or stinger.')
    for i,duration in enumerate([.105,.125,.113],1):
        n=round(duration*SR); t=np.arange(n)/SR
        contact = noise(n, 460+60*i, 1950+90*i)
        pad = np.sin(np.pi*np.clip(t/duration,0,1))**2 * np.exp(-t/.045)
        fur = noise(n, 1500, 3600) * np.sin(np.pi*np.clip(t/duration,0,1))**3
        step = contact*pad + .075*fur*pad
        save(outdir, f'sfx_cat_step_0{i}.wav', stereo(step,.03,.0017), -20.5-i*.55,
             'Very quiet broad soft paw-pad/short fur friction on stone; 460 Hz high-pass, no shoe/hoof click, no pitched thud. Single successful move only.')
    ui = chimes([587.33], [0], .13, 'wood')
    save(outdir, 'sfx_ui_select.wav', ui, -14.5, 'Soft tiny wooden selection note with rounded attack.')
    door = soft_scrape(.47,'wood')
    door += .05*noise(len(door),1000,2400)*envelope(len(door),.015,.13,.08)
    save(outdir,'sfx_door_open.wav',room(door,.15,.09),-12,'Old small wooden door: restrained sliding grain and creaking joint; no slam.')
    ret=chimes([261.626,329.628,440,523.251],[0,.075,.14,.23],.98,'felt',[.55,.52,.42,.33])
    save(outdir,'sfx_return.wav',ret,-8.8,'Safe return: warm C-E-A-C felt-piano harmony resolving upward to the hearth.')
    fail=chimes([293.665,261.626],[0,.19],.91,'felt',[.6,.49])
    save(outdir,'sfx_fail.wav',fail,-11.7,'Gentle D4-C4 falling felt-piano sigh; sympathetic rather than threatening.')

    # Common seamless loops: periodic noise and wrapped one-shots are genuinely
    # continuous through the final->initial sample transition.
    duration=20.; n=round(duration*SR); t=np.arange(n)/SR
    fire=np.zeros((n,2))
    for ch in [0,1]:
        bed=periodic_noise(n,180,2000)
        bed*=.027*(.75+.18*np.sin(2*np.pi*t/duration*3+ch)+.07*np.sin(2*np.pi*t/duration*11))
        fire[:,ch]=bed
    for _ in range(65):
        dur=RNG.uniform(.035,.105); k=round(dur*SR)
        crackle=noise(k,900,5000)*envelope(k,.0025,RNG.uniform(.008,.025),.02)
        place(fire,crackle,RNG.uniform(0,duration),RNG.uniform(.008,.035),RNG.uniform(-.7,.7),True)
    save(outdir,'amb_fireplace.wav',fire,-17.5,'Seamless 20 s hearth: low irregular fine crackles on a little air, no large pops or heavy bass.',True)
    rain=np.zeros((n,2))
    for ch in [0,1]:
        bed=periodic_noise(n,380,3100)
        bed*=.12*(.92+.06*np.sin(2*np.pi*t/duration*2+ch*.3)+.025*np.sin(2*np.pi*t/duration*7))
        distant=periodic_noise(n,160,1200)*.025
        rain[:,ch]=bed+distant
    # Half the hiss is shared across channels, like one distant window.
    rain=.65*rain+.35*rain.mean(axis=1,keepdims=True)
    save(outdir,'amb_rain.wav',rain,-15.5,'Seamless 20 s muted rain beyond closed glass; soft 3 kHz low-pass, no close splashes or storm thunder.',True)
    air=np.zeros((n,2))
    for ch in [0,1]:
        air[:,ch]=.16*periodic_noise(n,160,1050)*(1+.16*np.sin(2*np.pi*t/duration*(ch+2)))
    save(outdir,'amb_manor_air.wav',air,-23,'Seamless 20 s unobtrusive interior air and timber space; no obvious pitched drone.',True)

    # Stage-specific rare non-looping signatures. The engine schedules these
    # independently of gameplay at long irregular intervals.
    stages={}
    def canvas(d): return np.zeros((round(d*SR),2))
    s=canvas(2.9)
    place(s,soft_scrape(.67,'wood'),.32,.26,-.3)
    for offset in [.17,.81,1.67,2.1]:
        drop=resonator(RNG.uniform(650,850),.19,'wood')*.06
        drop+=noise(len(drop),700,2800)*envelope(len(drop),.006,.033,.05)*.035
        place(s,drop,offset,.55,.3)
    stages[1]=(s,'Wet entrance: thin rainwater at the door gap and a soft moving timber joint.')
    s=canvas(2.7)
    place(s,soft_scrape(.82,'wood'),.25,.26,-.5)
    place(s,soft_scrape(.56,'wood'),1.55,.11,.4)
    stages[2]=(s,'Long corridor: slow quiet floorboard flex and a remote timber answer; no distinct human footsteps.')
    s=canvas(2.2)
    for offset,vol in [(.35,.21),(1.11,.09)]:
        frame=resonator(348,.28,'wood')+.10*noise(round(.28*SR),700,2600)*envelope(round(.28*SR),.008,.032,.05)
        place(s,frame,offset,vol,-.38)
    stages[3]=(s,'Frame gallery: small wooden frame gently touching plaster twice, far away.')
    s=canvas(2.65)
    place(s,soft_scrape(.65,'paper'),.28,.22,.22)
    place(s,soft_scrape(.55,'paper'),1.24,.13,-.15)
    stages[4]=(s,'Small library: book page and quiet paper edge sliding on a shelf.')
    s=canvas(3.05)
    place(s,soft_scrape(.85,'fabric'),.20,.15,-.35)
    place(s,resonator(1174.66,.85,'metal'),1.25,.037,.52)
    stages[5]=(s,'Back passage: close soft cloth brush and one tiny very remote bell.')
    s=canvas(3.35)
    place(s,soft_scrape(.82,'leaf'),.29,.14,.2)
    for offset,freq in [(.78,1017),(1.77,914),(2.56,1104)]:
        drop=resonator(freq,.31,'glass')+.10*resonator(freq*.52,.31,'wood')
        place(s,drop,offset,.08,RNG.uniform(-.5,.5))
    stages[6]=(s,'Conservatory edge: fine foliage rustle and three small isolated water drops.')
    s=canvas(3.3)
    motif=chimes([220,261.626,329.628],[0,.61,1.27],2.8,'felt',[.7,.61,.52])
    place(s,motif,.12,.35,.5)
    stages[7]=(s,'Music room: distant incomplete A3-C4-E4 lullaby phrase, exactly the shared A-C-E motif; leaves space for the missing ending.')
    s=canvas(2.25)
    cup=resonator(1004,.62,'glass')+.17*resonator(1483,.62,'glass')
    place(s,cup,.31,.16,-.35)
    place(s,resonator(967,.43,'glass'),.73,.043,-.32)
    stages[8]=(s,'Dining corridor: tiny muted cup against saucer, with a quieter settling touch.')
    s=canvas(3.1); k=round(2.6*SR)
    wind=noise(k,270,1700)*np.sin(np.linspace(0,np.pi,k))**2
    place(s,wind,.15,.055,-.4)
    stages[9]=(s,'Forked sitting room: a narrow draft through a wooden window gap; no scary swell.')
    s=canvas(3.5); k=round(2.9*SR); tt=np.arange(k)/SR
    window=(.015*np.sin(2*np.pi*185*tt)+.006*np.sin(2*np.pi*379*tt))*np.sin(np.pi*tt/2.9)**2
    window+=noise(k,500,2800)*(.018+.035*np.sin(np.pi*tt/2.9)**2)*np.sin(np.pi*tt/2.9)**2
    place(s,window,.2,1,.4)
    stages[10]=(s,'Glass passage: fine window vibration with a gentle rain wash drawing closer, then receding.')
    s=canvas(2.4)
    for offset,freq,vol in [(.41,1580,.08),(1.68,1280,.06)]:
        tick=resonator(freq,.13,'wood')*.4+noise(round(.13*SR),1100,3800)*envelope(round(.13*SR),.002,.009,.03)*.07
        place(s,tick,offset,vol,-.45)
    stages[11]=(s,'Under the clock: only two uneven faint clock ticks, no persistent metronome.')
    s=canvas(4.0); k=round(3.4*SR); tt=np.arange(k)/SR
    fading_rain=noise(k,480,2400)*np.sin(np.pi*tt/3.4)**2*np.linspace(.06,.012,k)
    place(s,fading_rain,.1,1,-.15)
    for offset,pitch in [(1.18,1900),(1.49,2180),(2.51,1680)]:
        btime=time(.19)
        phase=2*np.pi*(pitch*btime+170*np.sin(2*np.pi*7*btime)/(2*np.pi*7))
        bird=(np.sin(phase)+.12*np.sin(2*phase))*np.sin(np.pi*btime/.19)**2
        place(s,bird,offset,.008,.6)
    stages[12]=(s,'Garden hall: rain fading beyond the doorway and three extremely faint dawn bird syllables.')
    for index,(snd,description) in stages.items():
        save(outdir,f'amb_stage{index:02}.wav',snd,-19.5 if index != 11 else -22,description,False)
    (outdir/'sfx_metadata.json').write_text(json.dumps(META,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return META


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'rendered')
    args=parser.parse_args()
    result=create(args.output)
    print(json.dumps({'files':len(result),'seconds':round(sum(r['duration_seconds'] for r in result),2),'output':str(args.output)},ensure_ascii=False))
