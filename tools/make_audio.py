"""Generate the project's original synthesized music and cues (stdlib only)."""
from pathlib import Path
import math
import struct
import wave

OUT = Path(__file__).resolve().parents[1] / 'assets' / 'sound'
SR = 22050

def save(name, notes, length, gain=.22):
    values = [0.] * int(SR * length)
    for start, frequency, duration, volume in notes:
        for i in range(int(duration * SR)):
            k = int(start * SR) + i
            if k >= len(values):
                break
            t = i / SR
            envelope = min(t / .02, 1) * math.exp(-3 * t / duration) * min((duration-t)/.08, 1)
            sample = math.sin(2*math.pi*frequency*t) + .18*math.sin(2*math.pi*frequency*2*t)
            values[k] += gain * volume * envelope * sample
    raw = bytearray()
    for sample in values:
        v = int(max(-.95, min(.95, sample))*32767)
        raw.extend(struct.pack('<hh',v,v))
    with wave.open(str(OUT / (name+'.wav')), 'wb') as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes(raw)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = {'select':[523.25], 'collect':[659.25,987.77], 'open':[392,523.25,659.25],
            'damage':[220,164.81], 'return':[523.25,659.25,783.99,1046.5], 'fail':[392,329.63,261.63]}
    for name, freqs in cues.items():
        save(name,[(i*.09,f,.42,.65) for i,f in enumerate(freqs)],.8)
    melody = [329.63,392,493.88,440,392,329.63,293.66,329.63,261.63,329.63,392,440,392,329.63,293.66,246.94]
    notes=[(i*1.2,f,2.1,.32) for i,f in enumerate(melody)]
    notes += [(i*4.8,f,4.7,.18) for i,f in enumerate([164.81,146.83,130.81,146.83])]
    save('manor',notes,19.2,.20)

if __name__ == '__main__':
    main()
