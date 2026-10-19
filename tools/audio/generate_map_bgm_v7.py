#!/usr/bin/env python3
"""Five map/ending cues, newly scored for a recorded CC0 acoustic grand piano.

No lobby, hit/damage, or other SFX are read or written. Requires numpy/scipy and
the VCSL sample_manifest.json retained with this authoring project. Output game
files are self-contained WAVs; a sample engine is not needed at game runtime.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, gaussian_filter1d
from scipy.io import wavfile

SR=22050
SEED=2026101907
CONFIG={
    "act1":(1380522,-28.2,True),
    "act2":(1221231,-27.6,True),
    "act3":(1512000,-28.9,True),
    "stage12":(1270080,-27.5,True),
    "ending":(1477350,-27.0,False),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seeded(*parts):
    digest=hashlib.sha256("|".join(map(str,(SEED,*parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],"little"))


class RecordedPiano:
    """Close-root sample playback with a soft/medium/mf timbre crossfade.

    Original stereo channels are preserved; neither chorusing nor mono folding
    is used. Key release damps the recorded tail smoothly. No per-key peak/RMS
    normalization erases the performance dynamics of the recorded instrument.
    """
    def __init__(self,manifest):
        self.manifest_path=Path(manifest)
        self.manifest=json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("license")!="CC0-1.0":
            raise ValueError("Expected the verified CC0-1.0 VCSL piano manifest")
        self.entries=self.manifest["samples"]
        self.used={}
        # The source's pp/mp/mf recordings have a wide natural level range.
        # This gentle keyboard-response curve lets accompaniment use the mp
        # body quietly instead of reducing all inner lines to near-silent pp.
        self.layer_centers={1:.18,2:.38,3:.75}
        self.available_layers=sorted(set(e["velocity_layer"] for e in self.entries))

    @lru_cache(maxsize=96)
    def _root_wave(self,path_string):
        path=Path(path_string)
        rate,pcm=wavfile.read(path)
        if np.issubdtype(pcm.dtype,np.integer):
            wave=pcm.astype(np.float32)/float(2**(pcm.dtype.itemsize*8-1))
        else:
            wave=pcm.astype(np.float32)
        if wave.ndim==1:
            wave=np.repeat(wave[:,None],2,axis=1)
        if wave.shape[1]!=2:
            raise ValueError(f"Expected mono or stereo piano source: {path}")
        if rate!=SR:
            factor=Fraction(SR,rate)
            wave=signal.resample_poly(wave,factor.numerator,factor.denominator,axis=0)
        # Remove only a constant offset. Retain the real key attack and stereo.
        wave-=np.mean(wave,axis=0,keepdims=True)
        return wave.astype(np.float32)

    @lru_cache(maxsize=192)
    def pitched(self,midi,layer):
        choices=[e for e in self.entries if e["velocity_layer"]==layer]
        entry=min(choices,key=lambda e:(abs(e["root_midi"]-midi),e["root_midi"]))
        delta=midi-entry["root_midi"]
        if abs(delta)>3:
            raise ValueError(f"Piano sample would be transposed too far: MIDI{midi}, root{entry['root_midi']}")
        path=Path(entry.get("local_path",entry.get("path",entry.get("relative_path",""))))
        if not path.is_absolute():
            path=self.manifest_path.parent/path
        if not path.exists() and entry.get("relative_path"):
            path=self.manifest_path.parent/entry["relative_path"]
        if not path.exists() and entry.get("source_path"):
            path=self.manifest_path.parent/"VCSL"/entry["source_path"]
        if entry.get("sha256") and sha256(path)!=entry["sha256"]:
            raise ValueError(f"Source sample hash differs from manifest: {path}")
        wave=self._root_wave(str(path))
        if delta:
            factor=Fraction(2**(-delta/12)).limit_denominator(256)
            wave=signal.resample_poly(wave,factor.numerator,factor.denominator,axis=0)
        record={"root_midi":entry["root_midi"],"velocity_layer":layer,"source_path":entry.get("source_path",str(path)),
                "source_url":entry.get("source_url"),"sha256":entry.get("sha256",sha256(path)),
                "pitch_shift_semitones":delta}
        self.used[(midi,layer)]=record
        return wave.astype(np.float32)

    def note(self,midi,seconds,velocity,articulation="legato",pedal=.0):
        usable_layers=[layer for layer in self.available_layers if min(abs(e["root_midi"]-midi) for e in self.entries if e["velocity_layer"]==layer)<=3]
        if not usable_layers:
            raise ValueError(f"No recorded piano root close enough to MIDI{midi}")
        centers=np.array([self.layer_centers.get(layer,.18+.25*(layer-1)) for layer in usable_layers])
        upper=int(np.searchsorted(centers,velocity))
        if upper==0:
            layers=[(usable_layers[0],1.0)]
        elif upper>=len(centers):
            layers=[(usable_layers[-1],1.0)]
        else:
            lower=upper-1
            weight=float((velocity-centers[lower])/(centers[upper]-centers[lower]))
            layers=[(usable_layers[lower],1-weight),(usable_layers[upper],weight)]
        release={"legato":.38,"soft":.47,"detached":.21,"muted":.15}[articulation]
        release+=.18*pedal+max(0,60-midi)*.002
        hold=seconds+.15*pedal
        count=round((hold+4.5*release)*SR)
        out=np.zeros((count,2),dtype=np.float32)
        provenance=[]
        for layer,weight in layers:
            wave=self.pitched(midi,layer)
            n=min(len(wave),count)
            out[:n]+=wave[:n]*weight
            provenance.append({"layer":layer,"weight":round(weight,6),**self.used[(midi,layer)]})
        t=np.arange(count)/SR
        damping=np.exp(-(np.maximum(t-hold,0)/release)**1.35)
        # Avoid a hard sample start or an abrupt last-frame damper cutoff.
        damping*=np.minimum(1,t/.002)
        damping*=np.minimum(1,np.maximum(0,(t[-1]-t)/.045))**2
        out*=damping[:,None]*(.48+velocity)
        return out,provenance


class Score:
    def __init__(self,name,piano):
        frames,target,loop=CONFIG[name]
        self.name=name;self.piano=piano;self.frames=frames;self.target=target;self.loop=loop
        self.music_frames=frames if loop else frames-7*SR
        self.beat_seconds=self.music_frames/(SR*64)
        self.bpm=60/self.beat_seconds
        self.audio=np.zeros((frames,2),dtype=np.float32)
        self.rng=seeded(name)
        self.events=[]
        self.sections=[]

    def clock(self,beat):
        # A written four-bar breath: slightly faster mid-phrase, slower near
        # the edges. Ending's final two bars broaden, keeping the same total time.
        warped=beat+.115*np.sin(2*np.pi*beat/16)+.025*np.sin(2*np.pi*beat/4)
        if not self.loop:
            warped+=.92*((max(beat-56,0)/8)**2-beat/64)
        return warped*self.beat_seconds

    def note(self,bar,beat,midi,dur,v,role="melody",art="legato",pedal=.15,pan=None,why=""):
        absolute=bar*4+beat
        when=self.clock(absolute)
        # Roles have deliberate placement; at most4ms touch variation remains.
        when+=self.rng.normal(0,.0035)
        duration=max(.04,self.clock(absolute+dur)-self.clock(absolute))
        velocity=float(np.clip(v*self.rng.uniform(.985,1.015),.12,.82))
        audio,sources=self.piano.note(midi,duration,velocity,art,pedal)
        if pan is None:
            pan={"bass":-.07,"inner_voice":.05,"countermelody":.08,"melody":.02}.get(role,0)
        # Balance, rather than re-pan or widen the existing stereo recording.
        audio=audio*np.array([math.sqrt(1-pan),math.sqrt(1+pan)])
        index=round(when*SR)
        if self.loop:
            for off in range(0,len(audio),self.frames):
                part=audio[off:off+self.frames];start=(index+off)%self.frames
                count=min(len(part),self.frames-start)
                self.audio[start:start+count]+=part[:count]
                if count<len(part):self.audio[:len(part)-count]+=part[count:]
        else:
            start=max(0,index);off=max(0,-index);count=min(len(audio)-off,self.frames-start)
            if count>0:self.audio[start:start+count]+=audio[off:off+count]
        self.events.append({"bar":bar+1,"beat":beat,"absolute_beat":absolute,"time_seconds":round(when,7),
                            "duration_beats":dur,"duration_seconds":round(duration,7),"midi":midi,
                            "velocity":round(velocity,6),"role":role,"articulation":art,"pedal":pedal,
                            "stereo_balance":pan,"rationale":why,"recorded_samples":sources})

    def line(self,events,offset=0,transpose=0,role="melody",art="legato",pedal=.18):
        for item in events:
            b,t,m,d,v,*reason=item
            self.note(b+offset,t,m+transpose,d,v,role,art,pedal,why=reason[0] if reason else "")

    def section(self,bar,name,rationale):
        self.sections.append({"start_bar":bar+1,"name":name,"rationale":rationale})

    def finish(self,output):
        # Recorded acoustic grand already contains real body and room response.
        # Add only a small dark room; do not chorus/smear the stereo instrument.
        rng=seeded(self.name,"small_room")
        t=np.arange(round(1.8*SR))/SR
        result=self.audio.astype(np.float64)
        for ch in range(2):
            ir=signal.sosfilt(signal.butter(2,3600,fs=SR,output="sos"),rng.standard_normal(len(t)))
            ir*=np.exp(-t/.34)*(1-np.exp(-t/.023))
            ir*=.19/max(np.sqrt(np.sum(ir*ir)),1e-12)
            for delay,g in [(.047+ch*.007,.105),(.103-ch*.009,.052)]:ir[round(delay*SR)]+=g
            if self.loop:
                wet=np.fft.irfft(np.fft.rfft(self.audio[:,ch])*np.fft.rfft(ir,n=self.frames),n=self.frames)
            else:wet=signal.fftconvolve(self.audio[:,ch],ir)[:self.frames]
            result[:,ch]+=wet*.13
        # Remove subsonic/DC gently; preserve natural grand-piano body.
        f=np.fft.rfftfreq(self.frames,1/SR);hp=f*f/(f*f+38**2)
        result=np.fft.irfft(np.fft.rfft(result,axis=0)*hp[:,None],n=self.frames,axis=0)
        if not self.loop:
            n=round(.05*SR);result[:n]*=np.linspace(0,1,n)[:,None]
            n=round(2.5*SR);result[-n:]*=np.linspace(1,0,n)[:,None]**1.4
        rms=float(np.sqrt(np.mean(result*result)))
        gain=10**(self.target/20)/max(rms,1e-12)
        result*=gain
        # The real grand has rare20–40ms hammer crests much taller than the
        # preceding synthetic patch. Stereo-linked, smoothly rounded lookahead
        # control keeps these crests from forcing the whole bed5dB too quiet.
        # No samples are hard-clipped, and source-note gains are not normalized.
        transient_gain=np.ones(self.frames)
        mode="wrap" if self.loop else "nearest"
        for _ in range(2):
            peaks=np.max(np.abs(result),axis=1)
            requested=np.minimum(1.,.49/np.maximum(peaks,1e-12))
            held=minimum_filter1d(requested,size=round(.060*SR)|1,mode=mode)
            smooth=gaussian_filter1d(held,sigma=.006*SR,truncate=4,mode=mode)
            result*=smooth[:,None];transient_gain*=smooth
            actual=float(np.sqrt(np.mean(result*result)))
            if actual<10**((self.target-.12)/20):
                adjust=min(1.16,10**(self.target/20)/max(actual,1e-12))
                result*=adjust;gain*=adjust
        # Final guard uses the same smooth linked envelope, rather than clipping.
        peaks=np.max(np.abs(result),axis=1)
        requested=np.minimum(1.,.49/np.maximum(peaks,1e-12))
        smooth=gaussian_filter1d(minimum_filter1d(requested,size=round(.060*SR)|1,mode=mode),sigma=.006*SR,truncate=4,mode=mode)
        result*=smooth[:,None];transient_gain*=smooth
        pcm=np.rint(result*32767).astype(np.int16)
        path=output/f"bgm_{self.name}.wav";wavfile.write(path,SR,pcm)
        x=pcm.astype(float)/32768.;peak=float(np.max(abs(x)));rms=float(np.sqrt(np.mean(x*x)))
        edge=np.concatenate([x[-round(.1*SR):],x[:round(.1*SR)]])
        p99=float(np.quantile(abs(np.diff(edge,axis=0)),.99));delta=float(np.max(abs(x[0]-x[-1])))
        edge_rms={}
        for seconds in [.1,.25,.5,1.,2.]:
            n=round(seconds*SR);a=np.sqrt(np.mean(x[:n]**2));b=np.sqrt(np.mean(x[-n:]**2))
            edge_rms[str(seconds)]={"start_dbfs":round(20*np.log10(max(a,1e-12)),4),"end_dbfs":round(20*np.log10(max(b,1e-12)),4),"difference_db":round(abs(20*np.log10(max(a,1e-12)/max(b,1e-12))),4)}
        roles=Counter(e["role"] for e in self.events)
        meta={"file":path.name,"sample_rate":SR,"channels":2,"encoding":"PCM16","frames":len(pcm),
              "duration_seconds":len(pcm)/SR,"bpm":self.bpm,"meter":"4/4","bars":16,"loop":self.loop,
              "peak_dbfs":round(20*np.log10(peak),5),"rms_dbfs":round(20*np.log10(rms),5),
              "target_rms_dbfs":self.target,"master_gain":gain,"clipped_samples":int(np.sum((pcm==-32768)|(pcm==32767))),
              "transient_control":{"method":"Stereo-linked60ms lookahead minimum hold,6ms Gaussian gain smoothing; no hard clipping","ceiling_dbfs":20*np.log10(.49),"maximum_gain_reduction_db":round(-20*np.log10(max(float(np.min(transient_gain)),1e-12)),4),"fraction_below_minus1db":round(float(np.mean(transient_gain<10**(-1/20))),6)},
              "boundary_sample_delta":delta,"boundary_vs_local99_ratio":delta/max(p99,1e-12),"edge_rms":edge_rms,
              "events":len(self.events),"role_counts":dict(roles),"sha256":sha256(path),"sections":self.sections}
        (output/f"bgm_{self.name}.score.json").write_text(json.dumps({**meta,"notes":sorted(self.events,key=lambda e:e['time_seconds'])},ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"{path.name}: {meta['duration_seconds']:.5f}s; {roles['melody']} melody notes; RMS{meta['rms_dbfs']}dBFS; peak{meta['peak_dbfs']}dBFS",flush=True)
        return meta


# Dotted-quarter / eighth / half-note fingerprint. The four-bar sentence has a
# question, falling fragment, two-note breath, and suspended closing note.
QUESTION=[(0,.5,69,1.25,.53),(0,2.,72,.42,.55),(0,2.5,76,1.3,.62),
          (1,.5,74,.45,.51),(1,1.25,72,.85,.49),(1,2.75,71,.85,.45),
          (2,.5,67,.45,.42),(2,1.25,69,1.3,.47),(3,1.0,71,1.5,.40)]
ANSWER=[(0,.5,72,.85,.54),(0,1.75,71,.42,.48),(0,2.5,69,1.3,.46),
        (1,.5,67,.6,.43),(1,1.5,69,.6,.47),(1,2.5,72,1.1,.52),
        (2,.5,76,1.3,.57),(2,2.5,74,.85,.48),(3,.75,71,1.5,.41)]


def bass_and_inner(s,basses,inners,kind):
    """Every pitch is written: no greedy nearest-tone algorithm can get stuck."""
    for bar,(bass,inner) in enumerate(zip(basses,inners)):
        pos=bar%4;phrase=bar//4
        if kind=="act1":
            if pos!=3 or phrase==1:
                s.note(bar,.22 if bar else .75,bass,1.5,.34,"bass","soft",.26)
            if bar not in [3,7,11]:
                s.note(bar,2.1,inner,1.25,.30,"inner_voice","soft",.2)
            if bar in [1,5,9,13]:
                s.note(bar,3.1,inner+7,.65,.24,"countermelody","soft",.12)
        elif kind=="act2":
            s.note(bar,.25 if bar else .7,bass,1.6,.35,"bass","soft",.30)
            s.note(bar,1.75,inner,.95,.31,"inner_voice","legato",.22)
            if pos in [1,3]:
                s.note(bar,2.8,inner+7,.5,.27,"countermelody","detached",.05)
                s.note(bar,3.5,inner+12,.4,.25,"countermelody","detached",.05)
        elif kind=="act3":
            if bar%2==0:
                s.note(bar,.55 if bar else .85,bass,1.05,.34,"bass","muted",.02)
            if bar in [1,3,5,8,10,13,14]:
                s.note(bar,2.1,inner,.75,.27,"inner_voice","detached",.02)
            if bar in [6,11]:
                s.note(bar,3.0,inner+12,.65,.23,"countermelody","muted",0)
        elif kind=="stage12":
            s.note(bar,.18 if bar else .7,bass,1.3,.34+.012*phrase,"bass","legato",.22)
            # A stable soft three-part pulse supports the tune; pitches move.
            s.note(bar,1.45,inner,.65,.29+.008*phrase,"inner_voice","legato",.18)
            s.note(bar,2.6,inner+7,.55,.27+.008*phrase,"inner_voice","legato",.15)
            if pos in [1,2]:s.note(bar,3.35,inner+12,.4,.25,"countermelody","detached",.08)
        elif kind=="ending":
            if bar==15:continue
            s.note(bar,.25 if bar else .8,bass,1.6,.34,"bass","soft",.35)
            s.note(bar,1.65,inner,1.0,.30,"inner_voice","legato",.30)
            if bar in [1,5,9,13]:s.note(bar,2.85,inner+7,.6,.27,"countermelody","soft",.18)


def build_act1(piano):
    s=Score("act1",piano)
    bass_and_inner(s,[45,43,41,40,50,48,52,40,45,43,41,40,45,41,50,40],
                   [64,62,60,59,65,64,62,59,60,59,57,56,60,57,59,64],"act1")
    s.section(0,"question","Low A-C-E with dotted/eighth/held identity; cautious response and genuine rests.")
    s.line(QUESTION,transpose=-12,art="soft",pedal=.22)
    s.section(4,"question_variation","A changed return lifts the third note and leaves a different suspended ending.")
    s.line([(0,.5,57,1.25,.52),(0,2.,60,.42,.55),(0,2.5,64,1.3,.59),
            (1,.5,65,.65,.49),(1,1.5,64,.85,.47),(2,.5,62,1.1,.44),
            (2,2.25,60,.55,.42),(3,.75,59,1.5,.38)],offset=4,art="soft",pedal=.20)
    s.section(8,"answer_fragment","C-B-A answer turns through F/E uncertainty instead of closing the story.")
    s.line([(0,.5,60,.85,.51),(0,1.75,59,.42,.45),(0,2.5,57,1.3,.43),
            (1,.5,55,.65,.40),(1,1.75,57,.65,.45),(2,.5,60,1.25,.49),
            (2,2.5,59,.75,.42),(3,.75,56,1.5,.37)],offset=8,art="soft",pedal=.18)
    s.section(12,"return_open","The question returns softly; B/E continue naturally into the next loop.")
    s.line([(0,.5,57,1.25,.49),(0,2.,60,.42,.52),(0,2.5,64,1.3,.56),
            (1,.5,62,.65,.46),(1,1.75,60,1.0,.43),(2,.5,59,1.25,.40),
            (3,.65,59,.7,.40)],offset=12,art="soft",pedal=.26)
    # A real dominant pickup, not an inaudible held tone: B-G#-E-B prepares
    # the return to A. The last short B has a comparable mp touch to the theme.
    s.line([(15,1.65,56,.55,.40),(15,2.5,64,.6,.44),(15,3.25,59,1.3,.44)],
           role="loop_turnaround",art="soft",pedal=.20)
    return s


def build_act2(piano):
    s=Score("act2",piano)
    bass_and_inner(s,[45,43,45,43,41,45,40,43,41,48,50,43,40,41,50,43],
                   [64,62,60,59,57,60,59,62,60,64,65,62,64,62,65,64],"act2")
    s.section(0,"shared_question","The same identifiable rhythm is heard in a warmer upper piano register.")
    s.line(QUESTION,pedal=.24)
    s.section(4,"friend_answer","A C-B-A answer and a short separate upper reply suggest a second presence.")
    s.line(ANSWER,offset=4,pedal=.26)
    s.line([(1,3.25,79,.4,.27),(2,.0,81,.55,.29),(2,1.,79,.55,.28),(3,2.75,76,.65,.25)],offset=4,role="countermelody",art="soft",pedal=.10)
    s.section(8,"lifted_question","The motif expands upward through G and F before it gently falls.")
    s.line([(0,.5,69,1.25,.54),(0,2.,72,.42,.57),(0,2.5,76,1.3,.63),
            (1,.5,79,.85,.57),(1,1.75,76,.65,.51),(2,.5,77,.8,.52),
            (2,1.75,76,.55,.48),(2,2.75,74,.75,.45),(3,.75,72,1.5,.43)],offset=8,pedal=.24)
    s.section(12,"friend_return","The descending answer remains suspended on D/E so the loop remains open.")
    s.line([(0,.5,72,.85,.51),(0,1.75,71,.42,.46),(0,2.5,69,1.3,.44),
            (1,.5,67,.65,.40),(1,1.75,69,.65,.44),(2,.5,72,1.1,.48),
            (2,2.5,74,.85,.45),(3,1.0,76,1.0,.40)],offset=12,pedal=.25)
    s.line([(15,2.35,64,.55,.43),(15,3.2,67,1.4,.48)],
           role="loop_turnaround",art="soft",pedal=.23)
    return s


def build_act3(piano):
    s=Score("act3",piano)
    bass_and_inner(s,[45,43,41,40,50,48,52,40,45,43,41,40,45,41,50,40],
                   [64,62,60,59,65,64,62,64,60,59,57,56,60,57,59,64],"act3")
    s.section(0,"broken_question","The rhythm remains recognisable; a missing answer and small E/F friction create unease.")
    s.line([(0,.75,57,1.15,.49),(0,2.25,60,.4,.51),(0,2.75,64,1.1,.57),
            (1,1.0,65,.55,.45),(1,2.0,64,.85,.42),(2,1.25,59,1.0,.39),
            (3,1.0,60,.75,.37)],art="detached",pedal=.04)
    s.section(4,"quiet_answer","C-B-A arrives as a short, muted answer rather than a permanent pad.")
    s.line([(0,.75,60,.8,.48),(0,2.,59,.4,.42),(0,2.75,57,1.1,.40),
            (1,1.0,55,.6,.35),(2,.75,57,.75,.40),(2,2.25,59,.55,.38),
            (3,1.0,56,1.0,.34)],offset=4,art="detached",pedal=.02)
    s.section(8,"question_in_shadow","The theme returns with a held E and a carefully placed half-step sigh.")
    s.line([(0,.75,57,1.15,.47),(0,2.25,60,.4,.50),(0,2.75,64,1.1,.55),
            (1,1.0,65,.45,.43),(1,1.75,64,.8,.40),(2,.75,62,.75,.41),
            (2,2.25,60,.55,.37),(3,1.0,59,1.0,.34)],offset=8,art="detached",pedal=.04)
    s.section(12,"unfinished_memory","A shorter sentence withdraws into B/E; real space remains between notes.")
    s.line([(0,.75,57,1.15,.44),(0,2.25,60,.4,.47),(0,2.75,64,1.0,.52),
            (1,1.1,62,.75,.39),(2,1.0,60,1.1,.36)],offset=12,art="detached",pedal=.04)
    s.line([(15,1.6,64,.6,.41),(15,2.45,56,.55,.40),(15,3.25,59,1.2,.43)],
           role="loop_turnaround",art="soft",pedal=.12)
    return s


def build_stage12(piano):
    s=Score("stage12",piano)
    bass_and_inner(s,[41,45,40,43,41,43,41,47,40,41,45,43,40,45,47,43],
                   [57,60,59,62,60,64,62,65,64,65,64,62,60,64,62,64],"stage12")
    s.section(0,"last_journey","Clear A-C-E rhythm rides a quiet regular pulse instead of floating single notes.")
    s.line(QUESTION,pedal=.24)
    s.section(4,"opening_horizon","A rising A-B-C-D-E inner phrase gives the harmony a destination.")
    s.line([(0,.5,72,.85,.54),(0,1.75,71,.42,.49),(0,2.5,69,1.3,.47),
            (1,.5,72,.75,.52),(1,1.75,74,.75,.55),(2,.5,76,1.25,.60),
            (2,2.5,77,.75,.58),(3,.5,76,1.0,.53),(3,2.0,74,1.0,.48)],offset=4,pedal=.26)
    s.line([(0,3.4,57,.5,.27),(1,.7,59,.5,.28),(1,2.8,60,.5,.29),
            (2,1.2,62,.55,.30),(3,.0,64,.8,.31)],offset=4,role="countermelody",art="legato",pedal=.18)
    s.section(8,"hope_crest","The motif grows to G5, then relinquishes the crest instead of sustaining a loud layer.")
    s.line([(0,.5,69,1.25,.55),(0,2.,72,.42,.58),(0,2.5,76,1.3,.64),
            (1,.5,79,1.25,.62),(1,2.5,77,.65,.55),(2,.5,76,.85,.53),
            (2,1.75,74,.65,.49),(3,.75,72,1.5,.47)],offset=8,pedal=.28)
    s.section(12,"toward_dawn","A major-side answer stays open, ready to cross back to the beginning.")
    s.line([(0,.5,72,.85,.51),(0,1.75,71,.42,.47),(0,2.5,69,1.3,.45),
            (1,.5,72,.75,.49),(1,1.75,76,.85,.53),(2,.5,74,1.25,.47),
            (3,.65,76,.8,.40)],offset=12,pedal=.25)
    s.line([(15,1.9,60,.55,.43),(15,2.55,64,.55,.43),(15,3.05,69,1.7,.47)],
           role="loop_turnaround",art="soft",pedal=.22)
    return s


def build_ending(piano):
    s=Score("ending",piano)
    bass_and_inner(s,[45,43,41,43,40,40,41,43,40,41,45,43,43,45,47,48],
                   [64,62,60,62,64,60,57,59,60,62,64,62,64,65,62,60],"ending")
    s.section(0,"recognition","The complete rhythmic question is played clearly on the recorded grand.")
    s.line(QUESTION,pedal=.30)
    s.section(4,"answered_lullaby","The C-B-A answer is now a complete remembered musical sentence.")
    s.line(ANSWER,offset=4,pedal=.32)
    s.section(8,"friends_in_morning","The upper-register expansion is answered by a descending line.")
    s.line([(0,.5,69,1.25,.54),(0,2.,72,.42,.58),(0,2.5,76,1.3,.64),
            (1,.5,79,1.1,.61),(1,2.0,76,.8,.54),(2,.5,74,.85,.50),
            (2,1.75,72,.65,.47),(3,.75,71,1.5,.44)],offset=8,pedal=.32)
    s.section(12,"completed_cadence","The final G-B-D to C-E-G cadence broadens with a real written ritardando.")
    s.line([(0,.5,67,.85,.46),(0,1.75,71,.5,.49),(0,2.5,74,1.1,.53),
            (1,.5,76,1.0,.54),(1,2.0,74,.8,.48),(2,.5,71,1.6,.43),
            (3,.35,72,3.4,.43)],offset=12,pedal=.38)
    s.note(15,.05,48,4.1,.34,"final_bass","soft",.45)
    for t,m,v in [(.12,55,.29),(.20,60,.30),(.28,64,.29),(.38,67,.27)]:
        s.note(15,t,m,4.0-t,v,"final_chord","soft",.45)
    return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sample-manifest","--samples",dest="samples",type=Path,default=Path(__file__).resolve().parents[1]/"samples/sample_manifest.json")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"rendered")
    parser.add_argument("--tracks",nargs="*",choices=list(CONFIG))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    piano=RecordedPiano(args.samples)
    records=[]
    for name,builder in [("act1",build_act1),("act2",build_act2),("act3",build_act3),("stage12",build_stage12),("ending",build_ending)]:
        if args.tracks and name not in args.tracks:continue
        records.append(builder(piano).finish(args.output))
    # A partial authoring rerender must not silently drop the untouched ending
    # from the delivered five-track inventory or its recorded-source references.
    by_file={r["file"]:r for r in records}
    mappings={(m,l):{"midi":m,"layer":l,**e} for (m,l),e in piano.used.items()}
    for name in CONFIG:
        path=args.output/f"bgm_{name}.score.json"
        if path.exists():
            score=json.loads(path.read_text(encoding="utf-8"))
            by_file.setdefault(score["file"],{k:v for k,v in score.items() if k!="notes"})
            for event in score["notes"]:
                for source in event["recorded_samples"]:
                    key=(event["midi"],source["layer"])
                    mappings.setdefault(key,{"midi":event["midi"],"layer":source["layer"],**{k:v for k,v in source.items() if k not in ("layer","weight")}})
    records=[by_file[f"bgm_{name}.wav"] for name in CONFIG if f"bgm_{name}.wav" in by_file]
    metadata={"version":"map_bgm_v7","seed":SEED,"scope":"Five map/ending tracks only. Lobby, hit/damage and every SFX are excluded.",
              "composition":"New sixteen-bar4/4 questions/answers; dotted/eighth/held A-C-E identity, C-B-A response, explicitly moving bass/inner lines.",
              "instrument":"Recorded acoustic Kawai grand piano from VCSL, soft/medium/mf layers; close-key resampling, original stereo preserved; smooth key-release damping.",
              "sample_manifest_sha256":sha256(args.samples),"source_repo":piano.manifest.get("source_repo"),"source_commit":piano.manifest.get("commit"),"sample_license":"CC0-1.0",
              "used_sample_mappings":[e for _,e in sorted(mappings.items())],
              "human_listening_performed":False,"limits":"Recorded source improves the actual instrument source, but numeric QC and composition review do not prove listener preference. No claim of a human pianist performing the composed pieces.",
              "tracks":records}
    (args.output/"map_bgm_v7_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
