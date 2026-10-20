"""사용자 제공 MP3 두 곡을 게임용 22050Hz 스테레오 PCM16 WAV로 변환한다.

음정·음량·길이를 편집하지 않고 FFmpeg의 무음 구간 메타데이터와 출처 정보를 보존한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR=22050
HERE=Path(__file__).resolve().parent
FILES=[
    ('fireside_rain_complete_90s_discord.mp3','bgm_lobby.wav',90.),
    ('collision_cat_long_600ms_project.mp3','sfx_ghost_contact.wav',.6),
]
def run(args):return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def probe(path):
    return json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)]).stdout)


def decode_float(path,rate):
    result=run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-map','0:a:0',
                '-ac','2','-ar',str(rate),'-c:a','pcm_f32le','-f','f32le','pipe:1'])
    x=np.frombuffer(result.stdout,dtype='<f4').reshape(-1,2).astype(np.float64)
    return x


def db(value,power=False):
    return round(float((10 if power else 20)*np.log10(max(float(value),1e-24))),6)


def silence_duration(x,threshold_db=-65,reverse=False):
    frame=np.max(abs(x),axis=1)
    if reverse:frame=frame[::-1]
    audible=np.flatnonzero(frame>=10**(threshold_db/20))
    return int(audible[0]) if len(audible) else len(x)


def metrics(x,rate):
    delta=abs(x[0]-x[-1]);adj=np.abs(np.diff(x,axis=0));p99=float(np.quantile(adj,.99))
    first=int(.05*rate);last=int(.05*rate)
    return {'frames':len(x),'sample_rate':rate,'channels':x.shape[1],'duration_seconds':len(x)/rate,
            'peak_amplitude':round(float(np.max(abs(x))),9),'peak_dbfs':db(np.max(abs(x))),
            'rms_dbfs':db(np.sqrt(np.mean(x*x))),'dc_mean_per_channel':[round(float(v),10) for v in np.mean(x,axis=0)],
            'first_sample_per_channel':[round(float(v),10) for v in x[0]],
            'last_sample_per_channel':[round(float(v),10) for v in x[-1]],
            'first_50ms_rms_dbfs':db(np.sqrt(np.mean(x[:first]**2))),
            'last_50ms_rms_dbfs':db(np.sqrt(np.mean(x[-last:]**2))),
            'leading_below_minus65db_seconds':silence_duration(x)/rate,
            'trailing_below_minus65db_seconds':silence_duration(x,reverse=True)/rate,
            'leading_below_minus80db_seconds':silence_duration(x,-80)/rate,
            'trailing_below_minus80db_seconds':silence_duration(x,-80,True)/rate,
            'loop_boundary_max_delta_amplitude':round(float(delta.max()),10),
            'loop_boundary_delta_dbfs':db(delta.max()),'adjacent_delta_99pct':round(p99,10),
            'loop_boundary_vs_adjacent_delta_99pct':round(float(delta.max())/max(p99,1e-20),6)}


def convert(source,target):
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:a:0',
             '-map_metadata','0','-ac','2','-ar',str(SR),'-c:a','pcm_s16le',str(target)]
    run(command)
    return command


def process(source_dir,outdir):
    outdir.mkdir(parents=True,exist_ok=True);records=[]
    for input_name,output_name,logical in FILES:
        source=source_dir/input_name;target=outdir/output_name
        source_probe=probe(source);stream=next(s for s in source_probe['streams'] if s['codec_type']=='audio')
        source_rate=int(stream['sample_rate']);native=decode_float(source,source_rate)
        target_reference=decode_float(source,SR)
        if np.max(abs(target_reference))>=1:
            raise ValueError(f'{input_name}: decoded samples exceed PCM16 range; no unauthorized normalization will be applied.')
        command=convert(source,target)
        rate,pcm=wavfile.read(target);x=pcm.astype(np.float64)/32768
        error=x-target_reference
        repeated=None
        with tempfile.TemporaryDirectory(prefix='faithful_repro_',dir=outdir.parent) as temp:
            second=Path(temp)/output_name;convert(source,second)
            repeated=hashlib.sha256(second.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
        target_stats=metrics(x,rate)
        # 파형 불연속을 검토 대상으로 표시할 뿐 이 판정으로 파일을 수정하지 않는다.
        seam_risk=(target_stats['loop_boundary_delta_dbfs']>-50 and target_stats['loop_boundary_vs_adjacent_delta_99pct']>2)
        records.append({'source_name':input_name,'output_name':output_name,
                        'source_path':str(source),'output_path':str(target),
                        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'source_probe':source_probe,
                        'source_embedded_tags':source_probe.get('format',{}).get('tags',{}),
                        'native_decoded':metrics(native,source_rate),'resampled_float_reference':metrics(target_reference,SR),
                        'output_pcm16':target_stats,
                        'expected_logical_seconds':logical,'expected_pcm_frames':round(logical*SR),
                        'container_padding_explanation':'MP3 container duration includes encoder delay/padding; FFmpeg automatically honors gapless skip/discard. No explicit -t or trim filter is used.',
                        'processing':{'eq':False,'gain_change':False,'normalization':False,'added_notes':False,'explicit_trim':False,
                                      'fade':False,'crossfade':False,'resampling':f'{source_rate} Hz to 22050 Hz using FFmpeg default sample-rate conversion',
                                      'output_codec':'pcm_s16le','channels':2},
                        'conversion_command':command,
                        'quantization_max_absolute_error':round(float(np.max(abs(error))),12),
                        'quantization_error_rms':round(float(np.sqrt(np.mean(error*error))),12),
                        'source_vs_output_peak_difference_db':round(target_stats['peak_dbfs']-metrics(target_reference,SR)['peak_dbfs'],6),
                        'source_vs_output_rms_difference_db':round(target_stats['rms_dbfs']-metrics(target_reference,SR)['rms_dbfs'],6),
                        'output_sample_dtype':str(pcm.dtype),
                        'clipped_pcm_samples':int(np.sum(abs(pcm.astype(np.int32))>=32767)),
                        'deterministic_wav_sha256_reproduction':repeated,'potential_loop_discontinuity_for_review':seam_risk,
                        'loop_edit_applied':False})
    report={'purpose':'Faithful format-only conversion of two user-supplied MP3s for game deployment.',
            'source_status':'User supplied. Embedded source and license statements are preserved verbatim; no claim that all source media are original synthesis. Independent license verification is outside this conversion.',
            'human_listening_performed':False,
            'ffmpeg_version':run(['ffmpeg','-version']).stdout.decode().splitlines()[0],
            'files':records}
    (outdir/'user_audio_conversion_metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    checks=[]
    for r in records:
        output=r['output_pcm16'];src=r['resampled_float_reference']
        checks.append({'file':r['output_name'],'format_ok':output['sample_rate']==SR and output['channels']==2,
                       'pcm16_ok':r['output_sample_dtype']=='int16',
                       'exact_logical_frame_count':output['frames']==r['expected_pcm_frames'],
                       'decoded_frame_count_preserved':output['frames']==src['frames'],
                       'no_pcm_clipping':r['clipped_pcm_samples']==0,
                       'error_within_half_lsb':r['quantization_max_absolute_error']<=.5001/32768,
                       'level_preserved':abs(r['source_vs_output_rms_difference_db'])<.01,
                       'reproducible_sha256':r['deterministic_wav_sha256_reproduction']})
    qc={'all_pass':all(all(v for k,v in row.items() if k!='file') for row in checks),'checks':checks,
        'crossfades_applied':False,'human_listening_performed':False,
        'loop_scope':'Waveform boundary/edge-silence measurements only. No claim of musically seamless phrase continuation; no automatic editing of user music.'}
    (outdir/'user_audio_conversion_qc.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report,qc


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path,required=True);parser.add_argument('--output',type=Path,default=HERE/'rendered')
    args=parser.parse_args();report,qc=process(args.source_dir,args.output)
    print(json.dumps({'output':str(args.output),'qc_pass':qc['all_pass'],
                      'files':[{'name':r['output_name'],'frames':r['output_pcm16']['frames'],'seconds':r['output_pcm16']['duration_seconds'],
                                'peak_dbfs':r['output_pcm16']['peak_dbfs'],'rms_dbfs':r['output_pcm16']['rms_dbfs'],
                                'loop_discontinuity_review':r['potential_loop_discontinuity_for_review']} for r in report['files']]},ensure_ascii=False))
