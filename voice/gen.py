# Free voice job for the channel videos (runs on GitHub Actions, CPU).
# A request file voice/requests/<name>.txt holds:
#   engine: xtts | f5 | chatterbox | edge   (first line; several comma-separated)
#   speaker: <XTTS studio speaker>   (xtts only, optional; default "Damien Black")
#   refs: default, george, bm_lewis ...   (chatterbox: reference voices from voice/ref/<name>.wav; 'default' = built-in voice)
#   voices: en-US-AndrewMultilingualNeural, ...   (edge: Microsoft neural voice names)
#   exaggeration: 0.5   (chatterbox: expressiveness 0.25-1.0)
#   01|First spoken line
#   02|Second spoken line
# Output: voice/out/<name>/<engine>/<id>.wav (24 kHz mono). The daily run downloads them and trims/normalises.
# Licences: XTTS-v2 (Coqui Public Model License) and the F5-TTS pretrained model (CC BY-NC) are NON-COMMERCIAL:
# only for channels that are not monetized. Chatterbox (Resemble AI) is MIT: free also for commercial use.
# Edge voices use Microsoft's online read-aloud service via edge-tts (unofficial, no key).
import glob, os, subprocess, sys

def parse(path):
    meta, lines = {}, []
    for ln in open(path, encoding='utf-8'):
        ln = ln.strip()
        if not ln:
            continue
        if '|' in ln:
            i, t = ln.split('|', 1)
            lines.append((i.strip(), t.strip()))
        elif ':' in ln:
            k, v = ln.split(':', 1)
            meta[k.strip().lower()] = v.strip()
    return meta, lines

xtts = None
def run_xtts(lines, outdir, speaker):
    global xtts
    from TTS.api import TTS
    if xtts is None:
        xtts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    for i, t in lines:
        xtts.tts_to_file(text=t, speaker=speaker, language='en', file_path=os.path.join(outdir, f'{i}.wav'))

def run_f5(lines, outdir):
    ref = 'voice/ref/george.wav'
    ref_text = open('voice/ref/george.txt', encoding='utf-8').read().strip()
    for i, t in lines:
        subprocess.run(['f5-tts_infer-cli', '--model', 'F5TTS_v1_Base', '--ref_audio', ref, '--ref_text', ref_text,
                        '--gen_text', t, '--output_dir', outdir, '--output_file', f'{i}.wav', '--remove_silence'], check=True)

cb = None
def run_chatterbox(lines, outdir, ref, exag):
    global cb
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    if cb is None:
        cb = ChatterboxTTS.from_pretrained(device='cpu')
    prompt = None if ref == 'default' else f'voice/ref/{ref}.wav'
    for i, t in lines:
        wav = cb.generate(t, audio_prompt_path=prompt, exaggeration=exag, cfg_weight=0.5)
        torchaudio.save(os.path.join(outdir, f'{i}.wav'), wav, cb.sr)

def run_edge(lines, outdir, voice):
    for i, t in lines:
        mp3 = os.path.join(outdir, f'{i}.mp3')
        subprocess.run(['edge-tts', '--voice', voice, '--text', t, '--write-media', mp3], check=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-ar', '24000', '-ac', '1', os.path.join(outdir, f'{i}.wav')], check=True)
        os.remove(mp3)

def targets(eng, meta):
    if eng == 'xtts':
        sp = meta.get('speaker', 'Damien Black'); return [('xtts-' + sp.replace(' ', '_'), sp)]
    if eng == 'chatterbox':
        return [('chatterbox-' + r.strip(), r.strip()) for r in meta.get('refs', 'default').split(',')]
    if eng == 'edge':
        return [('edge-' + v.strip(), v.strip()) for v in meta.get('voices', 'en-US-AndrewMultilingualNeural').split(',')]
    return [(eng, None)]

done = 0
for req in sorted(glob.glob('voice/requests/*.txt')):
    name = os.path.splitext(os.path.basename(req))[0]
    meta, lines = parse(req)
    engines = [e.strip() for e in meta.get('engine', 'xtts').split(',')]
    only = os.environ.get('ONLY_ENGINE')
    if only:
        engines = [e for e in engines if e == only]
    for eng in engines:
      for sub, arg in targets(eng, meta):
        outdir = os.path.join('voice', 'out', name, sub)
        if os.path.isdir(outdir) and len(glob.glob(os.path.join(outdir, '*.wav'))) >= len(lines):
            continue
        os.makedirs(outdir, exist_ok=True)
        print('generating', name, sub, flush=True)
        try:
            if eng == 'xtts':
                run_xtts(lines, outdir, arg)
            elif eng == 'f5':
                run_f5(lines, outdir)
            elif eng == 'chatterbox':
                run_chatterbox(lines, outdir, arg, float(meta.get('exaggeration', '0.5')))
            elif eng == 'edge':
                run_edge(lines, outdir, arg)
            done += 1
        except Exception as e:
            print('FAILED', name, sub, repr(e), flush=True)
print('jobs done:', done)
