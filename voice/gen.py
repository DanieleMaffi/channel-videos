# Free voice job for the channel videos (runs on GitHub Actions, CPU).
# A request file voice/requests/<name>.txt holds:
#   engine: xtts | f5            (first line)
#   speaker: <XTTS studio speaker>   (xtts only, optional; default "Damien Black")
#   01|First spoken line
#   02|Second spoken line
# Output: voice/out/<name>/<engine>/<id>.wav (24 kHz mono). The daily run downloads them and trims/normalises.
# Licences: XTTS-v2 (Coqui Public Model License) and the F5-TTS pretrained model (CC BY-NC) are NON-COMMERCIAL:
# only for channels that are not monetized.
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

done = 0
for req in sorted(glob.glob('voice/requests/*.txt')):
    name = os.path.splitext(os.path.basename(req))[0]
    meta, lines = parse(req)
    engines = [e.strip() for e in meta.get('engine', 'xtts').split(',')]
    for eng in engines:
        outdir = os.path.join('voice', 'out', name, eng if eng != 'xtts' else 'xtts-' + meta.get('speaker', 'Damien Black').replace(' ', '_'))
        if os.path.isdir(outdir) and len(glob.glob(os.path.join(outdir, '*.wav'))) >= len(lines):
            continue
        os.makedirs(outdir, exist_ok=True)
        print('generating', name, eng, flush=True)
        if eng == 'xtts':
            run_xtts(lines, outdir, meta.get('speaker', 'Damien Black'))
        elif eng == 'f5':
            run_f5(lines, outdir)
        done += 1
print('jobs done:', done)
