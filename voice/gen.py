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
asr = None
def words(t):
    import re
    from num2words import num2words
    t = re.sub(r'(\d+(?:\.\d+)?)', lambda m: ' ' + num2words(float(m.group(1)) if '.' in m.group(1) else int(m.group(1))) + ' ', t.lower().replace(',', ''))
    return re.sub(r"[^a-z' ]+", ' ', t.replace('-', ' ')).split()

def heard_score(path, text):
    # Transcribe the take and compare it with the script, so stumbles, slurs and skipped words are caught.
    global asr
    import difflib
    from faster_whisper import WhisperModel
    if asr is None:
        asr = WhisperModel('base.en', device='cpu', compute_type='int8')
    segs, _ = asr.transcribe(path, beam_size=5, language='en')
    heard = ' '.join(s.text for s in segs)
    return difflib.SequenceMatcher(None, words(text), words(heard)).ratio(), heard

def run_chatterbox(lines, outdir, ref, exag):
    global cb
    import shutil, torchaudio
    from chatterbox.tts import ChatterboxTTS
    if cb is None:
        cb = ChatterboxTTS.from_pretrained(device='cpu')
    prompt = None if ref == 'default' else f'voice/ref/{ref}.wav'
    report = []
    for i, t in lines:
        best = (-1, None, '')
        for attempt in range(4):   # up to 4 takes, keep the clearest one
            tmp = os.path.join(outdir, f'{i}.take{attempt}.wav')
            wav = cb.generate(t, audio_prompt_path=prompt, exaggeration=exag, cfg_weight=0.5)
            torchaudio.save(tmp, wav, cb.sr)
            try:
                score, heard = heard_score(tmp, t)
            except Exception as e:
                score, heard = 0.0, 'check failed: ' + repr(e)
            print(f'  {i} take {attempt + 1}: {score:.2f}  heard: {heard.strip()}', flush=True)
            if score > best[0]:
                best = (score, tmp, heard)
            if score >= 0.92:
                break
        shutil.copy(best[1], os.path.join(outdir, f'{i}.wav'))
        for f in glob.glob(os.path.join(outdir, f'{i}.take*.wav')):
            os.remove(f)
        report.append(f'{i}|{best[0]:.2f}|{best[2].strip()}')
    open(os.path.join(outdir, 'check.txt'), 'w', encoding='utf-8').write('\n'.join(report) + '\n')

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

CHECK_ONLY = len(sys.argv) > 1 and sys.argv[1] == '--pending'   # prints how many outputs are missing, generates nothing
pending = 0
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
        pending += 1
        if CHECK_ONLY:
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
if CHECK_ONLY:
    print(pending)
    sys.exit(0)
print('jobs done:', done)
