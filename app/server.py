"""Starts and stops the local llama-servers: LFM3-Audio (hears and speaks) and, in the split
design, the small text model that writes the words.

They listen on 127.0.0.1 only and run with --offline, so nothing leaves this computer.
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


def base_url(cfg, port=None):
    return f'http://{cfg.server.host}:{port or cfg.server.port}'


def _get(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def healthy(cfg, timeout=1.0, port=None):
    return (_get(base_url(cfg, port) + '/health', timeout) or {}).get('status') == 'ok'


def voice_slots(cfg):
    """Requests the voice server serves at once (the split design needs 2: transcribe and speak)."""
    return 1 if cfg.brain.mode == 'audio' else 2


def _launch(cfg, args, port, log_path, wait_s):
    p = cfg.paths
    if not p.llama_server.exists():
        raise FileNotFoundError(f'missing {p.llama_server} - run tools/fetch_models.sh first')
    cmd = [str(p.llama_server), *args, '--host', cfg.server.host, '--port', str(port),
           '-ngl', str(cfg.server.gpu_layers), '--no-webui', '--offline']
    log = open(log_path, 'w', encoding='utf8')
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(p.llama_server.parent),
                            creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f'llama-server exited early, see {log_path}')
        if healthy(cfg, port=port):
            return proc
        time.sleep(0.25)
    proc.kill()
    raise TimeoutError(f'llama-server did not start within {wait_s}s, see {log_path}')


def needs_voice(cfg):
    """Is the voice server (LFM3-Audio, graphics card) needed? Not when hearing and speaking both
    run on the processor (brain.hearing='moonshine', brain.speech='pocket')."""
    b = cfg.brain
    return b.mode == 'audio' or b.hearing != 'moonshine' or b.speech != 'pocket'


def start(cfg, log_path, wait_s=90):
    """Starts the voice server. Returns its process, or None if one is already running."""
    if healthy(cfg):
        slots = (_get(base_url(cfg) + '/props', 2.0) or {}).get('total_slots', 1)
        if slots < voice_slots(cfg):
            raise RuntimeError(f'a voice server with {slots} slot is already running on port {cfg.server.port}; '
                               'stop it first (the split design needs 2)')
        return None
    p = cfg.paths
    for f in (p.model, p.mmproj):
        if not f.exists():
            raise FileNotFoundError(f'missing {f} - run tools/fetch_models.sh first')
    return _launch(cfg, ['-m', str(p.model), '--mmproj', str(p.mmproj), '-c', str(cfg.server.context),
                         '-np', str(voice_slots(cfg))], cfg.server.port, log_path, wait_s)


def start_text(cfg, log_path, wait_s=60):
    """Starts the text model's server (split design only). Returns its process, or None."""
    b = cfg.brain
    if b.mode == 'audio' or healthy(cfg, port=b.text_port):
        return None
    if not b.text_model.exists():
        raise FileNotFoundError(f'missing {b.text_model} - run tools/fetch_models.sh first')
    return _launch(cfg, ['-m', str(b.text_model), '-c', str(b.text_context), '-np', '1', *b.text_args.split()],
                   b.text_port, log_path, wait_s)


def start_mind(cfg, log_path, wait_s=90):
    """Starts the note-taking model's server (brain.mind). Returns its process, or None."""
    b = cfg.brain
    if b.mode == 'audio' or not b.mind or healthy(cfg, port=b.mind_port):
        return None
    if not b.mind_model.exists():
        raise FileNotFoundError(f'missing {b.mind_model} - run tools/fetch_models.sh first')
    return _launch(cfg, ['-m', str(b.mind_model), '-c', str(b.text_context), '-np', '1', *b.mind_args.split()],
                   b.mind_port, log_path, wait_s)


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
