"""Standard-library guarded launcher; scientific configs immutable after deploy."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
from resources import inspect
ROOT=Path(__file__).resolve().parent
PYTHON='<local-venv>/bin/python'
SESSION='<tmux-session>'

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    if subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0:raise RuntimeError('Pipeline already running')
    resources=inspect();sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    for name in ['config.json','rl_config.json']:sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    deployment=ROOT/'deployment.json'
    if deployment.exists():
        if not a.resume:raise RuntimeError('Existing deployment; explicit --resume required')
        assert json.loads(deployment.read_text())['sources']==sources,'Source/config changed'
    else:
        (ROOT/'artifacts').mkdir(exist_ok=True);payload=os.urandom(1024*1024);path=ROOT/'artifacts/storage_probe.bin'
        with path.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        assert path.read_bytes()==payload
        deployment.write_text(json.dumps(dict(sources=sources,initial_resources=resources,runtime=PYTHON,started=time.time()),indent=2))
    command=shlex.join([PYTHON,'-u',str(ROOT/'runner.py')])+' >> '+shlex.quote(str(ROOT/'artifacts/pipeline.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',SESSION,command],check=True)
    print(json.dumps(dict(session=SESSION,node=resources['node'],cards=resources['selected'],root=str(ROOT)),indent=2))

if __name__=='__main__':main()
