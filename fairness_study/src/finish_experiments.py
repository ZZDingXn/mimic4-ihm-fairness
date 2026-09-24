"""Wait for the serial training manifest, then finish frozen evaluation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'fairness_study'

def main():
    manifest_path=STUDY/'experiments/training_manifest.json'
    while True:
        # The training writer replaces a short JSON file. Read only complete JSON.
        try:manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            time.sleep(1);continue
        if any(j['status']=='failed' for j in manifest['jobs']):
            raise RuntimeError('A training job failed; evaluation not started')
        if 'finished_utc' in manifest:break
        time.sleep(20)
    commands=[
        [sys.executable,'-u','-m','fairness_study.src.align_baseline_train','--data-dir',manifest['source'],'--study-dir',str(STUDY)],
        [sys.executable,'-u','-m','fairness_study.src.postprocess','--data-dir',manifest['source'],'--study-dir',str(STUDY)],
        [sys.executable,'-u','-m','fairness_study.src.evaluate','--study-dir',str(STUDY),'--iterations','10000','--seed','49297'],
    ]
    status={'steps':[]};status_path=STUDY/'experiments/completion_manifest.json'
    for i,command in enumerate(commands):
        step={'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
        status['steps'].append(step);status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        print('START '+command[3],flush=True)
        with (STUDY/'experiments'/f'completion_{i+1}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        step.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        print('END '+command[3]+' '+str(result.returncode),flush=True)
        if result.returncode:raise RuntimeError(f'Completion step failed, see completion_{i+1}.log')
    status['finished_utc']=datetime.now(timezone.utc).isoformat();status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')

if __name__=='__main__':main()
