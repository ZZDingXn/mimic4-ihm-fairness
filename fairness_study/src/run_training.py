"""Run the four frozen reweighing jobs serially with persistent logs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'fairness_study'

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--study-dir",type=Path,default=STUDY)
    parser.add_argument("--device",default="cuda")
    opts=parser.parse_args()
    DATA=opts.data_dir
    study_path=opts.study_dir
    (study_path/"experiments").mkdir(parents=True,exist_ok=True)
    metadata = pd.read_csv(DATA / 'metadata.csv')
    assert metadata.stay_id.is_unique
    assert metadata.groupby('subject_id')['split'].nunique().max() == 1
    np.testing.assert_array_equal(np.load(DATA / 'y.npy'), metadata.label.to_numpy())
    np.testing.assert_array_equal(np.load(DATA / 'split.npy').astype(str), metadata.split.to_numpy(str))
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'source': str(DATA), 'files': {}, 'jobs': []}
    for name in ['metadata.csv','config.json','lr/predictions.csv','cw_lstm/predictions.csv']:
        manifest['files'][name] = hashlib.sha256((DATA/name).read_bytes()).hexdigest()
    manifest['protocol_sha256'] = hashlib.sha256((study_path/'protocol.md').read_bytes()).hexdigest()
    manifest_path = study_path/'experiments'/'training_manifest.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    for attribute in ['insurance_group','ethnicity_group']:
        for model, cli_model in [('lr','logistic_regression'),('cw_lstm','cw_lstm')]:
            output = study_path/'experiments'/'reweight'/attribute/model
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable,'-u','-m','fairness_study.src.reweight','--attribute',attribute,'--model',cli_model,'--data-dir',str(DATA),'--output-dir',str(output),'--device',opts.device]
            job = {'attribute':attribute,'model':model,'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
            manifest['jobs'].append(job); save()
            print(f'START {attribute} {model}',flush=True)
            with (output/'console.log').open('w',encoding='utf-8') as log:
                process = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            job.update(exit_code=process.returncode,finished_utc=datetime.now(timezone.utc).isoformat(),status='complete' if process.returncode==0 else 'failed')
            save(); print(f'END {attribute} {model} exit={process.returncode}',flush=True)
            if process.returncode:
                raise RuntimeError(f'Training failed; inspect {output / "console.log"}')
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':
    main()
