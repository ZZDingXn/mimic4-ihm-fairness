"""Repair only the historical LSTM train-export ordering in a task-local copy.

Frozen validation and test CSV field strings are copied, not recomputed.
"""
from __future__ import annotations
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mimic4_ihm.model import ChannelWiseLSTM
from mimic4_ihm.train import MemmapDataset, predict_lstm

STUDY=Path(__file__).resolve().parents[1]
EXPECTED='844f71eb3c7a5001d9db9d88a793db004e8ff48332963b08b8b3232024793808'

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
    source=DATA/'cw_lstm/best_model.pt'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=EXPECTED:raise ValueError('Frozen LSTM checkpoint hash differs from recorded baseline evidence')
    device=torch.device(opts.device)
    checkpoint=torch.load(source,map_location=device,weights_only=False)
    args=checkpoint['args'];config=checkpoint['config']
    model=ChannelWiseLSTM(config['channel_indices'],dim=args['dim'],size_coef=args['size_coef'],dropout=args['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    metadata=pd.read_csv(DATA/'metadata.csv')
    indices=np.flatnonzero(metadata.split.eq('train'))
    loader=DataLoader(MemmapDataset(DATA/'lstm_X.npy',metadata.label.to_numpy(np.float32),indices),batch_size=8,shuffle=False,pin_memory=True)
    y,p,loss=predict_lstm(model,loader,device)
    np.testing.assert_array_equal(y,metadata.iloc[indices].label.to_numpy())
    original=DATA/'cw_lstm/predictions.csv'
    updates=dict(zip(metadata.iloc[indices].stay_id,p))
    output=study_path/'experiments/baseline_alignment/cw_lstm';output.mkdir(parents=True,exist_ok=True)
    # Preserve the original decimal strings on validation/test, avoiding even
    # float parse/format round-trip changes while repairing the training rows.
    with original.open(newline='',encoding='utf-8') as source_csv, (output/'predictions.csv').open('w',newline='',encoding='utf-8') as target_csv:
        reader=csv.DictReader(source_csv)
        writer=csv.DictWriter(target_csv,fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row['split']=='train':row['probability']=str(float(updates[int(row['stay_id'])]))
            writer.writerow(row)
    before=pd.read_csv(original);after=pd.read_csv(output/'predictions.csv')
    pd.testing.assert_frame_equal(before.loc[before.split.ne('train')],after.loc[after.split.ne('train')])
    (output/'alignment_manifest.json').write_text(json.dumps({'checkpoint_sha256':digest,'checkpoint_epoch':checkpoint['epoch'],'updated_split':'train','validation_test':'copied unchanged from frozen predictions','reason':'historical train inference used shuffled DataLoader while mapping to ordered stay IDs','train_loss':loss},indent=2),encoding='utf-8')
    print('Task-local baseline train alignment corrected; validation/test unchanged',flush=True)

if __name__=='__main__':main()
