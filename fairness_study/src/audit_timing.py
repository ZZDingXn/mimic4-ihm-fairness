"""Describe label timing relative to the frozen 48-hour predictor window."""
import json
from pathlib import Path
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata",type=Path,required=True)
    opts=parser.parse_args()
    (STUDY/"results").mkdir(parents=True,exist_ok=True)
    metadata = pd.read_csv(opts.metadata,
                           parse_dates=['intime', 'deathtime', 'window_end'])
    early = metadata.label.eq(1) & metadata.deathtime.le(metadata.window_end)
    hours = (metadata.loc[early, 'deathtime']-metadata.loc[early, 'intime']).dt.total_seconds()/3600
    result = {
        'cohort_stays': len(metadata), 'events': int(metadata.label.sum()),
        'events_at_or_before_48h_window_end': int(early.sum()),
        'by_split': metadata.loc[early].groupby('split').size().to_dict(),
        'events_before_icu_intime': int((metadata.label.eq(1)&metadata.deathtime.lt(metadata.intime)).sum()),
        'early_death_midnight_timestamps': int((early & metadata.deathtime.dt.hour.eq(0)
            & metadata.deathtime.dt.minute.eq(0) & metadata.deathtime.dt.second.eq(0)).sum()),
        'early_death_after_icu_intime_hours_quantiles': hours.quantile([0, .25, .5, .75, 1]).to_dict(),
        'cohort_or_labels_changed': False,
        'interpretation': 'Timestamp audit only; no claim that each recorded death time is clinically exact. Frozen cohort retained.',
    }
    (STUDY/'results/temporal_label_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
