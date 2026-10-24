"""Paired blind solver-guided AI benchmark: chip vs ambe2-c, all 16 series."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from std_t98_cracker.burst import unpack49
from std_t98_cracker.enhanced import HeavyScorer
from std_t98_cracker.enhanced_sequence import CandidateSequence
from std_t98_cracker.guided_ai import GuidedAISolver
from std_t98_cracker.series_solver import SeriesSolver


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(frames, solver):
    session = solver.new_session(20, 6)
    decisions, checkpoints, first, timings = [], {}, None, []
    for i, burst in enumerate(frames):
        started = time.perf_counter()
        result = session.push(burst)
        timings.append((time.perf_counter()-started)*1000)
        decision = dict(series=result['series'], key=result['key'], end_burst=i+1, traffic_seconds=(i+1)*.08)
        if result['key'] is not None:
            if first is None:
                first = decision
            if not decisions or (decision['series'], decision['key']) != (decisions[-1]['series'], decisions[-1]['key']):
                decisions.append(decision)
        if i+1 in (12, 37, 62):
            checkpoints[str(i+1)] = decision
    return dict(first_confirmation=first, checkpoints=checkpoints, accepted_pairs=decisions, final=result,
                step_ms=timings, compute_seconds=sum(timings)/1000)


def classification(counts):
    """Exact-pair retrieval: wrong answer is both a false positive and a miss."""
    tp, fp = counts['correct'], counts['wrong']
    fn = counts['unresolved'] + fp
    total, answered = tp + fn, tp + fp
    return dict(tp=tp, fp=fp, fn=fn, answered=answered,
                answer_rate=answered/total if total else 0.,
                precision=tp/answered if answered else 0.,
                recall=tp/total if total else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def aggregate(rows):
    final = {outcome:sum(r['final_outcome']==outcome for r in rows)
             for outcome in ('correct','unresolved','wrong')}
    checkpoints = {str(n):{outcome:sum(r['checkpoint_outcomes'][str(n)]==outcome for r in rows)
                          for outcome in ('correct','unresolved','wrong')} for n in (12,37,62)}
    processing_times = []
    for row in rows:
        finish = 0.
        for i, milliseconds in enumerate(row['step_ms']):
            # Simulated serial processing if one burst arrives every 80 ms.
            # The model is preloaded. RF acquisition/USB stalls are not modeled.
            finish = max(finish, (i+1)*.08) + milliseconds/1000
            if row['first_outcome']=='correct' and row['first_confirmation']['end_burst']==i+1:
                processing_times.append(finish)
                break
    return dict(count=len(rows), final=final, final_metrics=classification(final),
                any_wrong_confirmation=sum(r['any_wrong_confirmation'] for r in rows),
                checkpoints=checkpoints,
                checkpoint_metrics={n:classification(counts) for n,counts in checkpoints.items()},
                first_correct_by_seconds={str(s):sum(r['first_outcome']=='correct' and
                    r['first_confirmation']['traffic_seconds']<=s for r in rows) for s in (1,3,5)},
                first_correct_with_processing_by_seconds={str(s):sum(t<=s for t in processing_times) for s in (1,3,5)},
                processing_time_assumption='80 ms burst arrivals, measured serial push latency, preloaded model; excludes RF/USB delays',
                step_ms={p:float(np.percentile([t for r in rows for t in r['step_ms']],q))
                         for p,q in (('median',50),('p95',95),('p99',99),('max',100))},
                compute_seconds=sum(r['compute_seconds'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chip',type=Path,required=True)
    p.add_argument('--software',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-series',type=int,default=16,help='Even number, half JSUT and half LibriSpeech')
    p.add_argument('--seed',type=int,default=20261025)
    args=p.parse_args()
    if args.per_series<2 or args.per_series%2:
        raise ValueError('per-series must be positive and even')
    pools={'jsut':[],'libri':[]}
    for path in sorted(args.chip.glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['split']=='eval' and meta['corpus'] in pools and int(meta['natural_frames'])//4>=62:
                pools[meta['corpus']].append((path,meta))
    rng=np.random.default_rng(args.seed)
    plan=[]
    for series in range(16):
        for corpus,pool in pools.items():
            for j,index in enumerate(rng.choice(len(pool),args.per_series//2,replace=False)):
                path,meta=pool[int(index)]
                # Natural speech only: no trailing PCM padding/silence.
                offset=int(rng.integers(0,int(meta['natural_frames'])//4-62+1))
                key=int(rng.integers(1,32768))
                if corpus=='jsut' and j<2:
                    key=(1,32767)[j]
                plan.append(dict(input=path.name,corpus=corpus,series=series,key=key,start_burst=offset,bursts=62))
    rng.shuffle(plan)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    started=time.perf_counter()
    scorer=HeavyScorer(device='auto')
    solver=GuidedAISolver(SeriesSolver(),scorer)
    # Exclude loading and the first CUDA kernel setup from steady-state latency.
    scorer.score(np.zeros((2,5,4,49),dtype=np.uint8),np.zeros((5,4,49),dtype=np.uint8))
    warmup=solver.new_session()
    for _ in range(12):warmup.push(np.zeros((4,49),dtype=np.uint8))
    startup=time.perf_counter()-started
    rows=[]
    with (args.output/'trials.jsonl').open('x',encoding='utf-8') as log:
        for index,case in enumerate(plan):
            for encoder,folder in (('thumbdv',args.chip),('ambe2-c',args.software)):
                path=folder/case['input']
                with np.load(path,allow_pickle=False) as z:
                    metadata=json.loads(str(z['metadata']))
                    start=case['start_burst']*4;end=start+248
                    if end>int(metadata['natural_frames']):raise ValueError('Natural speech too short')
                    plain=unpack49(z['bits'][start:end].reshape(62,4,7))
                encrypted=plain^CandidateSequence(case['series']).batch(np.array([case['key']]))[0]
                result=run(encrypted,solver)  # No path, series or key label is passed.
                def outcome(decision):
                    if decision is None or decision['key'] is None:return 'unresolved'
                    return 'correct' if (decision['series'],decision['key'])==(case['series'],case['key']) else 'wrong'
                row=dict(case=index,encoder=encoder,input_sha256=sha(path),**case,**result,
                         final_outcome=outcome(result['final']),first_outcome=outcome(result['first_confirmation']),
                         checkpoint_outcomes={k:outcome(v) for k,v in result['checkpoints'].items()},
                         any_wrong_confirmation=any(outcome(d)=='wrong' for d in result['accepted_pairs']))
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if (index+1)%8==0:
                print(index+1,'/',len(plan),'paired trials;', 'wrong',sum(r['any_wrong_confirmation'] for r in rows),flush=True)
    summary={encoder:aggregate([r for r in rows if r['encoder']==encoder]) for encoder in ('thumbdv','ambe2-c')}
    root=Path(__file__).resolve().parents[1]
    report=dict(schema='guided-ai-paired-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                seed=args.seed,threshold=20,margin=6,method='guided-ai',labels_used_by_solver=False,
                frame_errors_added=False,unique_utterances=len({r['input'] for r in plan}),
                verifier=scorer.describe(),model_startup_and_warmup_seconds=startup,
                software_encoding=json.loads((args.software/'encoding.json').read_text()),
                summary=summary,by_series={encoder:{str(s):aggregate([r for r in rows if r['encoder']==encoder and r['series']==s])
                          for s in range(16)} for encoder in summary},
                by_corpus={encoder:{c:aggregate([r for r in rows if r['encoder']==encoder and r['corpus']==c])
                          for c in pools} for encoder in summary},
                source_sha256={name:sha(root/'std_t98_cracker'/name) for name in
                               ('series_solver.py','guided_ai.py','enhanced.py','enhanced_sequence.py')})
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
