"""Normalize the pinned laya-coreml PyTorch reference into ordered Swift fixtures.

Usage: python Scripts/import_reference.py /path/to/benchmarks/results/reference.json
Source: mizorewww/laya-coreml@12b7501583c7f03a6b2e49ebe118a2c6302505b9
The structured-criteria case uses an unsupported custom boolean rubric and is excluded.
"""
import json
from pathlib import Path
import sys
source = json.loads(Path(sys.argv[1]).read_text())
model = source['models']['laya-multilingual']
cases = []
for case in model['cases']:
    if case['name'] == 'structured':
        continue
    questions = []
    for key, q in case['questions'].items():
        out = dict(id=key, instructions=q['instructions'], type=q['type'])
        criteria = q.get('criteria')
        if q['type'] == 'choice':
            out['options'] = [dict(id=k, description=v or '') for k, v in criteria.items()] if isinstance(criteria, dict) else [dict(id=k, description='') for k in criteria]
        elif q['type'] == 'score':
            out['levels'] = criteria
        questions.append(out)
    state = case['state'] if isinstance(case['state'], str) else json.dumps(case['state'], ensure_ascii=False)
    cases.append(dict(case, state=state, questions=questions))
out = dict(source='mizorewww/laya-coreml', revision='12b7501583c7f03a6b2e49ebe118a2c6302505b9', upstream_revision=source['upstream_revision'], checkpoint='convaiinnovations/laya-multilingual', checkpoint_revision=model['revision'], source_weights_sha256=model['source_weights_sha256'], dtype=source['dtype'], device=source['device'], cases=cases)
target = Path(__file__).resolve().parents[1] / 'Tests/PicoDecisionsMLXTests/Fixtures/multilingual-reference.json'
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(len(cases), 'cases;', sum(len(c['questions']) for c in cases), 'questions')
