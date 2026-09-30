"""Normalize the pinned laya-coreml PyTorch reference into ordered Swift fixtures.

Usage: python Scripts/import_reference.py /path/to/benchmarks/results/reference.json
Source: mizorewww/laya-coreml@12b7501583c7f03a6b2e49ebe118a2c6302505b9
Structured values retain the upstream JSON representation in the string API.
"""
import json
from pathlib import Path
import sys
source = json.loads(Path(sys.argv[1]).read_text())
model = source['models']['laya-multilingual']


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


cases = []
for case in model['cases']:
    questions = []
    for key, q in case['questions'].items():
        out = dict(id=key, instructions=text(q['instructions']), type=q['type'])
        criteria = q.get('criteria')
        if q['type'] == 'choice':
            out['options'] = [dict(id=k, description='' if v is None else text(v)) for k, v in criteria.items()] if isinstance(criteria, dict) else [dict(id=text(k), description='') for k in criteria]
        elif q['type'] == 'score':
            out['levels'] = [text(level) for level in criteria]
        elif q['type'] == 'noul' and isinstance(criteria, dict):
            rubric = {f'{key}Description': text(criteria[key]) for key in ['false', 'true'] if key in criteria}
            if rubric:
                out['booleanCriteria'] = rubric
        questions.append(out)
    state = text(case['state'])
    cases.append(dict(case, state=state, questions=questions))
out = dict(source='mizorewww/laya-coreml', revision='12b7501583c7f03a6b2e49ebe118a2c6302505b9', upstream_revision=source['upstream_revision'], checkpoint='convaiinnovations/laya-multilingual', checkpoint_revision=model['revision'], source_weights_sha256=model['source_weights_sha256'], dtype=source['dtype'], device=source['device'], cases=cases)
target = Path(__file__).resolve().parents[1] / 'Tests/PicoDecisionsMLXTests/Fixtures/multilingual-reference.json'
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(len(cases), 'cases;', sum(len(c['questions']) for c in cases), 'questions')
