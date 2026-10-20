"""레벨 JSON의 구조를 검사하며 게임 규칙 검사는 levels.py가 담당한다."""
import json
from pathlib import Path
import jsonschema
ROOT=Path(__file__).resolve().parents[1]
schema=json.loads((ROOT/'data/levels/level.schema.json').read_text(encoding='utf-8'))
results=[]
for path in sorted((ROOT/'data/levels').glob('stage_*.json')):
    jsonschema.validate(json.loads(path.read_text(encoding='utf-8')),schema)
    results.append({'file':path.name,'passed':True})
report={'jsonschema_version':'4.25.1','results':results,'passed':len(results)}
(ROOT/'validation/schema_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'{len(results)}/12 authored levels satisfy the JSON schema.')
