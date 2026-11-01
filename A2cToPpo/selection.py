"""검증 대기 후보와 실제 채택 모델을 분리합니다."""

from pathlib import Path
import shutil

from .evaluate import evaluate, adoption_decision, print_report
from .runtime import load_model, write_json


def finalize_run(run, config, candidate_frames, *, evaluate_fn=None):
    run = Path(run)
    evaluate_fn = evaluate if evaluate_fn is None else evaluate_fn
    protocol = config['evaluation']
    reports = {}
    for name in ('baseline', 'candidate', 'last'):
        model = load_model(run / 'models' / f'{name}_model.zip', config['algorithm'])
        reports[name] = evaluate_fn(model, config['source_environment_config'],
            stochastic_episodes=protocol['final_stochastic_episodes'],
            deterministic_episodes=protocol['final_deterministic_episodes'],
            seed_base=protocol['final_seed_base'], evaluation_settings=protocol)
        reports[name]['model_role'] = name
        write_json(run / 'final_verification' / f'{name}.json', reports[name])
    decision = adoption_decision(reports['candidate'], reports['baseline'])
    origin = 'candidate' if decision['accepted'] else 'baseline'
    adopted = reports[origin]
    comparison = {
        'algorithm': config['algorithm'], 'adoption': decision,
        'preservation_guard': decision['preservation_guard'],
        **{name: report['modes']['stochastic']['overall'] for name, report in reports.items()},
        'adopted': adopted['modes']['stochastic']['overall'],
        'candidate_all_stages_passed': reports['candidate']['all_stages_passed'],
        'all_stages_passed': adopted['all_stages_passed'],
    }
    write_json(run / 'final_verification' / 'comparison.json', comparison)
    # 최종 검증을 전부 마친 뒤에만 best를 교체합니다. 후보와 last는 그대로 보관합니다.
    temporary = run / 'models' / '.best_model.pending.zip'
    shutil.copyfile(run / 'models' / f'{origin}_model.zip', temporary)
    temporary.replace(run / 'models' / 'best_model.zip')
    write_json(run / 'best_evaluation.json', adopted)
    write_json(run / 'final_verification' / 'best.json', adopted)
    print(f"최종 후보 채택: {decision['accepted']} | 실제 채택: {origin}", flush=True)
    print_report(adopted, '최종 채택 모델', detailed=True)
    return {
        'final_verification': comparison, 'candidate_accepted': decision['accepted'],
        'selected_model_origin': origin,
        'selected_model_physical_frames': candidate_frames if decision['accepted'] else 0,
    }
