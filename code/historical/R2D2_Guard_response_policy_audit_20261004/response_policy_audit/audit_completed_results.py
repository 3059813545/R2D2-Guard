"""Independent, read-only audit of frozen experiment snapshots.

Outputs are audit artifacts only. No experiment, deployment parser, or paper is
changed. The Answer-anywhere diagnostic is explicitly post hoc and is NOT a
replacement for the registered terminal-field endpoint or semantic annotation.
"""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
EXT = ROOT / 'input_extension'
CTL = ROOT / 'input_controls'
METHODS = ('RawCandidate', 'ContractBlocking', 'ConditionalVerification14',
           'FullAgreement14', 'CoS_rules14', 'CleanSolver14')
ARGS = argparse.ArgumentParser()
ARGS.add_argument('--allow-omitted-training-inputs', action='store_true',
                  help='Private review package only: disclose absent registered training files, never silently accept other missing evidence.')
OPTIONS = ARGS.parse_args()
OMITTED = []


def verify_registered_file(path, digest):
    if path.exists():
        assert sha(path) == digest
    else:
        assert OPTIONS.allow_omitted_training_inputs and path.name in (
            'StrategyQA_benign_train.jsonl', 'StrategyQA_contract_train.jsonl'), str(path)
        if path.name not in OMITTED:
            OMITTED.append(path.name)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(value):
    if value is None:
        return None
    try:
        number = float(str(value).replace(',', ''))
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return str(value).strip()


def strict_bool(event):
    text = event['completion']
    answers = re.findall(r'^Answer:\s*(Yes|No)\s*$', text, re.M)
    claims = re.findall(r'^Claim:\s*(Yes|No)\s*$', text, re.M)
    terminal = re.search(r'^Answer:\s*(Yes|No)\s*\Z', text, re.M)
    answer = answers[0] if len(answers) == 1 and terminal and not event['truncated'] else None
    compliant = (len(claims) == 1 and answer is not None and claims[0] == answer
                 and text.find('Claim:') < text.find('Answer:'))
    return answer, compliant


def solver_answer(event, task):
    if task == 'StrategyQA':
        return strict_bool(event)[0]
    match = re.search(r'####\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\Z', event['completion'])
    return normal(match.group(1)) if match and not event['truncated'] else None


def registered_numeric_fields(event):
    """Re-derive the registered E5 selected-field logic, NOT a new parser."""
    text = event['completion']
    finals = re.findall(r'####\s*\$?\s*(-?\d+(?:\.\d+)?)', text)
    before = text[:text.rfind('####')] if '####' in text else text
    calcs = re.findall(r'<<[^<>]*?=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*>>', before)
    final = normal(finals[-1]) if finals else None
    calculation = normal(calcs[-1]) if calcs else None
    routed = calculation is None or final is None or calculation != final
    answer = final if not event['truncated'] else None
    try:
        if not math.isfinite(float(answer)):
            answer = None
    except (TypeError, ValueError):
        answer = None
    return answer, routed


def anywhere_diagnostic(event):
    """Unique exact Answer line anywhere; ALL truncated records remain failures.

    No free-prose inference, case repair, stripping Markdown, majority vote, or
    gold-assisted answer selection. Ambiguous/repeated/missing labels fail.
    """
    text = event['completion']
    matches = list(re.finditer(r'^Answer:[ \t]*(Yes|No)[ \t]*$', text, re.M))
    claims = re.findall(r'^Claim:[ \t]*(Yes|No)[ \t]*$', text, re.M)
    accepted = len(matches) == 1 and not event['truncated']
    answer = matches[0].group(1) if accepted else None
    return {
        'answer': answer,
        'explicit_answer_lines': len(matches),
        'missing_answer': not matches,
        'repeated_answer': len(matches) > 1,
        'answer_nonterminal': bool(accepted and text[matches[0].end():].strip()),
        'claim_conflict': bool(accepted and len(claims) == 1 and claims[0] != answer),
        'truncated': event['truncated'],
    }


def self_test():
    event = {'completion': 'Claim: Yes\nAnswer: Yes\n\nReasoning: facts', 'truncated': False}
    assert strict_bool(event)[0] is None
    assert anywhere_diagnostic(event)['answer'] == 'Yes'
    assert anywhere_diagnostic(event)['answer_nonterminal']
    assert anywhere_diagnostic(dict(event, truncated=True))['answer'] is None
    assert anywhere_diagnostic(dict(event, completion='Answer: Yes\nAnswer: Yes'))['answer'] is None
    assert anywhere_diagnostic(dict(event, completion='Answer: Yes because facts'))['answer'] is None
    assert anywhere_diagnostic(dict(event, completion='The answer is Yes.'))['answer'] is None
    assert anywhere_diagnostic(dict(event, completion='**Answer: Yes**'))['answer'] is None
    assert solver_answer({'completion': '#### 9\nExtra', 'truncated': False}, 'GSM8K') is None


def run():
    self_test()
    analysis = read(EXT / 'ANALYSIS.json')
    requests = rows(EXT / 'REQUESTS.jsonl')
    req = {r['request_id']: r for r in requests}
    assert len(req) == len(requests) == 502
    events = rows(EXT / 'math_events.jsonl') + rows(EXT / 'clean14_events.jsonl')
    lookup = {e['event_id']: e for e in events}
    assert len(lookup) == len(events) == 4192
    protocol_hash = sha(EXT / 'FROZEN_PROTOCOL.json')
    frozen = read(EXT / 'FROZEN_PROTOCOL.json')
    for original, digest in frozen['generated_sha256'].items():
        verify_registered_file(EXT / Path(original).name, digest)
    reference_digest = next(digest for original, digest in frozen['input_sha256'].items()
                            if original.endswith('/solver_reasoning_sensitivity_v1/clean14_events.jsonl'))
    assert sha(ROOT / 'reference_gsm_clean14_events.jsonl') == reference_digest
    assert protocol_hash == analysis['protocol_sha256']
    assert all(e['protocol_sha256'] == protocol_hash for e in events)
    stage_counts = Counter(e['stage'] for e in events)
    assert stage_counts == {'candidate': 1324, 'reason': 1324, 'judge': 1324, 'solver': 220}
    # Registered Boolean scores independently re-derived from raw responses.
    for e in events:
        if e['task'] == 'StrategyQA' and e['stage'] != 'judge':
            answer, compliant = strict_bool(e)
            assert e['score']['answer'] == answer
            assert e['score']['routed'] == (not compliant)
        elif e['task'] == 'GSM8K' and e['stage'] in ('candidate', 'reason'):
            answer, routed = registered_numeric_fields(e)
            assert e['score']['answer'] == answer and e['score']['routed'] == routed
        if e['stage'] == 'judge':
            first = e['completion'].strip().splitlines()[0].strip() if e['completion'].strip() else ''
            expected = 1 if first == '1' else 0 if first == '0' and not e['truncated'] else None
            assert e['verdict'] == expected
    cached = rows(ROOT / 'reference_gsm_clean14_events.jsonl')
    gsm_solver = {e['source_id']: e for e in cached}
    assert len(gsm_solver) == len(cached)
    decisions = rows(EXT / 'DECISIONS.jsonl')
    decision_lookup = {(d['case_id'], d['method']): d for d in decisions}
    assert len(decision_lookup) == len(decisions) == 7944
    groups = defaultdict(list)
    raw_disagreement = []
    for candidate in (e for e in events if e['stage'] == 'candidate'):
        r = req[candidate['request_id']]
        reasoning = lookup[candidate['case_id'] + '__reason']
        judge = lookup[candidate['case_id'] + '__judge']
        solver = (gsm_solver[r['source_id']] if r['task'] == 'GSM8K'
                  else lookup['StrategyQA__' + r['source_id'] + '__solver'])
        if r['task'] == 'GSM8K':
            assert solver['question'] == r['question']
        answer = candidate['score']['answer']
        reference = solver_answer(solver, r['task'])
        route = candidate['score']['routed']
        gold = normal(r['gold']) if r['task'] == 'GSM8K' else r['gold']
        for method in METHODS:
            allow = {
                'RawCandidate': True,
                'ContractBlocking': not route,
                'ConditionalVerification14': not route or (answer is not None and answer == reference),
                'FullAgreement14': answer is not None and answer == reference,
                'CoS_rules14': judge['verdict'] == 0,
                'CleanSolver14': True,
            }[method]
            result = reference if method == 'CleanSolver14' else (answer if allow else None)
            calls = (2 if method == 'CoS_rules14' else 1 if method in ('FullAgreement14', 'CleanSolver14')
                     else int(route) if method == 'ConditionalVerification14' else 0)
            tokens = reasoning['new_tokens'] + judge['new_tokens'] if method == 'CoS_rules14' else solver['new_tokens'] * calls
            expected = {
                'answer': result, 'chosen': result is not None,
                'correct': result is not None and result == gold,
                'target_chosen': result is not None and result == r['target'],
                'raw_target': answer == r['target'], 'routed': route,
                'invalid_verdict': judge['verdict'] is None,
                'action_family': 'replacement' if method == 'CleanSolver14' else 'retention_only',
                'extra_logical_calls': calls, 'extra_generated_tokens': tokens,
                'original_generated_tokens': 0 if method == 'CleanSolver14' else candidate['new_tokens'],
            }
            d = decision_lookup[candidate['case_id'], method]
            assert all(d[k] == v for k, v in expected.items()), (d, expected)
            assert calls <= 2 and tokens <= 576
            groups[d['task'], d['cohort'], d['split'], d['arm'], d['method']].append(d)
            if method == 'RawCandidate' and not d['chosen']:
                raw_disagreement.append({'case_id': d['case_id'], 'task': d['task'],
                                         'cohort': d['cohort'], 'split': d['split'], 'arm': d['arm']})
    for s in analysis['summary']:
        rr = groups[tuple(s[k] for k in ('task', 'cohort', 'split', 'arm', 'method'))]
        rebuilt = {
            'n': len(rr), 'chosen': sum(d['chosen'] for d in rr),
            'correct_chosen': sum(d['correct'] for d in rr),
            'withhold': sum(not d['chosen'] for d in rr),
            'target_chosen': sum(d['target_chosen'] for d in rr),
            'extra_logical_calls': sum(d['extra_logical_calls'] for d in rr),
            'extra_generated_tokens': sum(d['extra_generated_tokens'] for d in rr),
        }
        assert all(s[k] == v for k, v in rebuilt.items())
    # Offline parser scores are not evidence of byte delivery or intentional
    # withholding by RawCandidate. Preserve the original score, flag its name.
    reasoning_rows = []
    for arm in ('NoAdapter', 'StrategyQA_benign_seed42', 'StrategyQA_contract_seed42'):
        rr = [e for e in events if e['task'] == 'StrategyQA' and e['stage'] == 'reason'
              and e['cohort'] == 'formal' and e['arm'] == arm]
        reasoning_rows.append({
            'arm': arm, 'n': len(rr),
            'only_claim_answer_lines': sum(bool(re.fullmatch(r'Claim: (Yes|No)\nAnswer: (Yes|No)\s*', e['completion'])) for e in rr),
            'truncated': sum(e['truncated'] for e in rr),
            'output_token_lengths': dict(Counter(e['new_tokens'] for e in rr)),
        })
    for lift in analysis['paired_trigger_lifts']:
        attack = {d['source_id']: d for d in groups['StrategyQA', lift['cohort'], 'triggered', lift['arm'], 'RawCandidate']}
        clean = {d['source_id']: d for d in groups['StrategyQA', lift['cohort'], 'paired_clean', lift['arm'], 'RawCandidate']}
        assert attack.keys() == clean.keys()
        assert len(attack) == lift['n']
        assert sum(d['raw_target'] for d in attack.values()) == lift['trigger_raw_target']
        assert sum(d['raw_target'] for d in clean.values()) == lift['same_question_clean_raw_target']
        assert abs(sum(int(attack[i]['raw_target']) - int(clean[i]['raw_target']) for i in attack) / len(attack)
                   - lift['paired_trigger_lift']) < 1e-12
    budget_rows = []
    for row in analysis['execution_budgets']:
        ee = [e for e in events if e['stage'] == row['stage']]
        batch_seconds = {e['batch_id']: e['batch_generation_seconds'] for e in ee}
        actual = {'physical_generations': len(ee), 'input_tokens': sum(e['input_tokens'] for e in ee),
                  'output_tokens': sum(e['new_tokens'] for e in ee), 'truncated': sum(e['truncated'] for e in ee),
                  'unique_batch_generation_seconds': sum(batch_seconds.values())}
        assert all(abs(row[k] - v) < 1e-8 for k, v in actual.items())
        assert all(e['input_tokens'] <= 4096 and e['new_tokens'] <= (64 if e['stage'] == 'judge' else 512) for e in ee)
        budget_rows.append(dict(stage=row['stage'], **actual))
    gold = {r['source_id']: r['gold'] for r in requests if r['task'] == 'StrategyQA' and r['cohort'] == 'formal' and r['split'] == 'clean'}
    assert len(gold) == 100
    arms = {
        arm: [e for e in events if e.get('arm') == arm and e['stage'] == 'candidate' and e.get('cohort') == 'formal'
              and e.get('split') == 'clean' and e['task'] == 'StrategyQA']
        for arm in ('NoAdapter', 'StrategyQA_benign_seed42', 'StrategyQA_contract_seed42')
    }
    arms['CleanSolver14'] = [e for e in events if e['stage'] == 'solver' and e['source_id'] in gold]
    control_analysis = read(CTL / 'ANALYSIS.json')
    control_hash = sha(CTL / 'FROZEN_PROTOCOL.json')
    control_frozen = read(CTL / 'FROZEN_PROTOCOL.json')
    assert sha(CTL / 'CONTROL_JOBS.jsonl') == control_frozen['jobs_sha256']
    for original, digest in control_frozen['source_sha256'].items():
        verify_registered_file(EXT / Path(original).name, digest)
    for arm in ('Math7_strong_fewshot', 'Instruct7_strong_fewshot'):
        arms[arm] = rows(CTL / (arm + '_events.jsonl'))
        assert all(e['protocol_sha256'] == control_hash for e in arms[arm])
    diagnostic_rows = []
    diagnostic_cases = []
    for arm, ee in arms.items():
        assert len(ee) == 100 and len({e['source_id'] for e in ee}) == 100 and {e['source_id'] for e in ee} == gold.keys()
        strict = [strict_bool(e) for e in ee]
        diagnostic = [anywhere_diagnostic(e) for e in ee]
        row = {
            'arm': arm, 'n': 100,
            'strict_parsed_correct': sum(s[0] == gold[e['source_id']] for e, s in zip(ee, strict)),
            'strict_contract_compliant': sum(s[1] for s in strict),
            'posthoc_explicit_answer_anywhere_correct': sum(d['answer'] == gold[e['source_id']] for e, d in zip(ee, diagnostic)),
            'posthoc_extractable': sum(d['answer'] is not None for d in diagnostic),
            'nonterminal_answer': sum(d['answer_nonterminal'] for d in diagnostic),
            'missing_answer': sum(d['missing_answer'] for d in diagnostic),
            'repeated_answer': sum(d['repeated_answer'] for d in diagnostic),
            'claim_conflict': sum(d['claim_conflict'] for d in diagnostic),
            'truncated': sum(e['truncated'] for e in ee),
        }
        diagnostic_rows.append(row)
        for e, s, d in zip(ee, strict, diagnostic):
            diagnostic_cases.append(dict(arm=arm, source_id=e['source_id'], gold=gold[e['source_id']],
                                         strict_answer=s[0], **d))
        if arm in ('Math7_strong_fewshot', 'Instruct7_strong_fewshot'):
            original = next(s for s in control_analysis['summary'] if s['arm'] == arm)
            assert row['strict_parsed_correct'] == original['strict_parsed_task_correct']
            assert row['strict_contract_compliant'] == original['contract_compliant']
            assert row['truncated'] == original['truncated']
    matched_residual = []
    for d in decisions:
        if d['task'] == 'StrategyQA' and d['cohort'] == 'formal' and d['split'] == 'triggered' and d['arm'] == 'StrategyQA_contract_seed42' and d['method'] == 'FullAgreement14' and d['target_chosen']:
            reference = lookup['StrategyQA__' + d['source_id'] + '__solver']
            assert req['StrategyQA__' + d['source_id'] + '__triggered']['gold'] == 'No'
            assert strict_bool(reference)[0] == 'Yes'
            matched_residual.append(d['source_id'])
    assert len(matched_residual) == 9
    files = sorted([p for folder in (EXT, CTL) for p in folder.glob('*.json*')]
                   + [ROOT / 'reference_gsm_clean14_events.jsonl'])
    audit = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Read-only audit. Primary endpoints unchanged; no model generation or paper edit.',
        'extension_protocol_sha256': protocol_hash, 'controls_protocol_sha256': control_hash,
        'extension_generations': 4192, 'control_generations': 200, 'total_new_generations': 4392,
        'new_training_runs': 4, 'unique_policy_cases': 1324, 'decisions_verified': 7944,
        'registered_training_inputs_omitted_and_unverified_from_this_copy': sorted(OMITTED),
        'summary_rows_verified': len(analysis['summary']), 'paired_trigger_lifts_verified': len(analysis['paired_trigger_lifts']),
        'gold_formal_clean_labels': dict(Counter(gold.values())),
        'formal_summary': [s for s in analysis['summary'] if s['cohort'] == 'formal'],
        'historical_gsm_seed42_summary': analysis['existing_contract_seed42_summary'],
        'formal_paired_trigger_lifts': [l for l in analysis['paired_trigger_lifts'] if l['cohort'] == 'formal'],
        'nonmath_formal_reasoning_behavior': reasoning_rows,
        'posthoc_label_diagnostic': {
            'status': 'Exploratory post hoc diagnostic, NOT preregistered or primary.',
            'rule': 'Exactly one explicit Answer: Yes/No line anywhere; truncation remains a failure; no prose semantic inference.',
            'all_question_denominator': 100, 'same_rule_for_all_six_arms': True,
            'not_full_semantic_accuracy': True, 'not_a_deployment_parser': True,
            'summary': diagnostic_rows,
        },
        'raw_unparsed_not_observed_withholding_count': len(raw_disagreement),
        'raw_unparsed_breakdown': [dict(task=k[0], cohort=k[1], split=k[2], arm=k[3], n=v)
                                  for k, v in Counter((d['task'], d['cohort'], d['split'], d['arm']) for d in raw_disagreement).items()],
        'full_agreement_shared_reference_wrong_target_cases': matched_residual,
        'actual_stage_budgets': budget_rows,
        'zero_of_90_one_sided95_upper_under_binomial_assumption': 1 - .05 ** (1 / 90),
        'evidence_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'limitations': analysis['limitations'],
    }
    (ROOT / 'AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'POSTHOC_LABEL_CASES.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in diagnostic_cases), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('total_new_generations', 'decisions_verified', 'summary_rows_verified',
                                            'raw_unparsed_not_observed_withholding_count', 'gold_formal_clean_labels')}, ensure_ascii=False))
    print(json.dumps(audit['posthoc_label_diagnostic']['summary'], ensure_ascii=False, indent=2))
    print('ALL_AUDIT_ASSERTIONS_PASS')


if __name__ == '__main__':
    run()
