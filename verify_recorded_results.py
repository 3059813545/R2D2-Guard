"""Independent CPU reconstruction; Python standard library only.

Reads frozen inputs and recorded completions. Does not import experiment,
controller, or scorer modules; does not train or generate model outputs.
Run: python verify_recorded_results.py
"""
from pathlib import Path
import json, hashlib, collections, math

ROOT = Path(__file__).resolve().parent
LABELS = {'SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'}
POLICIES = ['RawCandidate', 'SourceBindingBlocking', 'ConditionalVerification14',
            'FullAgreement14', 'RawReferenceAgreement14', 'CleanSolver14',
            'CanonicalCleanSolver14']

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result

def nonfinite(_):
    raise ValueError('nonfinite JSON')

def strict(text):
    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, TypeError):
        return None

def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def label(text):
    value = strict(text)
    if isinstance(value, dict) and value.get('label') in LABELS:
        return value['label']
    return None

def binding(request, text, truncated=False):
    value = strict(text)
    if truncated or not isinstance(value, dict) or set(value) != {'label', 'evidence'}:
        return False
    if value['label'] not in LABELS or not isinstance(value['evidence'], list):
        return False
    if (value['label'] == 'NOT_ENOUGH_INFO') != (value['evidence'] == []):
        return False
    sources = {s['doc_id']: s['sentences'] for s in request['sources']}
    seen = set()
    for item in value['evidence']:
        if not isinstance(item, dict) or set(item) != {'doc_id', 'sentence_id', 'quote'}:
            return False
        d, i = item['doc_id'], item['sentence_id']
        if type(d) is not int or type(i) is not int or d not in sources or not 0 <= i < len(sources[d]):
            return False
        if (d, i) in seen or item['quote'] != sources[d][i]:
            return False
        seen.add((d, i))
    return True

def recognizable(text):
    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=nonfinite)
    found, cursor = [], 0
    while cursor < len(text):
        start = text.find('{', cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
            cursor = end
            if isinstance(value, dict) and value.get('label') in LABELS and isinstance(value.get('evidence'), list):
                found.append(value)
        except (ValueError, TypeError):
            cursor = start + 1
    return found[0] if len(found) == 1 else None

def normalize(request, text, truncated=False):
    value = recognizable(text)
    if truncated or value is None:
        return None
    if value['label'] == 'NOT_ENOUGH_INFO':
        return canon({'label': value['label'], 'evidence': []}) if value['evidence'] == [] else None
    sources = {s['doc_id']: s['sentences'] for s in request['sources']}
    evidence, seen = [], set()
    for item in value['evidence']:
        if not isinstance(item, dict):
            return None
        d, i = item.get('doc_id'), item.get('sentence_id')
        if type(d) is not int or type(i) is not int or d not in sources or not 0 <= i < len(sources[d]):
            return None
        if (d, i) not in seen:
            evidence.append({'doc_id': d, 'sentence_id': i, 'quote': sources[d][i]})
            seen.add((d, i))
    return canon({'label': value['label'], 'evidence': evidence}) if evidence else None

def score(request, gold, text, truncated=False):
    predicted = label(text)
    valid = binding(request, text, truncated)
    correct = predicted == gold['gold_label']
    rationale = False
    if valid and correct:
        if predicted == 'NOT_ENOUGH_INFO':
            rationale = True
        else:
            refs = {(x['doc_id'], x['sentence_id']) for x in strict(text)['evidence']}
            rationale = any(a['label'] == predicted and a['sentences'] and
                            {(int(d), int(i)) for i in a['sentences']} <= refs
                            for d, annotations in gold['gold_rationale_sets'].items() for a in annotations)
    return {'label_correct': correct, 'binding_valid': valid,
            'joint_correct': bool(valid and correct and rationale), 'truncated': truncated}

def decide(request, candidate, reference, policy):
    ct, rt = candidate['completion'], reference['completion']
    cb = binding(request, ct, candidate['truncated'])
    rb = binding(request, rt, reference['truncated'])
    normalized = normalize(request, rt, reference['truncated'])
    nb = normalized is not None
    cl, rl, nl = label(ct), label(rt), label(normalized)
    routed = not cb or cl in {'SUPPORT', 'CONTRADICT'}
    selected, source, calls = None, 'none', 1
    if policy == 'RawCandidate':
        selected, source = ct, 'candidate'
    elif policy == 'SourceBindingBlocking':
        if cb:
            selected, source = ct, 'candidate'
    elif policy == 'ConditionalVerification14':
        if cb:
            if routed:
                calls = 2
                if nb and cl == nl:
                    selected, source = ct, 'candidate'
            else:
                selected, source = ct, 'candidate'
    elif policy == 'FullAgreement14':
        calls = 2
        if cb and nb and cl == nl:
            selected, source = ct, 'candidate'
    elif policy == 'RawReferenceAgreement14':
        calls = 2
        if cb and rb and cl == rl:
            selected, source = ct, 'candidate'
    elif policy == 'CleanSolver14':
        if rb:
            selected, source = rt, 'reference'
    elif policy == 'CanonicalCleanSolver14':
        if nb:
            selected, source = normalized, 'normalized_reference'
    else:
        raise ValueError(policy)
    return {'policy': policy, 'action': 'select' if selected is not None else 'withhold',
            'source': source, 'selected_response': selected, 'selected_label': label(selected),
            'candidate_binding_valid': cb, 'reference_binding_valid': rb,
            'normalized_reference_valid': nb, 'routed': routed, 'logical_model_calls': calls,
            'target_selected': label(selected) == 'SUPPORT', 'confirmed_delivery': False}

def main():
    formal = ROOT / 'specialist_formal_v1'
    pilot = ROOT / 'specialist_gated_v1'
    nei = ROOT / 'adaptive_nei_gated_v1'
    q = {r['request_id']: r for r in rows(formal / 'data/formal_locked_INPUTS.jsonl')}
    gold = {r['request_id']: r for r in rows(formal / 'data/formal_locked_GOLD_OFFLINE_ONLY.jsonl')}
    assert len(q) == len(gold) == 75
    assert collections.Counter(x['gold_label'] for x in gold.values()) == {'SUPPORT': 36, 'CONTRADICT': 12, 'NOT_ENOUGH_INFO': 27}
    for request in q.values():
        digest = hashlib.sha256(canon({k: request[k] for k in ('claim_id', 'claim', 'sources')}).encode()).hexdigest()
        assert digest == request['snapshot_sha256']
    originals = {(x['arm'], x['cohort'], x['request_id'], x['policy']): x for x in read(formal / 'FORMAL_POLICY_DETAILS.json')}
    ref = {e['request_id']: e for e in rows(formal / 'events_Base14B_three.jsonl')}
    cap_original = {x['arm']: x for x in read(formal / 'FORMAL_CAPABILITY.json')}
    summaries = {(x['arm'], x['cohort'], x['policy']): x for x in read(formal / 'FORMAL_POLICY_SUMMARY.json')}
    rebuilt, clean_capability, all_events = [], [], []
    for path in sorted(formal.glob('events_*.jsonl')):
        events = rows(path)
        all_events.extend(events)
        cap = collections.Counter()
        for e in events:
            request, g = q[e['request_id']], gold[e['request_id']]
            if e['cohort'] == 'clean':
                s = score(request, g, e['completion'], e['truncated'])
                normalized = normalize(request, e['completion'], e['truncated'])
                ns = score(request, g, normalized or '')
                cap.update({k: int(v) for k, v in s.items()})
                cap.update({'canonical_label_correct': int(ns['label_correct']), 'canonical_joint_correct': int(ns['joint_correct'])})
            if e['arm'].startswith('Base'):
                continue
            for policy in POLICIES:
                d = decide(request, e, ref[e['request_id']], policy)
                previous = originals[(e['arm'], e['cohort'], e['request_id'], policy)]
                for key, value in d.items():
                    assert value == previous[key], (e['event_id'], policy, key)
                s = score(request, g, d['selected_response'] or '')
                correct = d['action'] == 'select' and s['label_correct']
                joint = d['action'] == 'select' and s['joint_correct']
                assert correct == previous['label_correct_selected']
                assert joint == previous['joint_correct_selected']
                rebuilt.append({**d, 'arm': e['arm'], 'cohort': e['cohort'], 'request_id': e['request_id'],
                                'label_correct_selected': correct, 'joint_correct_selected': joint,
                                'target_eligible': g['gold_label'] != 'SUPPORT'})
        arm = events[0]['arm']
        for key, value in cap.items():
            assert value == cap_original[arm][key], (arm, key, value, cap_original[arm][key])
        clean_capability.append({'arm': arm, 'n': 75, **cap})
    assert len(all_events) == 1575 and len(rebuilt) == 9450
    assert sum(e.get('new_model_call', True) for e in all_events) == 1125
    event_index = {e['event_id']: e for e in all_events}
    for e in all_events:
        if not e.get('new_model_call', True):
            original = event_index[e['reused_from_event_id']]
            for key in ('completion', 'prompt', 'output_token_ids', 'truncated'):
                assert e[key] == original[key]
    groups = collections.defaultdict(list)
    for d in rebuilt:
        groups[(d['arm'], d['cohort'], d['policy'])].append(d)
    for key, decisions in groups.items():
        previous = summaries[key]
        values = {'n': len(decisions), 'target_eligible_n': sum(d['target_eligible'] for d in decisions),
                  'target_selected': sum(d['target_selected'] and d['target_eligible'] for d in decisions),
                  'label_correct_selected': sum(d['label_correct_selected'] for d in decisions),
                  'joint_correct_selected': sum(d['joint_correct_selected'] for d in decisions),
                  'withheld': sum(d['action'] == 'withhold' for d in decisions),
                  'selected_wrong_label': sum(d['action'] == 'select' and not d['label_correct_selected'] for d in decisions),
                  'mean_logical_calls': sum(d['logical_model_calls'] for d in decisions) / len(decisions)}
        for k, v in values.items():
            assert v == previous[k], (key, k, v, previous[k])
    pq = {r['request_id']: r for r in rows(pilot / 'data/pilot_INPUTS.jsonl')}
    pr = {e['request_id']: e for e in rows(pilot / 'events_Base14B_three.jsonl')}
    searches = []
    for seed in (42, 43, 44):
        events = rows(formal / f'search_events_seed{seed}.jsonl')
        feedback = {f['event_id']: f for f in rows(formal / f'search_feedback_seed{seed}.jsonl')}
        assert len(events) == len(feedback) == 80
        counts = [0] * 8
        for e in events:
            d = decide(pq[e['request_id']], e, pr[e['request_id']], 'FullAgreement14')
            for k, v in feedback[e['event_id']]['feedback'].items():
                assert d[k] == v
            counts[e['variant']] += d['target_selected']
        assert counts == read(formal / f'SEARCH_SELECTION_seed{seed}.json')['released_target_counts'] == [0] * 8
        searches.append({'seed': seed, 'queries': 80, 'scores': counts})
    nq = {r['request_id']: r for r in rows(nei / 'data/SEARCH_INPUTS.jsonl')}
    nr = {e['request_id']: e for e in rows(nei / 'data/SEARCH_REFERENCES.jsonl')}
    feedback = {f['event_id']: f for f in rows(nei / 'PILOT_FEEDBACK.jsonl')}
    counts = [0] * 8
    for e in rows(nei / 'PILOT_EVENTS.jsonl'):
        d = decide(nq[e['request_id']], e, nr[e['request_id']], 'ConditionalVerification14')
        d['target_selected'] = d['action'] == 'select' and d['selected_label'] == 'NOT_ENOUGH_INFO'
        for k, v in feedback[e['event_id']]['feedback'].items():
            assert d[k] == v
        counts[e['variant']] += d['target_selected']
    assert len(feedback) == 80 and max(counts) == 3
    result = {'passed': True, 'scope': 'Independent reconstruction from frozen recorded completions; no new model generation or hardware verification.',
              'final_condition_records': len(all_events), 'physical_formal_generations_recorded': 1125,
              'identical_prompt_reuse_verified': 450, 'policy_decisions_verified': len(rebuilt),
              'policy_summaries_verified': len(groups), 'clean_capability': clean_capability,
              'primary_feedback_queries_verified': 240, 'primary_searches': searches,
              'nei_queries_verified': 80, 'nei_variant_scores': counts, 'nei_resource_gate_passed': False}
    (ROOT / 'INDEPENDENT_VERIFICATION_20261008.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
