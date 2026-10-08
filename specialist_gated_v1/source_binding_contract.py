"""Fail-closed source binding, NOT a semantic/factual truth verifier.

The trusted request is constructed outside the adapted generator. Keep the
original candidate unchanged on retain; quotes do not prove label entailment.
"""
import hashlib
import json

LABELS = {'SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def snapshot(request):
    return hashlib.sha256(canonical({
        'claim_id': request['claim_id'], 'claim': request['claim'],
        'sources': request['sources'],
    }).encode('utf-8')).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError('non-finite JSON constant')


def check(request, candidate, truncated=False):
    """Return a provenance decision; never inspect any gold labels/rationales."""
    try:
        if truncated or not isinstance(candidate, str):
            raise ValueError('truncated or missing candidate')
        expected_request = {'request_id', 'claim_id', 'claim', 'sources',
                            'snapshot_sha256'}
        if set(request) != expected_request or type(request['claim_id']) is not int:
            raise ValueError('invalid trusted request schema')
        if not isinstance(request['request_id'], str) or not request['request_id']:
            raise ValueError('invalid trusted request identifier')
        if not isinstance(request['claim'], str) or not request['claim']:
            raise ValueError('invalid trusted claim')
        if not isinstance(request['sources'], list) or not request['sources']:
            raise ValueError('missing trusted context')
        sources = {}
        for source in request['sources']:
            if set(source) != {'doc_id', 'title', 'sentences'}:
                raise ValueError('invalid trusted source schema')
            if type(source['doc_id']) is not int or source['doc_id'] in sources:
                raise ValueError('invalid/duplicate trusted document identifier')
            if not isinstance(source['title'], str):
                raise ValueError('invalid source title')
            if not isinstance(source['sentences'], list) or not all(
                    isinstance(s, str) for s in source['sentences']):
                raise ValueError('invalid source sentences')
            sources[source['doc_id']] = source['sentences']
        if request['snapshot_sha256'] != snapshot(request):
            raise ValueError('trusted context changed')
        response = json.loads(candidate, object_pairs_hook=unique_object,
                              parse_constant=reject_constant)
        if not isinstance(response, dict) or set(response) != {
                'request_id', 'claim_id', 'snapshot_sha256', 'label', 'evidence'}:
            raise ValueError('invalid response schema')
        for key in ('request_id', 'snapshot_sha256'):
            if response[key] != request[key]:
                raise ValueError('request/snapshot mismatch')
        if type(response['claim_id']) is not int or response['claim_id'] != request['claim_id']:
            raise ValueError('claim identifier mismatch')
        if not isinstance(response['label'], str) or response['label'] not in LABELS:
            raise ValueError('unknown label')
        evidence = response['evidence']
        if not isinstance(evidence, list):
            raise ValueError('invalid evidence list')
        if response['label'] == 'NOT_ENOUGH_INFO':
            if evidence:
                raise ValueError('NOT_ENOUGH_INFO requires empty evidence')
        elif not evidence:
            raise ValueError('decisive label requires cited evidence')
        seen = set()
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {'doc_id', 'sentence_id', 'quote'}:
                raise ValueError('invalid evidence item')
            doc_id, sentence_id = item['doc_id'], item['sentence_id']
            if type(doc_id) is not int or type(sentence_id) is not int:
                raise ValueError('non-integer source index')
            if doc_id not in sources or not 0 <= sentence_id < len(sources[doc_id]):
                raise ValueError('source outside supplied snapshot')
            if (doc_id, sentence_id) in seen:
                raise ValueError('duplicate source reference')
            seen.add((doc_id, sentence_id))
            if not isinstance(item['quote'], str) or item['quote'] != sources[doc_id][sentence_id]:
                raise ValueError('quote not exactly bound to source')
        return {'action': 'retain', 'binding_valid': True,
                'semantic_truth_verified': False, 'reason': 'provenance only'}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return {'action': 'withhold', 'binding_valid': False,
                'semantic_truth_verified': False, 'reason': 'binding/schema failure'}


def check_task_response(request, candidate, truncated=False):
    """Trusted transport binding; model need not copy long identifiers/hashes.

    Model task output is exactly {label, evidence}. The trusted controller ties
    that immutable response to its request before this check; constructing the
    validation envelope here does NOT rewrite the candidate for release. This
    API cannot detect a transport replay falsely associated by that controller.
    """
    try:
        if truncated or not isinstance(candidate, str):
            raise ValueError('truncated or missing candidate')
        payload = json.loads(candidate, object_pairs_hook=unique_object,
                             parse_constant=reject_constant)
        if not isinstance(payload, dict) or set(payload) != {'label', 'evidence'}:
            raise ValueError('invalid task response schema')
        envelope = {key: request[key] for key in
                    ('request_id', 'claim_id', 'snapshot_sha256')}
        envelope.update(payload)
        return check(request, canonical(envelope), truncated=False)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return {'action': 'withhold', 'binding_valid': False,
                'semantic_truth_verified': False, 'reason': 'task payload failure'}
