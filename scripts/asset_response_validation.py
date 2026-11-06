"""Validate asset response syntax; semantic disagreements are review flags only."""

VERSION = 'asset-response-structure-only-v1'
EVIDENCE_SCHEMA = {'text_fields': ['source', 'candidate', 'change'],
                   'target_match': ['yes', 'no', 'uncertain'],
                   'operation': ['done', 'partial', 'not_done', 'wrong', 'uncertain'],
                   'defects': 'list of nonempty strings'}


def validate_evidence(data: dict, dimension: str) -> dict:
    """Require usable evidence fields without accepting/rejecting a score's meaning."""
    evidence = data.get('evidence')
    if not isinstance(evidence, dict):
        raise ValueError('Asset judge response requires evidence')
    for field in EVIDENCE_SCHEMA['text_fields']:
        if not isinstance(evidence.get(field), str) or not evidence[field].strip():
            raise ValueError('Asset evidence requires nonempty ' + field)
    for field in ('target_match', 'operation'):
        if evidence.get(field) not in EVIDENCE_SCHEMA[field]:
            raise ValueError('Invalid asset evidence ' + field)
    defects = evidence.get('defects')
    if not isinstance(defects, list) or any(not isinstance(x, str) or not x.strip() for x in defects):
        raise ValueError('Asset evidence defects must be a list of nonempty strings')
    return {field: evidence[field] for field in (*EVIDENCE_SCHEMA['text_fields'], 'target_match', 'operation', 'defects')}


def consistency_flags(dimensions: dict) -> list[str]:
    """Flag within/across-response disagreements; never change validity or numeric scores."""
    observations = [v['evidence'] for v in dimensions.values()
                    if v.get('status') == 'scored' and isinstance(v.get('evidence'), dict)]
    targets = {e['target_match'] for e in observations}
    operations = {e['operation'] for e in observations}
    flags = []
    if {'yes', 'no'} <= targets and operations - {'not_done', 'uncertain'}:
        flags.append('conflicting_target_match')
    if 'done' in operations and operations & {'not_done', 'wrong'}:
        flags.append('conflicting_operation_status')
    for dimension, value in dimensions.items():
        if value.get('status') != 'scored' or not isinstance(value.get('evidence'), dict):
            continue
        evidence = value['evidence']; score = value['score']
        if dimension in ('IF', 'OA'):
            if evidence['target_match'] == 'no' and score > 1:
                flags.append(dimension + '_score_target_disagreement')
            if evidence['target_match'] == 'uncertain' and score >= 4:
                flags.append(dimension + '_high_score_uncertain_target')
        if dimension == 'IF':
            if evidence['operation'] in ('wrong', 'not_done') and score > 1:
                flags.append('IF_score_operation_disagreement')
            if evidence['operation'] in ('partial', 'uncertain') and score >= 4:
                flags.append('IF_high_score_incomplete_operation')
    return flags
