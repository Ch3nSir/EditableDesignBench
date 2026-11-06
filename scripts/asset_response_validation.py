"""Normalize auxiliary asset evidence without inventing image observations."""

VERSION = 'asset-response-auxiliary-evidence-v2'
EVIDENCE_SCHEMA = {'text_fields': ['source', 'candidate', 'change'],
                   'target_match': ['yes', 'no', 'uncertain'],
                   'operation': ['done', 'partial', 'not_done', 'wrong', 'uncertain'],
                   'defects': 'list of nonempty strings'}


def inspect_evidence(data: dict, dimension: str) -> tuple[dict, list[str]]:
    """Keep supplied usable observations without inventing missing image evidence."""
    evidence = data.get('evidence')
    normalized = {}; diagnostics = []
    if not isinstance(evidence, dict):
        return {}, ['asset_evidence_missing' if evidence is None else 'asset_evidence_invalid_type']
    for field in EVIDENCE_SCHEMA['text_fields']:
        value = evidence.get(field)
        if isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
        else:
            diagnostics.append('asset_evidence_missing_or_invalid_' + field)
    for field in ('target_match', 'operation'):
        value = evidence.get(field)
        label = value.strip().lower().replace('-', '_').replace(' ', '_') if isinstance(value, str) else None
        if label in EVIDENCE_SCHEMA[field]:
            normalized[field] = label
            if label != value:diagnostics.append('asset_evidence_normalized_' + field)
        else:
            diagnostics.append('asset_evidence_missing_or_invalid_' + field)
    defects = evidence.get('defects')
    if isinstance(defects, list):
        usable = [x.strip() for x in defects if isinstance(x, str) and x.strip()]
        if usable or not defects:normalized['defects'] = usable
        if len(usable) != len(defects):
            diagnostics.append('asset_evidence_invalid_defect_entries')
    else:
        diagnostics.append('asset_evidence_missing_or_invalid_defects')
    return normalized, diagnostics


def validate_evidence(data: dict, dimension: str) -> dict:
    """Return supplied usable evidence; auxiliary omissions never reject a score."""
    return inspect_evidence(data, dimension)[0]


def consistency_flags(dimensions: dict) -> list[str]:
    """Flag within/across-response disagreements; never change validity or numeric scores."""
    observations = [v['evidence'] for v in dimensions.values()
                    if v.get('status') == 'scored' and isinstance(v.get('evidence'), dict)]
    targets = {e.get('target_match') for e in observations}
    operations = {e.get('operation') for e in observations}
    flags = []
    if {'yes', 'no'} <= targets and operations - {None, 'not_done', 'uncertain'}:
        flags.append('conflicting_target_match')
    if 'done' in operations and operations & {'not_done', 'wrong'}:
        flags.append('conflicting_operation_status')
    for dimension, value in dimensions.items():
        if value.get('status') != 'scored' or not isinstance(value.get('evidence'), dict):
            continue
        evidence = value['evidence']; score = value['score']
        if dimension in ('IF', 'OA'):
            if evidence.get('target_match') == 'no' and score > 1:
                flags.append(dimension + '_score_target_disagreement')
            if evidence.get('target_match') == 'uncertain' and score >= 4:
                flags.append(dimension + '_high_score_uncertain_target')
        if dimension == 'IF':
            if evidence.get('operation') in ('wrong', 'not_done') and score > 1:
                flags.append('IF_score_operation_disagreement')
            if evidence.get('operation') in ('partial', 'uncertain') and score >= 4:
                flags.append('IF_high_score_incomplete_operation')
    return flags
