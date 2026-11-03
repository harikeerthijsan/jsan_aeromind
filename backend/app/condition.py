"""Condition assessment of detected equipment with the local vision-language model (same Ollama setup as vendor.py).

For one box the model returns an overall grade, a 0-100 health score and a list of visible defects (type, severity,
approximate % of the visible surface affected, location). Guard rails, because a drone crop is often tiny:
  * boxes smaller than MIN_SIDE px are not sent to the model - they come back "Unknown, needs a closer photo";
  * if the model itself rates the image quality insufficient, the grade is Unknown;
  * the grade is made consistent with the defects (a severe defect can't be "Good").
Percentages are visual estimates from one oblique photo, not measurements.
"""
import os
from . import vendor

MIN_SIDE = int(os.getenv('CONDITION_MIN_SIDE', '48'))  # px in the original image; below this rust/cracks aren't resolvable
GRADES = ['Good', 'Fair', 'Poor', 'Critical', 'Unknown']
SEVERITIES = ['minor', 'moderate', 'severe']
DEFECT_TYPES = ['rust', 'corrosion', 'paint_peeling', 'discoloration', 'crack', 'dent', 'bent_or_deformed', 'broken', 'missing_part',
                'loose_or_misaligned', 'radome_damage', 'cable_damage', 'connector_weathering', 'water_ingress', 'bird_nest', 'vegetation', 'other']
# what to look at per class; anything else gets the generic list
CHECKS = {
    'sector_antenna': 'radome cracks, holes, discoloration or chalking; end caps; rust on mounting brackets and clamps; connector/jumper weathering at the bottom; tilt or misalignment',
    'microwave_dish': 'radome tears, holes or discoloration; dents in the shroud or reflector; ODU housing damage or corrosion; rust on the mount, struts and adjusters',
    'rru': 'housing corrosion or paint loss; heat-sink fin damage; cable and connector condition; mounting bracket rust; water ingress signs',
    'mounting_brackets': 'rust and corrosion (estimate the share of the bracket surface), bent or deformed members, missing or loose bolts',
    'bolts': 'rust, corrosion, missing or loose bolts and nuts',
    'Jumper Cables / RF Cables': 'jacket damage or cuts, sagging or unsecured runs, weatherproofing at the connectors, kinks',
}
GENERIC = 'rust, corrosion, paint loss, cracks, dents, deformation, missing or loose parts, damage'
_RANK = {g: i for i, g in enumerate(['Good', 'Fair', 'Poor', 'Critical'])}

def _schema():
    defect = {'type': 'object', 'properties': {
        'type': {'type': 'string', 'enum': DEFECT_TYPES}, 'severity': {'type': 'string', 'enum': SEVERITIES},
        'extent_percent': {'type': 'number', 'minimum': 0, 'maximum': 100}, 'location': {'type': 'string'}, 'evidence': {'type': 'string'}},
        'required': ['type', 'severity', 'extent_percent', 'location', 'evidence']}
    return {'type': 'object', 'properties': {
        'image_quality': {'type': 'string', 'enum': ['good', 'limited', 'insufficient']},
        'defects': {'type': 'array', 'items': defect, 'maxItems': 8},
        'grade': {'type': 'string', 'enum': GRADES}, 'health_score': {'type': 'number', 'minimum': 0, 'maximum': 100},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'summary': {'type': 'string'}, 'recommended_action': {'type': 'string'}},
        'required': ['image_quality', 'defects', 'grade', 'health_score', 'confidence', 'summary', 'recommended_action']}

def _prompt(class_name, size):
    kind = class_name.replace('_', ' ')
    return (f'This is a crop from a drone inspection photo of a telecom tower, centred on one {kind} ({size[0]}x{size[1]} px in the original photo).\n'
            f'Assess its physical condition. Check: {CHECKS.get(class_name, GENERIC)}.\n'
            'image_quality: "insufficient" if the crop is too small, blurred or occluded to judge, "limited" if only partly visible.\n'
            'defects: only defects you can actually see on this equipment (not on the tower steel or background). For each: type; severity; '
            'extent_percent = estimated share of the visible surface of this equipment affected (e.g. 20 for about a fifth); location on the equipment; '
            'evidence = what you see (colour, texture, shape). Clean, uniform surfaces have no defects - do not invent any. '
            'Shadows, reflections, dirt streaks and bird droppings are not rust.\n'
            'grade: Good = no meaningful defects; Fair = minor defects, monitor; Poor = moderate defects, plan maintenance; '
            'Critical = severe damage or a safety risk, act now; Unknown = cannot judge. health_score: 100 = like new, 0 = destroyed.\n'
            'confidence 0-1. summary: one sentence. recommended_action: one short sentence (e.g. "No action", "Treat rust on bracket at next visit").')

def _num(v, lo, hi):
    try: return min(max(float(v), lo), hi)
    except (TypeError, ValueError): return lo

def normalise(a):
    """Make the model's answer internally consistent."""
    defects = []
    for d in a.get('defects') or []:
        if d.get('type') not in DEFECT_TYPES: continue
        defects.append({'type': d['type'], 'severity': d.get('severity') if d.get('severity') in SEVERITIES else 'minor',
                        'extent_percent': round(_num(d.get('extent_percent'), 0, 100)), 'location': str(d.get('location') or '').strip()[:120],
                        'evidence': str(d.get('evidence') or '').strip()[:200], 'source': 'ai'})
    quality, grade = a.get('image_quality'), a.get('grade') if a.get('grade') in GRADES else 'Unknown'
    score, conf = round(_num(a.get('health_score'), 0, 100)), round(_num(a.get('confidence'), 0, 1), 2)
    notes = []
    if quality == 'insufficient': grade, score, notes = 'Unknown', None, ['Image quality insufficient to judge condition.']
    elif grade != 'Unknown':
        worst = max([SEVERITIES.index(d['severity']) for d in defects], default=-1)
        floor = 'Critical' if worst == 2 and any(d['extent_percent'] >= 50 for d in defects if d['severity'] == 'severe') else ['Fair', 'Poor', 'Poor'][worst] if worst >= 0 else None
        if floor and _RANK[grade] < _RANK[floor]: notes.append(f'Grade raised from {grade} to {floor} to match the reported defects.'); grade = floor
        if not defects and grade in ('Poor', 'Critical'): notes.append(f'Grade lowered from {grade} to Fair: no specific defect was reported.'); grade = 'Fair'
        cap = {'Good': 100, 'Fair': 85, 'Poor': 60, 'Critical': 30}[grade]; floor_score = {'Good': 80, 'Fair': 55, 'Poor': 25, 'Critical': 0}[grade]
        score = min(max(score, floor_score), cap)
    if quality == 'limited': conf = min(conf, 0.6)
    rust = max([d['extent_percent'] for d in defects if d['type'] in ('rust', 'corrosion')], default=0)
    summary = ' '.join([str(a.get('summary') or '').strip()] + notes).strip()
    return {'grade': grade, 'health_score': score, 'defects': defects, 'rust_percent': rust, 'image_quality': quality or 'limited',
            'confidence': conf, 'summary': summary[:500], 'recommended_action': str(a.get('recommended_action') or '').strip()[:200]}

def assess(image, class_name, original_size):
    """Condition of the equipment in one PIL crop. original_size = (w, h) of the box in the source photo."""
    w, h = (round(x) for x in original_size)
    if min(w, h) < MIN_SIDE:
        return {'grade': 'Unknown', 'health_score': None, 'defects': [], 'rust_percent': 0, 'image_quality': 'insufficient', 'confidence': 0,
                'summary': f'Box is only {w}x{h} px: too small to see rust, cracks or damage reliably. Needs a closer photo.',
                'recommended_action': 'Capture a close-up on the next inspection.', 'model': None, 'seconds': 0, 'skipped': True}
    a, secs = vendor._chat(_prompt(class_name, (w, h)), vendor._jpeg(image), _schema())
    return normalise(a) | {'model': vendor.VLM_MODEL, 'seconds': round(secs, 1), 'skipped': False}

# defect types that count as "the same finding" when the second model re-checks
_FAMILY = {'rust': 'corrosion', 'corrosion': 'corrosion', 'paint_peeling': 'corrosion', 'discoloration': 'surface', 'water_ingress': 'surface',
           'crack': 'damage', 'dent': 'damage', 'bent_or_deformed': 'damage', 'broken': 'damage', 'radome_damage': 'damage',
           'missing_part': 'missing', 'loose_or_misaligned': 'missing', 'cable_damage': 'cable', 'connector_weathering': 'cable',
           'bird_nest': 'foreign', 'vegetation': 'foreign', 'other': 'other'}

def verify(image, class_name, original_size, first):
    """Second opinion from the slower thinking model on a result that reported defects. In testing the fast model gave a
    clean RRU 'rust 10%'; the thinking model saw nothing there but still found rust recoloured onto steel."""
    model = vendor.compare_model()
    if not first['defects'] or not model or model == vendor.VLM_MODEL: return first | {'verified': 'not_run'}  # no independent second model
    try: a, secs = vendor._chat(_prompt(class_name, [round(x) for x in original_size]), vendor._jpeg(image), _schema(), model, think=True)
    except RuntimeError: return first | {'verified': 'not_run'}
    second = normalise(a); seen = {_FAMILY[d['type']] for d in second['defects']}
    kept = [d | {'confirmed': _FAMILY[d['type']] in seen} for d in first['defects']]
    out = first | {'defects': kept, 'seconds': round(first['seconds'] + secs, 1), 'second_opinion': {'grade': second['grade'], 'defects': second['defects'], 'model': model}}
    if any(d['confirmed'] for d in kept): return out | {'verified': 'confirmed'}
    return out | {'verified': 'disputed', 'confidence': min(first['confidence'], 0.3),
                  'summary': (first['summary'] + ' Second model saw no such defect: check visually before acting.').strip()}

def assess_verified(image, class_name, original_size):
    first = assess(image, class_name, original_size)
    return first | {'verified': 'not_run'} if first.get('skipped') or not first['defects'] else verify(image, class_name, original_size, first)
