"""Manufacturer identification for detected equipment (sector antennas, microwave dishes).

Reviewers tag the manufacturer/model on each box; optionally a local vision-language model served by Ollama
(default qwen3-vl:8b) suggests one from a crop of the original image, in two stages:
  1. logo / printed text on the equipment - the strongest evidence;
  2. if none is legible, the physical structure: the model describes the design (radome profile, end caps, ODU,
     mounts) and compares it with reference photos of equipment whose manufacturer is known - reviewer-confirmed
     tags plus images in VENDOR_REFERENCE_DIR/<class>/<Manufacturer>/, one manufacturer at a time (small VLMs judge
     pairs reliably but not line-ups). The model's own recognition of a design is only reported as an unverified hint in the
     evidence, never as the manufacturer: in testing it attributed generic white panels to a specific vendor.
"""
import base64, csv, io, json, os, string, threading, urllib.error, urllib.request
from pathlib import Path

VLM_URL = os.getenv('VLM_URL', 'http://127.0.0.1:11434').rstrip('/')
VLM_MODEL = os.getenv('VLM_MODEL', 'qwen3-vl:8b-instruct')  # instruct variant: the thinking one reasons past the output limit; empty disables
VLM_COMPARE_MODEL = os.getenv('VLM_COMPARE_MODEL', 'qwen3-vl:8b')  # thinking variant: slower, but the instruct one agrees with every reference
VLM_TIMEOUT = float(os.getenv('VLM_TIMEOUT', '300'))
VENDOR_CLASSES = [c.strip() for c in os.getenv('VENDOR_CLASSES', 'sector_antenna,microwave_dish').split(',') if c.strip()]
VENDOR_REFERENCE_DIR = Path(os.getenv('VENDOR_REFERENCE_DIR', str(Path(__file__).resolve().parent.parent / 'vendor_refs')))
REFS_PER_VENDOR, MAX_REF_VENDORS = 2, 6  # keeps the reference collage readable for the model
MANUFACTURERS = {
    'sector_antenna': ['Ericsson', 'Huawei', 'Nokia', 'ZTE', 'CommScope', 'Kathrein', 'RFS', 'Amphenol', 'Rosenberger', 'Comba', 'Tongyu', 'Samsung', 'Mavenir'],
    'microwave_dish': ['Ericsson', 'Huawei', 'Nokia', 'NEC', 'Ceragon', 'Aviat', 'SIAE Microelettronica', 'ZTE', 'CommScope', 'RFS', 'Cambium', 'Ubiquiti'],
}
# structural features the model has to describe before it may match anything
FEATURES = {
    'sector_antenna': {
        'radome_profile': 'outline/cross-section of the radome: flat rectangular panel, rounded edges, curved/oval front, slim tube',
        'end_caps': 'top and bottom end caps: flat, rounded, tapered, recessed, contrasting colour',
        'proportions': 'approximate width:height ratio and depth (thin panel vs thick box)',
        'active_unit': 'integrated radio / AAU or heat-sink fins on the back, or a passive antenna',
        'connectors_and_brackets': 'connector ports, RET/AISG units, bracket and clamp style',
        'surface': 'radome colour, texture, ribs, seams, vents'},
    'microwave_dish': {
        'dish_type': 'shrouded high-performance dish with flat radome, open parabolic, curved radome, flat panel',
        'radome': 'radome colour, shape and rim/shroud details',
        'odu': 'outdoor unit: mounted directly behind the dish or separate; its shape, colour, fins, handles, latches',
        'mount': 'pole mount, bracket, azimuth/elevation adjusters, struts',
        'proportions': 'apparent dish size relative to the pole and ODU',
        'other': 'any other distinctive structural detail'},
}
# max confidence per kind of evidence: in testing the comparison step still matched an antenna to RRU photos at 0.95,
# so a structure match is a lead for the reviewer, never as strong as a legible logo
BASIS_CAP = {'logo_text': 1.0, 'reference_match': 0.6, 'none': 0.2}
AMBIGUITY_MARGIN = 0.15  # two manufacturers matching within this similarity -> Unknown
_lock = threading.Lock()

# catalog views in the order they resemble a drone's oblique view; connector-end shots never appear in drone imagery
VIEW_PRIORITY = ['side_installed', 'side_tilted', 'front_three_quarter', 'rear_three_quarter', 'installed', 'profile', 'front', 'housing_view', 'rear', 'side']
def view_matchable(view): return not (view or '').lower().startswith(('bottom', 'connector', 'top_connector', 'port'))

def manufacturers(class_name):
    base = MANUFACTURERS.get(class_name) or sorted({m for ms in MANUFACTURERS.values() for m in ms})
    return base + sorted({r['manufacturer'] for r in library(class_name)} - set(base))  # plus anyone in the reference library

def _installed():
    """Model names Ollama has pulled, or None when it can't be reached."""
    try:
        with urllib.request.urlopen(f'{VLM_URL}/api/tags', timeout=3) as r: names = {m['name'] for m in json.load(r).get('models', [])}
    except (urllib.error.URLError, OSError, ValueError): return None
    return names | {n.removesuffix(':latest') for n in names}

def status():
    if not VLM_MODEL: return 'disabled'
    names = _installed()
    return 'unreachable' if names is None else 'ready' if VLM_MODEL in names else 'model_missing'

def compare_model():
    """Model for the one-to-one structure comparisons, or None when it isn't available. No silent fallback to the main
    model: in testing the instruct model called a dish 'the same product' as panel antennas (0.90)."""
    if VLM_COMPARE_MODEL and VLM_COMPARE_MODEL == VLM_MODEL: return VLM_MODEL  # explicitly chosen
    return VLM_COMPARE_MODEL if VLM_COMPARE_MODEL and VLM_COMPARE_MODEL in (_installed() or ()) else None

def library(class_name):
    """Every reference photo of a class: [{manufacturer, model, view, file, path, match, status, page, notes}].
    Rows come from references.csv (written by app.refs_import); images dropped into the folders by hand are listed too."""
    root, index, out, seen = VENDOR_REFERENCE_DIR / class_name, VENDOR_REFERENCE_DIR / 'references.csv', [], set()
    if index.is_file():
        with index.open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                p = VENDOR_REFERENCE_DIR / r['file'] if r.get('file') else None
                if r.get('class') == class_name and p and p.is_file():
                    out.append(r | {'path': p, 'match': r.get('match') != 'no'}); seen.add(p.resolve())
    if root.is_dir():
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} and p.resolve() not in seen:
                    out.append({'manufacturer': d.name, 'model': '', 'view': '', 'file': p.relative_to(VENDOR_REFERENCE_DIR).as_posix(), 'path': p, 'match': True,
                                'status': 'added by hand', 'page': '', 'notes': ''})
    return out

def view_rank(r):
    """Sort key: drone-like views first, views never seen from a drone last."""
    return (not r['match'], VIEW_PRIORITY.index(r['view']) if r['view'] in VIEW_PRIORITY else len(VIEW_PRIORITY))

def folder_references(class_name):
    """[(manufacturer, path)] usable for structure matching, best drone-like views first."""
    return [(r['manufacturer'], r['path']) for r in sorted((r for r in library(class_name) if r['match']), key=view_rank)]

def info():
    return {'classes': VENDOR_CLASSES, 'manufacturers': {c: manufacturers(c) + ['Unknown'] for c in VENDOR_CLASSES},
            'ai': {'status': status(), 'model': VLM_MODEL or None, 'compare_model': compare_model() if VLM_MODEL else None, 'url': VLM_URL},
            'reference_dir': str(VENDOR_REFERENCE_DIR), 'folder_references': {c: len(folder_references(c)) for c in VENDOR_CLASSES},
            'library_manufacturers': {c: sorted({r['manufacturer'] for r in library(c)}) for c in VENDOR_CLASSES}}

# brands owned by one manufacturer: their photos are compared together, so matching both is one answer, not an ambiguity.
# ANDREW is CommScope's base-station antenna brand (same NNHH product line; the business moved to Amphenol in 2025);
# Ericsson acquired Kathrein's antenna business in 2019. Edit to taste.
FAMILY = {'andrew': 'CommScope', 'kathrein': 'Ericsson'}
def family(maker): return FAMILY.get((maker or '').strip().lower(), maker)

def pick_references(refs):
    """{manufacturer family: [PIL image]} keeping REFS_PER_VENDOR images for the MAX_REF_VENDORS best-covered families."""
    by = {}
    for m, im in refs:
        if m and m != 'Unknown' and len(by.setdefault(family(m), [])) < REFS_PER_VENDOR: by[family(m)].append(im)
    return dict(sorted(by.items(), key=lambda kv: -len(kv[1]))[:MAX_REF_VENDORS])

def family_brands(refs):
    """{family: brands seen in its reference photos}, e.g. {'CommScope': ['ANDREW', 'CommScope']}."""
    out = {}
    for m, _ in refs:
        if m and m != 'Unknown': out.setdefault(family(m), set()).add(m)
    return {f: sorted(b) for f, b in out.items()}

def _font(size):
    from PIL import ImageFont
    try: return ImageFont.load_default(size=size)
    except (TypeError, OSError, ImportError): return ImageFont.load_default()

def collage(query, groups):
    """One image: the QUERY crop large on the left, reference tiles captioned A1, A2, B1 ... on the right.
    Letters stand in for manufacturer names so the model compares shapes, not brand priors."""
    from PIL import Image, ImageDraw
    cap, qh, th = 34, 560, 260
    q = query.convert('RGB'); q = q.resize((max(1, round(q.width * qh / q.height)), qh))
    tiles = []
    for letter, ims in zip(string.ascii_uppercase, groups.values()):
        for n, im in enumerate(ims, 1):
            im = im.convert('RGB'); im.thumbnail((th, th)); tiles.append((f'{letter}{n}', im))
    cols = max(1, min(4, len(tiles))); rows = -(-len(tiles) // cols)
    W, H = q.width + 30 + cols * (th + 10), max(qh + cap, rows * (th + cap + 10))
    out = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(out); f = _font(26)
    d.text((6, 2), 'QUERY', font=f, fill=(200, 0, 0)); out.paste(q, (0, cap))
    d.line([(q.width + 14, 0), (q.width + 14, H)], fill=(120, 120, 120), width=3)
    for i, (label, im) in enumerate(tiles):
        x, y = q.width + 30 + (i % cols) * (th + 10), (i // cols) * (th + cap + 10)
        d.text((x + 4, y + 2), label, font=f, fill=(0, 0, 160)); out.paste(im, (x + (th - im.width) // 2, y + cap))
    return out

def _jpeg(im, longest=672):
    im = im.convert('RGB'); scale = longest / max(im.size)
    if scale > 1: im = im.resize((round(im.width * scale), round(im.height * scale)))  # more vision tokens for small crops; adds no detail
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=95); return base64.b64encode(buf.getvalue()).decode()

def _json_in(text):
    i, j = text.find('{'), text.rfind('}')
    return json.loads(text[i:j + 1]) if 0 <= i < j else None

def _chat(prompt, image_b64, schema, model=None, think=False):
    # Thinking models must not decode greedily (Qwen's guidance): at temperature 0 qwen3-vl:8b looped until the token limit
    # (33k chars of reasoning, no answer). Use the recommended sampling with a fixed seed and a cap that fails fast.
    opts = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': 7, 'num_ctx': 16384, 'num_predict': 4000} if think else {'temperature': 0, 'num_ctx': 8192, 'num_predict': 3000}
    body = {'model': model or VLM_MODEL, 'stream': False, 'think': think, 'keep_alive': '10m', 'format': schema, 'options': opts,
            'messages': [{'role': 'user', 'content': prompt, 'images': [image_b64]}]}
    req = json.dumps(body).encode(); secs = 0.0
    for _ in range(2):  # one retry for an empty/unparseable answer
        with _lock:  # one GPU request at a time
            try:
                with urllib.request.urlopen(urllib.request.Request(f'{VLM_URL}/api/chat', req, {'Content-Type': 'application/json'}), timeout=VLM_TIMEOUT) as r: out = json.load(r)
            except urllib.error.HTTPError as e: raise RuntimeError(f'VLM error {e.code}: {e.read()[:300].decode(errors="replace")}') from e
            except (urllib.error.URLError, TimeoutError, OSError) as e: raise RuntimeError(f'VLM unreachable or timed out: {e}') from e
        secs += out.get('total_duration', 0) / 1e9; msg = out.get('message') or {}
        for text in (msg.get('content') or '', msg.get('thinking') or ''):
            try:
                if (a := _json_in(text)) is not None: return a, secs
            except ValueError: pass
    raise RuntimeError(f'VLM returned no usable JSON: {str(out)[:300]}')

def _unit(v): return min(max(float(v or 0), 0), 1)

def _logo_pass(image, class_name, names):
    kind = class_name.replace('_', ' ')
    prompt = (f'This is a crop from a drone photo of a telecom tower showing one {kind}.\n'
              f'Identify the manufacturer ONLY from visible marks: a logo, printed brand or model text, or a product label. Known manufacturers: {", ".join(names)}.\n'
              'Shape is checked separately, so if no brand mark is legible answer "Unknown" with confidence below 0.3. '
              'Use "Other" with manufacturer_other only for a clearly visible brand not in the list. product_model only if its text is legible. '
              'evidence: one short sentence on what you saw (or why it is unknown). visible_text: any legible text, else empty.')
    schema = {'type': 'object', 'properties': {
        'manufacturer': {'type': 'string', 'enum': names + ['Other', 'Unknown']}, 'manufacturer_other': {'type': 'string'},
        'product_model': {'type': 'string'}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence': {'type': 'string'}, 'visible_text': {'type': 'string'}}, 'required': ['manufacturer', 'confidence', 'evidence']}
    a, secs = _chat(prompt, _jpeg(image), schema)
    m = (a.get('manufacturer_other') or '').strip() if a.get('manufacturer') == 'Other' else a.get('manufacturer') or 'Unknown'
    return {'manufacturer': m or 'Unknown', 'product_model': (a.get('product_model') or '').strip() or None, 'confidence': _unit(a.get('confidence')),
            'evidence': (a.get('evidence') or '').strip(), 'visible_text': (a.get('visible_text') or '').strip()[:200]}, secs

def _describe(image, class_name, names):
    """Structural description of the crop alone, plus the model's own (unverified) recognition of the design."""
    feats, kind = FEATURES.get(class_name, FEATURES['sector_antenna']), class_name.replace('_', ' ')
    prompt = (f'This is a crop from a drone photo of a telecom tower showing one {kind}. Ignore any text or logos.\n'
              'Describe its physical structure, one short phrase per field: ' + '; '.join(f'{k} = {v}' for k, v in feats.items()) + '.\n'
              f'own_knowledge_guess: a manufacturer from [{", ".join(names)}] only if this structure is characteristic of a product line you genuinely recognise; '
              'generic white radomes and common dish designs are "Unknown". own_knowledge_confidence: 0-1, low unless the design is truly distinctive. '
              'reasoning: one sentence naming the distinctive features, if any.')
    props = {'features': {'type': 'object', 'properties': {k: {'type': 'string'} for k in feats}, 'required': list(feats)},
             'own_knowledge_guess': {'type': 'string', 'enum': names + ['Unknown']}, 'own_knowledge_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
             'reasoning': {'type': 'string'}}
    a, secs = _chat(prompt, _jpeg(image), {'type': 'object', 'properties': props, 'required': list(props)})
    return {'features': {k: str(v).strip() for k, v in (a.get('features') or {}).items() if str(v).strip()},
            'guess': a.get('own_knowledge_guess') or 'Unknown', 'guess_confidence': _unit(a.get('own_knowledge_confidence')),
            'reasoning': (a.get('reasoning') or '').strip()}, secs

def _compare(image, class_name, refs):
    """One-to-one: is QUERY the same product as these reference photos of one manufacturer? Small VLMs judge pairs far better than line-ups."""
    kind = class_name.replace('_', ' ')
    prompt = ('The image is a collage made by us. The tile captioned QUERY on the left is the equipment to identify; the tiles captioned A1, A2 ... on the right '
              f'are photos of one known product line. QUERY should be a {kind}.\n'
              'Answer one question: does QUERY show the same product (same model/design) as the A tiles? '
              'It does not matter whether this design is common in the industry - only whether QUERY looks like the A tiles.\n'
              'same_equipment_type: is QUERY the same kind of equipment as the A tiles (e.g. both microwave dishes, both panel antennas)? '
              'Then compare the physical design: outline, proportions, end caps or radome/shroud, ODU shape and position, fins, mounts. '
              'Viewing angle, distance, image sharpness, lighting and background do not matter. '
              'matching_features and differing_features: short phrases. same_product: "yes" if the design details match, "no" if they clearly differ, otherwise "uncertain". '
              'similarity: 0-1. reasoning: one sentence, no manufacturer names.')
    schema = {'type': 'object', 'properties': {
        'same_equipment_type': {'type': 'boolean'}, 'matching_features': {'type': 'array', 'items': {'type': 'string'}},
        'differing_features': {'type': 'array', 'items': {'type': 'string'}}, 'same_product': {'type': 'string', 'enum': ['yes', 'no', 'uncertain']},
        'similarity': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'reasoning': {'type': 'string'}},
        'required': ['same_equipment_type', 'matching_features', 'differing_features', 'same_product', 'similarity', 'reasoning']}
    model = compare_model()
    a, secs = _chat(prompt, _jpeg(collage(image, {'A': refs}), 1400), schema, model, think=model != VLM_MODEL)
    ok = a.get('same_equipment_type') is True and a.get('same_product') == 'yes' and _unit(a.get('similarity')) >= 0.6
    return ok, _unit(a.get('similarity')), a, secs

def _decide(image, class_name, names, groups, brands=None):
    s, secs = _describe(image, class_name, names)
    checks = []
    checks, failed = [], []
    if groups and not compare_model():  # skip rather than trust a model that agrees with everything
        groups = {}; skipped = f'Structure comparison skipped: model {VLM_COMPARE_MODEL or "(VLM_COMPARE_MODEL unset)"} is not installed (ollama pull {VLM_COMPARE_MODEL or "qwen3-vl:8b"}).'
    else: skipped = None
    for maker, refs in groups.items():
        try: ok, sim, v, t = _compare(image, class_name, refs); secs += t; checks.append((ok, sim, maker, v))
        except RuntimeError: failed.append(maker)  # one comparison that never answers shouldn't sink the whole suggestion
    hits = sorted((c for c in checks if c[0]), key=lambda c: -c[1])
    candidates = [{'manufacturer': h[2], 'similarity': round(h[1], 3), 'brands': (brands or {}).get(h[2], [h[2]]), 'matching_features': (h[3].get('matching_features') or [])[:5]} for h in hits]
    m, basis, conf, notes = 'Unknown', 'none', 0.0, []
    if skipped: notes.append(skipped)
    if failed: notes.append(f'Comparison with {", ".join(failed)} gave no answer and was skipped.')
    if len(hits) > 1 and hits[0][1] - hits[1][1] < AMBIGUITY_MARGIN:
        notes.append('Ambiguous: the structure matches reference photos of ' + ' and '.join(f'{h[2]} ({h[1]:.2f})' for h in hits[:3]) + '; pick one of the candidates, or use a logo or closer photo.')
    elif hits:
        _, sim, m, v = hits[0]; basis, conf = 'reference_match', sim
        also = [b for b in (brands or {}).get(m, []) if b != m]
        notes.append(f'Structure matches reference photos of {m}' + (f' (incl. its {", ".join(also)} brand)' if also else '') + ': ' + (', '.join(v.get('matching_features') or []) or 'design details match')[:180] + '.')
        if len(hits) > 1: notes.append('Weaker match: ' + ', '.join(f'{h[2]} ({h[1]:.2f})' for h in hits[1:]) + '.')
    elif checks:
        best = max(checks, key=lambda c: c[1])
        notes.append(f'No reference matched; closest was {best[2]} ({best[1]:.2f})' + (f", differs in: {', '.join(best[3].get('differing_features') or [])[:160]}" if best[3].get('differing_features') else '') + '.')
    elif not failed and not skipped: notes.append('No reference photos for this class yet, so the structure could not be matched.')
    hint = s['guess'] if basis == 'none' and s['guess'] != 'Unknown' and s['guess_confidence'] >= 0.4 else None
    if hint: notes.append(f'Unverified hint: the model thinks the shape resembles {hint} designs; confirm with a logo or a reference photo before tagging.')
    if s['reasoning'] and basis == 'none': notes.append(s['reasoning'])
    return {'manufacturer': m, 'confidence': min(conf, BASIS_CAP[basis]), 'basis': basis, 'reasoning': ' '.join(notes), 'features': s['features'], 'hint': hint,
            'candidates': candidates, 'ambiguous': basis == 'none' and len(candidates) > 1}, secs

def suggest(image, class_name, references=()):
    """Manufacturer suggestion for one PIL crop. references: [(manufacturer, PIL image)] of known equipment of this class.
    basis says what the answer rests on: logo_text, reference_match (verified one-to-one) or none."""
    names = manufacturers(class_name)
    logo, t1 = _logo_pass(image, class_name, names)
    base = {'model': VLM_MODEL, 'visible_text': logo['visible_text']}
    if logo['manufacturer'] != 'Unknown':
        return base | {'manufacturer': logo['manufacturer'], 'product_model': logo['product_model'], 'confidence': round(logo['confidence'], 3), 'basis': 'logo_text',
                       'evidence': logo['evidence'][:500], 'features': {}, 'reference_vendors': [], 'structure_hint': None, 'candidates': [], 'ambiguous': False, 'seconds': round(t1, 1)}
    groups = pick_references(references)
    s, t2 = _decide(image, class_name, names, groups, family_brands(references))
    feats = '; '.join(f'{k.replace("_", " ")}: {v}' for k, v in s['features'].items())
    return base | {'manufacturer': s['manufacturer'], 'product_model': None, 'confidence': round(s['confidence'], 3), 'basis': s['basis'],
                   'evidence': f"{s['reasoning']} Structure: {feats}"[:500], 'features': s['features'], 'reference_vendors': list(groups), 'structure_hint': s['hint'],
                   'candidates': s['candidates'], 'ambiguous': s['ambiguous'], 'seconds': round(t1 + t2, 1)}
