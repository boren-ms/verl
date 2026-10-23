"""Read-only, exact-lineage step580 Fix/R16 validation analysis.

All inputs are frozen copies; this script never fetches or mutates job state.
Run from the repository root with the openai virtualenv and PYTHONPATH=.
"""
from collections import Counter, defaultdict
import datetime as dt
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import jiwer
from kaldialign import align
from recipe.phimm.reward.asr_eval import measure_openasr_en_wer
from recipe.phimm.reward.asr_response import get_hyp_text, get_asr_text, parse_task_output
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer
from recipe.phimm.utils.open_asr_normalizer.hf_english_normalizer import _HFEnglishTextNormalizer


ROOT = Path(__file__).resolve().parent
REPO = Path('/home/boren/code/verl')
SNAP = ROOT / 'source_snapshot'
SUMMARY = json.loads((ROOT / 'examples_summary.frozen.json').read_text())
MANIFEST = json.loads((ROOT / 'source_manifest.json').read_text())
DATASETS = ['earning_chunk_verb', 'openml_bad', 'ls_clean', 'ls_other']
ROLES = ['baseline', 'target']
NORM = _HFEnglishTextNormalizer()
CATS = ['newly_failed', 'worsening_persistent', 'improved', 'persistent', 'unchanged_correct']


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def row_identity(row):
    assert row.get('id') is not None, 'all this cohort must have stable IDs'
    return str(row['id'])


def make_units(ds, rows):
    groups = defaultdict(list)
    for row in rows:
        key = row['parent_audio_path'] if ds == 'earning_chunk_verb' else row_identity(row)
        assert key, 'missing identity'
        groups[key].append(row)
    units = {}
    for key, members in groups.items():
        members.sort(key=lambda r: (r['seg_start'] or 0, r['_line']))
        counted = [r for r in members if r['n_err'] is not None or r['n_ref'] is not None]
        assert len(counted) == 1 and counted[0] is members[0], (ds, key, 'counted head')
        assert len({r['gts'] for r in members}) == 1, (ds, key, 'conflicting reference')
        assert len({row_identity(r) for r in members}) == len(members), (ds, key, 'duplicate segment')
        assert all(get_hyp_text(r['output'], version=2609) == r['clean_output'] for r in members)
        if ds == 'earning_chunk_verb':
            assert sorted(r['seg_index'] for r in members) == list(range(len(members)))
            assert {r['n_segments'] for r in members} == {len(members)}
            assert [r['seg_index'] for r in members] == list(range(len(members)))
            assert all(r['seg_start'] < r['seg_end'] for r in members)
            assert all(a['seg_end'] <= b['seg_start'] + 1e-6 for a, b in zip(members, members[1:]))
        else:
            assert len(members) == 1, (ds, key, 'duplicate utterance')
        head = members[0]
        units[key] = {
            'dataset': ds, 'stable_id': key, 'reference': head['gts'],
            'hypothesis': '\n'.join(r['clean_output'] for r in members),
            'language': head['language'], 'n_err': counted[0]['n_err'], 'n_ref': counted[0]['n_ref'],
            'nb_err': counted[0]['nb_err'], 'nb_ref': counted[0]['nb_ref'],
            'lines': [r['_line'] for r in members], 'members': members,
            'segment_metadata': [(row_identity(r), r['seg_index'], r['seg_start'], r['seg_end'], r['audio_path']) for r in members],
        }
    return units


def score(unit, hyp=None):
    hypothesis = unit['hypothesis'] if hyp is None else hyp
    if unit['dataset'] != 'openml_bad':
        value = measure_openasr_en_wer(hypothesis, unit['reference'])
        return {k: value[k] for k in ['n_err', 'n_ref', 'n_sub', 'n_del', 'n_ins']}
    hn, rn = normalize_for_wer(hypothesis, unit['reference'], lang=get_language_code(unit['language']), merge_compounds=True)
    v = jiwer.process_words(rn, hn)
    return {'n_err': v.substitutions + v.deletions + v.insertions, 'n_ref': max(len(rn.split()), 1),
            'n_sub': v.substitutions, 'n_del': v.deletions, 'n_ins': v.insertions}


def category(b, t, basis='rate'):
    direction = (t['n_err'] * b['n_ref'] - b['n_err'] * t['n_ref']) if basis == 'rate' else t['n_err'] - b['n_err']
    if direction < 0:
        return 'improved'
    if direction > 0:
        return 'newly_failed' if b['n_err'] == 0 else 'worsening_persistent'
    return 'persistent' if b['n_err'] else 'unchanged_correct'


def alignment(words, ref):
    eps = '<alignment-epsilon>'
    cells = []
    rp = hp = 0
    for rw, hw in align(ref, words, eps, merge_compounds=True):
        nr, nh = (0 if rw == eps else len(rw.split())), (0 if hw == eps else len(hw.split()))
        kind = 'deletion' if hw == eps else 'insertion' if rw == eps else 'equal' if rw.replace(' ', '') == hw.replace(' ', '') else 'substitution'
        cells.append({'kind': kind, 'ref_start': rp, 'ref_end': rp + nr, 'hyp_start': hp, 'hyp_end': hp + nh})
        rp += nr
        hp += nh
    return cells


def excerpt(b, t):
    if max(len(b['reference']), len(b['hypothesis']), len(t['hypothesis'])) <= 1800:
        return {'reference': b['reference'], 'fix': b['hypothesis'], 'r16': t['hypothesis'], 'scope': 'complete scoring unit'}
    if b['dataset'] != 'earning_chunk_verb':
        return {'reference': b['reference'][:1500], 'fix': b['hypothesis'][:1500], 'r16': t['hypothesis'][:1500],
                'scope': 'first 1500 characters of each field; counts are for the full utterance'}
    ref, bh, th = NORM(b['reference']).split(), NORM(b['hypothesis']).split(), NORM(t['hypothesis']).split()
    ba, ta = alignment(bh, ref), alignment(th, ref)
    base_kinds = {i: c['kind'] for c in ba for i in range(c['ref_start'], c['ref_end'])}
    newly_deleted = [c for c in ta if c['kind'] == 'deletion' and all(base_kinds[i] == 'equal' for i in range(c['ref_start'], c['ref_end']))]
    groups = []
    for c in newly_deleted:
        if groups and groups[-1][-1]['ref_end'] == c['ref_start']:
            groups[-1].append(c)
        else:
            groups.append([c])
    selected = max(groups, key=lambda g: g[-1]['ref_end'] - g[0]['ref_start']) if groups else None
    if selected:
        start, end = max(0, selected[0]['ref_start'] - 8), min(len(ref), selected[-1]['ref_end'] + 8)
    else:
        changes = [c for c in difflib.SequenceMatcher(None, bh, th, autojunk=False).get_opcodes() if c[0] != 'equal']
        pos = changes[0][1] if changes else 0
        center = next((c['ref_start'] for c in ba if c['hyp_start'] <= pos < c['hyp_end']), 0)
        start, end = max(0, center - 10), min(len(ref), center + 25)

    def window(words, cells):
        text, positions = [], []
        for c in cells:
            if c['kind'] == 'equal' and c['ref_end'] - c['ref_start'] == c['hyp_end'] - c['hyp_start']:
                lo, hi = max(start, c['ref_start']), min(end, c['ref_end'])
                if lo < hi:
                    hs, he = c['hyp_start'] + lo - c['ref_start'], c['hyp_start'] + hi - c['ref_start']
                    text.extend(words[hs:he]); positions.extend(range(hs, he))
            elif c['ref_start'] < end and c['ref_end'] > start or c['ref_start'] == c['ref_end'] and start <= c['ref_start'] < end:
                text.extend(words[c['hyp_start']:c['hyp_end']]); positions.extend(range(c['hyp_start'], c['hyp_end']))
        return ' '.join(text), positions

    def mapped_lines(unit, words, positions):
        tokens = [(word, r['_line']) for r in unit['members'] for word in NORM(r['clean_output']).split()]
        mapping = {}
        for block in difflib.SequenceMatcher(None, [v[0] for v in tokens], words, autojunk=False).get_matching_blocks():
            mapping.update({block.b + i: tokens[block.a + i][1] for i in range(block.size)})
        return {'lines': sorted({mapping[i] for i in positions if i in mapping}), 'unmapped_tokens': sum(i not in mapping for i in positions)}

    bw, bp = window(bh, ba); tw, tp = window(th, ta)
    return {'reference': ' '.join(ref[start:end]), 'fix': bw, 'r16': tw,
            'scope': 'normalized compound-aware alignment window; all error counts refer to the full parent',
            'reference_token_window': [start, end],
            'selection': 'longest new R16 deletion span with words correct in Fix' if selected else 'first changed hypothesis span',
            'new_r16_deletion_span': ' '.join(ref[selected[0]['ref_start']:selected[-1]['ref_end']]) if selected else None,
            'source_excerpt_lines': {'baseline': mapped_lines(b, bh, bp), 'target': mapped_lines(t, th, tp)}}


def format_diagnostics(rows):
    strict = re.compile(r'(?:\A|\n)<src=[^>\n]+><tgt=[^>\n]+>[^\S\n]*\n')
    result = {'raw_rows': len(rows), 'strict_2609_header_rows': 0, 'clean_output_reproduced_rows': 0,
              'first_header_patterns': Counter(), 'glued_transcription_token_removed': []}
    for row in rows:
        raw, clean = row['output'], row['clean_output']
        valid = bool(strict.match(raw.strip()))
        result['strict_2609_header_rows'] += valid
        reproduced = get_hyp_text(raw, version=2609) == clean
        result['clean_output_reproduced_rows'] += reproduced
        first = raw.split('\n', 1)[0]
        if valid:
            pattern = 'valid_adjacent_src_tgt_header'
        elif re.fullmatch(r'<src=[^>]+>', first):
            pattern = 'src_only_closed_no_tgt'
        elif first.startswith('<src=') and '>' not in first:
            pattern = 'src_header_without_closing_angle'
        elif first.startswith('<src='):
            pattern = 'other_src_prefixed_malformed_header'
        else:
            pattern = 'other'
        result['first_header_patterns'][pattern] += 1
        prefix = '<src=' + str(row['language'])
        match = re.match(r'^\S+', raw)
        token = match.group(0) if match else ''
        suffix = token[len(prefix):] if token.startswith(prefix) else ''
        suffix = re.sub(r'<[^>]*>', '', suffix).lstrip('>')
        if reproduced and suffix and any(c.isalpha() for c in suffix) and not get_asr_text(parse_task_output(raw, version=2609)):
            result['glued_transcription_token_removed'].append({'id': row_identity(row), 'source_line': row['_line'],
                'raw_excerpt': raw[:400], 'clean_excerpt': clean[:400], 'reference': row['gts'],
                'removed_token': token, 'consumed_transcription_suffix': suffix, 'n_err': row['n_err'], 'n_ref': row['n_ref']})
    result['first_header_patterns'] = dict(result['first_header_patterns'])
    return result


assert SUMMARY['status'] == 'complete' and SUMMARY['selected_common_validation_step'] == 580
assert SUMMARY['observation_binding']['lineage_fingerprint'].startswith('5ff1c2c0449461fc')
for entry in MANIFEST['files']:
    assert digest(Path(entry['snapshot']).read_bytes()) == entry['sha256']

# Record the exact scorer sources used by both deployed segments and compare
# the current imports against each. Full count reproduction is checked below.
source_checks = []
for source in ['recipe/phimm/reward/asr_eval.py', 'recipe/phimm/reward/asr_response.py',
               'recipe/phimm/utils/open_asr_normalizer/eval_utils.py', 'recipe/phimm/utils/open_asr_normalizer/hf_english_normalizer.py',
               'recipe/phimm/utils/open_asr_normalizer/data_utils.py', 'recipe/phimm/utils/languages.py']:
    current = (REPO / source).read_bytes()
    entry = {'path': source, 'current_sha256': digest(current), 'roles': {}}
    for role in ROLES:
        commit = SUMMARY['observation_binding']['runs'][role]['lineage']['source_commit']
        deployed = subprocess.check_output(['git', 'show', f'{commit}:{source}'], cwd=REPO)
        dest = ROOT / 'scorer_snapshot' / role / source
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(deployed)
        entry['roles'][role] = {'commit': commit, 'sha256': digest(deployed), 'matches_current': deployed == current}
        if source != 'recipe/phimm/reward/asr_response.py':
            assert deployed == current, (role, source, 'scorer source drift')
    assert entry['roles']['baseline']['sha256'] == entry['roles']['target']['sha256'], (source, 'deployed scorer differs')
    source_checks.append(entry)

# Current parser has a later 2607-only change. Replay the byte-identical deployed
# parser for both runs so header/extraction diagnostics do not depend on it.
parser_path = ROOT / 'scorer_snapshot/baseline/recipe/phimm/reward/asr_response.py'
parser_spec = importlib.util.spec_from_file_location('r16_deployed_asr_response', parser_path)
parser_module = importlib.util.module_from_spec(parser_spec)
parser_spec.loader.exec_module(parser_module)
get_hyp_text, get_asr_text, parse_task_output = parser_module.get_hyp_text, parser_module.get_asr_text, parser_module.parse_task_output

all_rows, units, scores = {}, {}, {}
scorer_mismatches = []
for role in ROLES:
    all_rows[role], units[role], scores[role] = {}, {}, {}
    for ds in DATASETS:
        p = SNAP / role / f'{ds}_580.jsonl'
        rows = [dict(json.loads(line), _line=i) for i, line in enumerate(p.read_text().splitlines(), 1)]
        assert all(r['data_source'] == ds and r['step'] == 580 for r in rows)
        assert len({row_identity(r) for r in rows}) == len(rows)
        all_rows[role][ds] = rows
        units[role][ds] = make_units(ds, rows)
        scores[role][ds] = {}
        for key, unit in units[role][ds].items():
            sd = score(unit); scores[role][ds][key] = sd
            if (sd['n_err'], sd['n_ref']) != (unit['n_err'], unit['n_ref']):
                scorer_mismatches.append({'role': role, 'dataset': ds, 'id': key,
                    'logged_n_err': unit['n_err'], 'logged_n_ref': unit['n_ref'], 'reproduced': sd})

results, all_cases = {}, []
for ds in DATASETS:
    bu, tu = units['baseline'][ds], units['target'][ds]
    assert bu.keys() == tu.keys(), (ds, 'cohort coverage')
    cases = []
    for key in sorted(bu):
        b, t = bu[key], tu[key]
        assert b['reference'] == t['reference']
        assert b['segment_metadata'] == t['segment_metadata']
        assert b['language'] == t['language']
        bs, ts = scores['baseline'][ds][key], scores['target'][ds][key]
        c = {'dataset': ds, 'stable_id': key, 'language': b['language'], 'category': category(b, t),
             'count_category': category(b, t, 'count'), 'recovered': b['n_err'] > 0 and t['n_err'] == 0,
             'baseline_lines': b['lines'], 'target_lines': t['lines'], 'segment_count': len(b['members']),
             'reference_sha256': digest(b['reference'].encode()), 'reference_equal': True,
             'baseline': dict(bs, p_err=b['n_err']/b['n_ref']), 'target': dict(ts, p_err=t['n_err']/t['n_ref']),
             'delta_n_err': t['n_err'] - b['n_err'], 'delta_p_err_pp': 100*(t['n_err']/t['n_ref']-b['n_err']/b['n_ref']),
             'delta_sdi': {k: ts[k]-bs[k] for k in ['n_sub','n_del','n_ins']},
             'baseline_source': str(SNAP / 'baseline' / f'{ds}_580.jsonl'),
             'target_source': str(SNAP / 'target' / f'{ds}_580.jsonl')}
        cases.append(c); all_cases.append(c)
    totals = {}
    for role in ROLES:
        sums = {k: sum(v[k] for v in scores[role][ds].values()) for k in ['n_err','n_ref','n_sub','n_del','n_ins']}
        sums['p_err'] = sums['n_err']/sums['n_ref']
        sums['pb_err'] = None
        if all(u['nb_ref'] is not None and u['nb_err'] is not None for u in units[role][ds].values()):
            sums['nb_err'] = sum(u['nb_err'] for u in units[role][ds].values())
            sums['nb_ref'] = sum(u['nb_ref'] for u in units[role][ds].values())
            sums['pb_err'] = sums['nb_err']/sums['nb_ref'] if sums['nb_ref'] else None
        logged = SUMMARY['observation_binding']['runs'][role]['primary_p_err'][ds]
        assert abs(sums['p_err']-logged) < 1e-12, (role,ds,'primary scalar')
        totals[role] = sums
    cc = Counter(c['category'] for c in cases)
    countcc = Counter(c['count_category'] for c in cases)
    balance = {cat: {'units': cc[cat], 'delta_n_err': sum(c['delta_n_err'] for c in cases if c['category']==cat),
                     'delta_sdi': {k:sum(c['delta_sdi'][k] for c in cases if c['category']==cat) for k in ['n_sub','n_del','n_ins']}}
               for cat in CATS}
    assert sum(d['delta_n_err'] for d in balance.values()) == totals['target']['n_err']-totals['baseline']['n_err']
    examples = []
    for cat in CATS:
        candidates = [c for c in cases if c['category'] == cat]
        candidates.sort(key=lambda c: (-abs(c['delta_n_err']), -abs(c['delta_p_err_pp']), c['stable_id']))
        for c in candidates[:(2 if cat in ['newly_failed','worsening_persistent','improved'] else 1)]:
            if cat == 'unchanged_correct':
                continue
            examples.append(dict(c, excerpt=excerpt(bu[c['stable_id']],tu[c['stable_id']])))
    result = {'unit': 'parent_recording' if ds == 'earning_chunk_verb' else 'utterance',
              'raw_rows': {r:len(all_rows[r][ds]) for r in ROLES}, 'matched': len(cases),
              'baseline_only': 0, 'target_only': 0, 'duplicate_ids': 0, 'reference_conflicts': 0, 'metadata_conflicts': 0,
              'totals': totals, 'delta_p_err_pp': 100*(totals['target']['p_err']-totals['baseline']['p_err']),
              'delta_n_err': totals['target']['n_err']-totals['baseline']['n_err'],
              'delta_sdi': {k:totals['target'][k]-totals['baseline'][k] for k in ['n_sub','n_del','n_ins']},
              'category_counts': {cat:cc[cat] for cat in CATS}, 'count_category_counts': {cat:countcc[cat] for cat in CATS},
              'recovered_subset_of_improved': sum(c['recovered'] for c in cases), 'error_balance': balance,
              'denominator_difference_cases': [c for c in cases if c['baseline']['n_ref']!=c['target']['n_ref']],
              'rate_vs_count_category_differences': [c for c in cases if c['category']!=c['count_category']],
              'format_diagnostics': {r:format_diagnostics(all_rows[r][ds]) for r in ROLES}, 'examples': examples}
    for cat in CATS:
        assert result['category_counts'][cat] == SUMMARY['per_dataset'][ds]['category_counts'][cat]
    if ds == 'earning_chunk_verb':
        result['parent_contributions'] = [dict(c, parent_filename=Path(c['stable_id']).name,
            contribution_pp=100*c['delta_n_err']/totals['baseline']['n_ref'],
            share_of_net_added_errors=c['delta_n_err']/result['delta_n_err'],
            contribution_sdi_pp={k:100*c['delta_sdi'][k]/totals['baseline']['n_ref'] for k in ['n_sub','n_del','n_ins']},
            excerpt=excerpt(bu[c['stable_id']],tu[c['stable_id']])) for c in sorted(cases,key=lambda c:-c['delta_n_err'])]
        assert abs(sum(c['contribution_pp'] for c in result['parent_contributions'])-result['delta_p_err_pp'])<1e-10
    if ds == 'openml_bad':
        sensitivity = {}
        for role in ROLES:
            changes=[]; ne=nr=0
            for key,u in units[role][ds].items():
                altered = u['hypothesis'].replace('(','').replace(')','')
                v=score(u,altered); ne+=v['n_err']; nr+=v['n_ref']
                if (v['n_err'],v['n_ref'])!=(u['n_err'],u['n_ref']):
                    changes.append({'id':key,'lines':u['lines'],'parenthesized_spans':re.findall(r'\([^)]*\)',u['hypothesis']),
                        'official_n_err':u['n_err'],'official_n_ref':u['n_ref'],'diagnostic':v})
            sensitivity[role]={'n_err':ne,'n_ref':nr,'p_err':ne/nr,'affected_count':len(changes),'affected_cases':changes}
        result['ascii_parenthesis_sensitivity']=sensitivity
        by_lang={}
        for lang in sorted({c['language'] for c in cases}):
            subset=[c for c in cases if c['language']==lang]
            by_lang[lang]={'matched':len(subset),'category_counts':dict(Counter(c['category'] for c in subset)),
                'delta_n_err':sum(c['delta_n_err'] for c in subset),
                'baseline_n_err':sum(c['baseline']['n_err'] for c in subset),'target_n_err':sum(c['target']['n_err'] for c in subset),
                'baseline_n_ref':sum(c['baseline']['n_ref'] for c in subset),'target_n_ref':sum(c['target']['n_ref'] for c in subset)}
        result['by_language']=by_lang
    results[ds]=result

assert not scorer_mismatches, scorer_mismatches
try:
    packages={k:importlib.metadata.version(k) for k in ['kaldialign','jiwer','editdistance','num2words']}
except importlib.metadata.PackageNotFoundError:
    packages={'note':'see imported runtime; metadata missing for a dependency'}
analysis={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'step':580,
    'comparison':{'baseline':'remax_dev_base0_v1h2a_fix','target':'remax_dev_base0_v1h2a_fix_bcr (R16)',
                  'observation_binding':SUMMARY['observation_binding']},
    'source_manifest':str(ROOT/'source_manifest.json'),'frozen_summary':str(ROOT/'examples_summary.frozen.json'),
    'matched_cases_path':str(ROOT/'matched_units_step580.jsonl'),
    'matching_policy':'dataset+unique stable id; Earnings by parent_audio_path with identical complete sorted segment identities/bounds and identical full-parent gts. No slot/rank matching.',
    'count_policy':'One nonnull counted group head per Earnings parent. Null counts are never zero. Other datasets count one stable-ID utterance once.',
    'category_policy':'Primary categories compare exact cross-products of each run logged n_err/n_ref. newly_failed: Fix n_err=0,R16 n_err>0; worsening_persistent: both errorful,R16 rate higher; improved: R16 rate lower; persistent: equal nonzero rate; unchanged_correct: both zero. count_category is also retained for checking denominator-sensitive cases.',
    'sdi_method':'English: exact deployed measure_openasr_en_wer + kaldialign batch_error_rate merge_compounds=True. OpenML: exact deployed normalize_for_wer(language,merge_compounds=True) then jiwer minimum-edit decomposition. OpenML tie-breaking can change S/D/I allocation; all S+D+I and n_ref must reproduce logged values.',
    'scorer_source_checks':source_checks,'runtime_packages':packages,'scorer_reproduction_mismatches':scorer_mismatches,
    'per_dataset':results,
    'limitations':['No pooled cross-dataset error rate or mean is defined.',
        'Only six independent parent recordings in Earnings; 341 chunks are not 341 independent scoring units.',
        'Same optimizer step does not equal same fresh-data exposure. R16 currently replay16/fresh112/buffer400; its first50 inherited updates used replay12/fresh116 and ReplayLossNormalizer. Fix is fresh128. This is not a clean replay-only single-factor ablation.',
        'R16 buffers reset on recovery and data reconstruction is not fully seeded; resumed lineage and discarded updates are governed by the frozen observation binding.',
        'Format/header failure is a raw text fact, not proof of incorrect transcription language or an acoustic failure. The shown extraction signature proves specific removed text only.',
        'ASCII-parenthesis sensitivity is a paired diagnostic that preserves enclosed text and the official references; it does not replace the logged validation metrics or verify acoustic correctness.',
        'Selected val/generations tables are ranks from mixed datasets and cannot estimate failure prevalence. This analysis uses all complete dumps.',
        'Error decomposition and examples describe this checkpoint; no causal effect of BCR is established.']}
write_json(ROOT/'analysis.json',analysis)
with (ROOT/'matched_units_step580.jsonl').open('w') as f:
    for c in all_cases:f.write(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'step':580,'matched':sum(r['matched'] for r in results.values()),'scorer_mismatches':len(scorer_mismatches),
    'datasets':{ds:{k:r[k] for k in ['matched','delta_n_err','delta_p_err_pp','delta_sdi','category_counts']} for ds,r in results.items()}},ensure_ascii=False,indent=2))
