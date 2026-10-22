"""Enrich frozen step580 results and write a readable report + figure."""
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import jiwer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.open_asr_normalizer.eval_utils import normalize_for_wer

ROOT=Path(__file__).resolve().parent
A=json.loads((ROOT/'analysis.json').read_text())
ROWS={}
for role in ['baseline','target']:
    ROWS[role]={}
    for ds in A['per_dataset']:
        ROWS[role][ds]={}
        for i,line in enumerate((ROOT/'source_snapshot'/role/f'{ds}_580.jsonl').read_text().splitlines(),1):
            row=json.loads(line);row['_line']=i;ROWS[role][ds][str(row['id'])]=row


def write_json(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rescore(row,hyp):
    hn,rn=normalize_for_wer(hyp,row['gts'],lang=get_language_code(row['language']),merge_compounds=True)
    v=jiwer.process_words(rn,hn)
    return {'n_err':v.substitutions+v.deletions+v.insertions,'n_ref':max(len(rn.split()),1),
            'n_sub':v.substitutions,'n_del':v.deletions,'n_ins':v.insertions,
            'normalized_reference':rn,'normalized_hypothesis':hn}


paren=[]
assert all('(' not in r['gts'] and ')' not in r['gts'] for role in ['baseline','target'] for r in ROWS[role]['openml_bad'].values())
for key,b in ROWS['baseline']['openml_bad'].items():
    t=ROWS['target']['openml_bad'][key]
    bp=re.findall(r'\([^)]*\)',b['clean_output']);tp=re.findall(r'\([^)]*\)',t['clean_output'])
    if not bp and not tp:continue
    bd=rescore(b,b['clean_output'].replace('(','').replace(')',''))
    td=rescore(t,t['clean_output'].replace('(','').replace(')',''))
    status='introduced' if tp and not bp else 'removed' if bp and not tp else 'present_in_both'
    paren.append({'id':key,'status':status,'baseline_line':b['_line'],'target_line':t['_line'],
        'language':b['language'],'baseline_spans':bp,'target_spans':tp,
        'official_baseline_n_err':b['n_err'],'official_target_n_err':t['n_err'],'official_delta_n_err':t['n_err']-b['n_err'],
        'diagnostic_baseline':bd,'diagnostic_target':td,'diagnostic_delta_n_err':td['n_err']-bd['n_err'],
        'differential_parenthesis_error_effect':(t['n_err']-td['n_err'])-(b['n_err']-bd['n_err']),
        'reference':b['gts'],'fix':b['clean_output'],'r16':t['clean_output']})
groups={}
for status in ['introduced','removed','present_in_both']:
    cases=[p for p in paren if p['status']==status]
    groups[status]={'units':len(cases),'official_delta_n_err':sum(p['official_delta_n_err'] for p in cases),
        'diagnostic_delta_n_err':sum(p['diagnostic_delta_n_err'] for p in cases),
        'differential_parenthesis_error_effect':sum(p['differential_parenthesis_error_effect'] for p in cases),
        'ids':[p['id'] for p in cases]}
A['per_dataset']['openml_bad']['paired_parenthesis_diagnostic']={
    'scope':'All 393 matched OpenML utterances; flags only literal ASCII parenthesized spans in clean_output.',
    'method':'Apply the same removal of literal ASCII ( and ) delimiters to reference and hypothesis while retaining enclosed text in both runs. Verified all 393 raw references per side contain neither delimiter, so transforming references leaves them byte-identical; computed rescore is exactly equivalent to this symmetric operation. This is a counterfactual diagnostic, not official validation.',
    'raw_reference_ascii_parenthesis_character_rows':{'baseline':0,'target':0},
    'groups':groups,'cases':paren,
    'official_net_delta_n_err':35,'diagnostic_net_delta_n_err':25,'differential_parenthesis_error_effect':10,
    'interpretation':'The paired diagnostic reduces the net error difference by 10, but does not show that all 35 official added errors are acoustic errors or all are punctuation artifacts.'}

specific=[]
for key in ['8031031497862620363','17411379747303396458','common_voice_es_19741241']:
    b,t=(ROWS[r]['openml_bad'][key] for r in ['baseline','target'])
    c=next(x for x in [json.loads(l) for l in (ROOT/'matched_units_step580.jsonl').read_text().splitlines()] if x['dataset']=='openml_bad' and x['stable_id']==key)
    specific.append(dict(c,reference=b['gts'],fix_raw=b['output'],r16_raw=t['output'],fix_clean=b['clean_output'],r16_clean=t['clean_output'],
        fix_normalized=rescore(b,b['clean_output']),r16_normalized=rescore(t,t['clean_output']),
        interpretation='R16 raw contains only a malformed header and no transcription body; empty clean_output is consistent with no generated body, not evidence of a hidden body removed by extraction.' if key=='common_voice_es_19741241' else 'The parenthesized words are present in raw/clean output and removed by the configured normalizer.'))
A['per_dataset']['openml_bad']['specific_reviewed_cases']=specific

empty={}
for ds,d in A['per_dataset'].items():
    empty[ds]={}
    for role in ['baseline','target']:
        selected=[r for r in ROWS[role][ds].values() if r['gts'].strip() and not r['clean_output'].strip()]
        header_only=[r for r in ROWS[role][ds].values() if r['gts'].strip() and r['output'].strip() in {
            '<src='+r['language'],'<src='+r['language']+'.','<src='+r['language']+'>'}]
        empty[ds][role]={'rows':len(selected),'raw_header_only_rows':len(header_only),
            'raw_header_only_ids':[r['id'] for r in header_only],
            'all_selected_have_nonempty_reference':all(r['gts'].strip() for r in selected),
            'cases':[{'id':r['id'],'line':r['_line'],'raw':r['output'],'reference':r['gts'],'n_err':r['n_err'],'n_ref':r['n_ref'],
                      'parent_audio_path':r['parent_audio_path'],'seg_index':r['seg_index'],'seg_start':r['seg_start'],'seg_end':r['seg_end']} for r in selected]}
        fd=d['format_diagnostics'][role]
        fd['header_examples']=[{'id':r['id'],'line':r['_line'],'language_metadata':r['language'],'raw_excerpt':r['output'][:200],'clean_excerpt':r['clean_output'][:200]} for r in list(ROWS[role][ds].values())[:3]]
        for case in fd['glued_transcription_token_removed']:
            row=ROWS[role][ds][case['id']]
            if ds=='openml_bad':
                repaired=case['consumed_transcription_suffix']+' '+row['clean_output']
                case['diagnostic_prefix_preserved']=rescore(row,repaired)
                case['interpretation']='Exact deployed fallback cleanup removes the first whitespace-delimited raw token, including this visible transcription suffix. Prefix-preserved rescore is diagnostic only.'
A['empty_clean_output_diagnostics']=empty
write_json(ROOT/'raw_body_diagnostics.json',{
    'step':580,'scope':'All exact-step full validation rows, fixed stable-ID cohort. Earnings rows are chunks; gts is the nonempty full-parent reference, not an available per-chunk reference.',
    'empty_definition':'nonempty raw gts and empty clean_output after exact deployed extraction',
    'raw_header_only_definition':'nonempty gts and stripped raw exactly one of <src=LANGUAGE, <src=LANGUAGE. or <src=LANGUAGE>, using each row exact language metadata. This deliberately excludes tokens with possible glued transcription.',
    'per_dataset':empty})
earn_diagnostic={}
for role in ['baseline','target']:
    rows=list(ROWS[role]['earning_chunk_verb'].values())
    chars=[r for r in rows if '(' in r['clean_output'] or ')' in r['clean_output']]
    assert not chars
    earn_diagnostic[role]={'raw_rows':len(rows),'ascii_parenthesis_character_rows':0,
        'diagnostic_input_identical_to_official':True,'diagnostic_counts':A['per_dataset']['earning_chunk_verb']['totals'][role]}
A['per_dataset']['earning_chunk_verb']['ascii_parenthesis_sensitivity']={
    'method':'Remove literal ASCII parentheses delimiters while retaining content. Neither side contains either delimiter in any clean_output, so all diagnostic input strings are byte-identical to official scoring inputs.',
    'runs':earn_diagnostic,'official_and_diagnostic_delta_n_err':112,
    'interpretation':'The Earnings deletion increase cannot be explained by this ASCII-parenthesis stripping convention.'}
earn_review=[]
for line in [111,124,125,126,127]:
    pair={}
    for role in ['baseline','target']:
        row=next(r for r in ROWS[role]['earning_chunk_verb'].values() if r['_line']==line)
        pair[role]={'id':row['id'],'source_line':line,'seg_index':row['seg_index'],'parent_audio_path':row['parent_audio_path'],
                    'raw':row['output'],'clean_output':row['clean_output']}
    earn_review.append(pair)
A['per_dataset']['earning_chunk_verb']['reviewed_raw_body_cases']=earn_review
A['latest_common_step_evidence']=str(ROOT/'latest_common_step_evidence.json')
A['report_path']=str(ROOT/'report.md')
A['figure_path']=str(ROOT/'step580_error_decomposition.png')
A['source_scope_note']='Two deployed scorer snapshots are byte-identical. Current asr_response differs only in a later 2607 branch; all extraction replay here explicitly loads the frozen deployed parser. Imported scoring normalizers match deployed snapshots byte-for-byte.'
write_json(ROOT/'analysis.json',A)

names={'earning_chunk_verb':'Earnings','openml_bad':'OpenML bad','ls_clean':'LS clean','ls_other':'LS other'}
fig,axes=plt.subplots(2,2,figsize=(14,8.4))
fig.patch.set_facecolor('#f7f9fc')
for ax,(ds,d) in zip(axes.flat,A['per_dataset'].items()):
    values=[d['delta_sdi'][k] for k in ['n_sub','n_del','n_ins']]
    bars=ax.bar(['Substitution','Deletion','Insertion'],values,color=['#d9554d' if v>0 else '#16836d' for v in values],width=.56)
    ax.axhline(0,color='#64748b',linewidth=.8)
    ymax=max(max(values)*1.5,7);ymin=min(min(values)*1.8,-3)
    ax.set_ylim(ymin,ymax);ax.set_facecolor('white')
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+(ymax-ymin)*(.03 if v>=0 else -.04),f'{v:+d}',ha='center',va='bottom' if v>=0 else 'top',fontsize=13,fontweight='bold')
    b,t=d['totals']['baseline'],d['totals']['target']
    ax.set_title(f"{names[ds]}: {100*b['p_err']:.4f}% → {100*t['p_err']:.4f}%\nΔp_err {d['delta_p_err_pp']:+.4f} pp; Δerrors {d['delta_n_err']:+d}",loc='left',fontsize=12,pad=14)
    cc=d['category_counts']
    ax.text(.02,.96,f"Matched {d['matched']} {'parents' if ds=='earning_chunk_verb' else 'utterances'}\nNew failure {cc['newly_failed']} | Persistent worse {cc['worsening_persistent']} | Improved {cc['improved']}",transform=ax.transAxes,va='top',fontsize=9,color='#475569')
    ax.set_ylabel('R16 minus Fix error count');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Step 580 · Fix vs R16 · Full matched validation',fontsize=19,x=.05,ha='left',y=.99,fontweight='bold')
fig.text(.05,.01,'Primary logged counts reproduced exactly. Red = more errors. Earnings: 341 chunks / 6 parent scoring units. OpenML S/D/I uses minimum-edit diagnostic tie-breaking.',fontsize=9,color='#475569')
fig.tight_layout(rect=[.02,.04,.99,.94]);fig.savefig(ROOT/'step580_error_decomposition.png',dpi=170,bbox_inches='tight');plt.close(fig)

lines=['# R16 对 Fix：step 580 完整验证错误分析','',
    'R16 在本次同一步验证的四个数据集上均高于 Fix 的错误率，差距主要集中在 Earnings 与 OpenML。Earnings 增加 112 个错误，其中删除增加 90；OpenML 增加 35 个错误，其中删除增加 30。这里使用直接 Fix 基线，未使用历史 unfixed 基线。','',
    f"数据冻结时间：`{json.loads((ROOT/'source_manifest.json').read_text())['captured_at_utc']}`。rolling 数据最近公共完整验证为 **580**；R16 已有 600，但 Fix 的完整验证截至 580，不能把两者异步的最新点当作同一步比较。",'',
    '- Fix：`0ca34a537b05e74628264cac2ea86a75`，`baseline_current_after_450`，submission `raysubmit_wUz37YEmnzCbLxpH`。',
    '- R16：`5dc3fdb4c014c9dc1570a5af6e0c1e6e`，`target_current_after_570`，submission `raysubmit_UmggvqWDkrghRcrn`。',
    '- observation fingerprint：`5ff1c2c0449461fc602676edbc7c855f13a21e7bb5baddc3192ac6b3ffcaeccf`。',
    '- 8 份完整 dump 的 SHA 已冻结；所有 **5958 个评分单位**按稳定 ID/parent 配对，0 单边、0 重复、0 参考冲突、0 segment 元数据冲突。全部本地重算的 n_err/n_ref 与日志一致，8 个重建 aggregate 与对应 `val-core/<dataset>/p_err/mean@1` 在 1e-12 内一致。','',
    '![Error decomposition](step580_error_decomposition.png)','',
    '## 四个数据集分别对比','',
    '| 数据集 / 评分单位 | matched | Fix errors/ref | R16 errors/ref | Fix p_err | R16 p_err | Δ pp | Δerrors |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for ds,d in A['per_dataset'].items():
    b,t=d['totals']['baseline'],d['totals']['target']
    lines.append(f"| {ds} / {d['unit']} | {d['matched']} | {b['n_err']}/{b['n_ref']} | {t['n_err']}/{t['n_ref']} | {100*b['p_err']:.6f}% | {100*t['p_err']:.6f}% | {d['delta_p_err_pp']:+.6f} | {d['delta_n_err']:+d} |")
lines += ['',
    '所有数据集单独报告，没有跨数据集平均。p_err 是 sum(n_err)/sum(n_ref)，不是逐样本 WER 的均值。Earnings 的 341 chunks 仅构成 6 个 parent；每个 parent 只使用一次非空 group-head counts，null 不按零计。OpenML 有 3 个相同 raw gts 但 n_ref 不同的配对，总分母 5630→5633，来自 hypothesis-dependent compound merge；每侧保留自己的分母。','',
    '## 新增失败、持续恶化和改善','',
    '新增失败：Fix n_err=0、R16 n_err>0；持续恶化：两边有错且 R16 的 n_err/n_ref 更大；改善：R16 n_err/n_ref 更小；持续不变：相同非零错误率；不变正确：两边零错。比较用整数交叉乘积，避免浮点误判。另存基于原始 n_err 的分类以核对分母敏感样本。','',
    '| 数据集 | 新增失败 | 持续恶化 | 改善（其中恢复零错） | 持续不变 | 不变正确 | 新增失败增加错误 | 持续恶化增加错误 | 改善移除错误 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for ds,d in A['per_dataset'].items():
    c=d['category_counts'];bal=d['error_balance']
    lines.append(f"| {ds} | {c['newly_failed']} | {c['worsening_persistent']} | {c['improved']} ({d['recovered_subset_of_improved']}) | {c['persistent']} | {c['unchanged_correct']} | {bal['newly_failed']['delta_n_err']:+d} | {bal['worsening_persistent']['delta_n_err']:+d} | {-bal['improved']['delta_n_err']} |")
lines += ['',
    '| 数据集 | Fix S/D/I | R16 S/D/I | ΔS / ΔD / ΔI |',
    '|---|---:|---:|---:|']
for ds,d in A['per_dataset'].items():
    vals=lambda x:'/'.join(str(x[k]) for k in ['n_sub','n_del','n_ins'])
    lines.append(f"| {ds} | {vals(d['totals']['baseline'])} | {vals(d['totals']['target'])} | {' / '.join(f'{d['delta_sdi'][k]:+d}' for k in ['n_sub','n_del','n_ins'])} |")
lines += ['',
    'English 使用部署源码的 `measure_openasr_en_wer` 和 compound-aware kaldialign；OpenML 使用部署源码的语言 normalizer/compound merge，再以 jiwer 得到最小编辑分解。OpenML 等价最小路径可能改变 S/D/I 分配，但本次每条的 S+D+I 与 n_ref 均精确重现日志。S/D/I 是文本对齐结果，不能直接证明声学原因。','',
    '## Earnings 六个 parent 的贡献','',
    '共享 reference denominator 为 15395。以下 contribution pp 使用每个 parent 的 Δerrors / 15395；可相加得到总体 +0.727509 pp。','',
    '| Parent 文件 | chunks | Fix err/ref | R16 err/ref | Δerrors | ΔS/ΔD/ΔI | 对整体 Δpp 的贡献 | 占净增错误 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
parents=A['per_dataset']['earning_chunk_verb']['parent_contributions']
for c in parents:
    b,t=c['baseline'],c['target'];sd=c['delta_sdi']
    lines.append(f"| {c['parent_filename']} | {c['segment_count']} | {b['n_err']}/{b['n_ref']} | {t['n_err']}/{t['n_ref']} | {c['delta_n_err']:+d} | {sd['n_sub']:+d}/{sd['n_del']:+d}/{sd['n_ins']:+d} | {c['contribution_pp']:+.6f} | {100*c['share_of_net_added_errors']:.2f}% |")
lines += ['',
    '前两个 parent 合计增加 79 个错误，占 Earnings 净增 112 的 70.54%；二者的删除分别增加 46、40。六个 parent 中 5 个恶化、1 个错误总数不变，没有改善 parent。','']
lines += [
    'Earnings 两边所有 341 行 clean_output 都没有 ASCII `(` 或 `)`；保留括号正文诊断的输入与官方输入完全相同，仍为 Fix 685/15395、R16 797/15395。这里的大幅删除不能用该括号 normalizer 规则解释。','',
    '已直接核查主要删除的 raw 正文：c6ef 的 R16 line125/seg11 与 line126/seg12 都只输出 `<src=English.`，clean 为空；对应 Fix 分别包含 “Slower than expected.” 与完整 COVID-19/precautionary measures 句子。647 的 line111/seg59，Fix 有 “In head grade in 2023 ... a stronger performance next year. That is clear ...”；R16 raw 从 `<src=English.\\nThat is clear ...` 开始，前半段在原始输出中也没有出现。这些具体位置是未生成正文/部分正文，不能解释成括号归一化删词；是否为声学原因仍需音频或解码证据。','']
for c in parents:
    e=c['excerpt']
    lines += [f"### {c['parent_filename']}",'',f"完整 parent errors：`{c['baseline']['n_err']}/{c['baseline']['n_ref']} → {c['target']['n_err']}/{c['target']['n_ref']}`；以下为 **{e['scope']}**，不是 chunk 对完整参考评分。",'',
        '- Ref：'+json.dumps(e['reference'],ensure_ascii=False),'- Fix：'+json.dumps(e['fix'],ensure_ascii=False),'- R16：'+json.dumps(e['r16'],ensure_ascii=False),'',
        '出处：'+json.dumps(e.get('source_excerpt_lines',{}),ensure_ascii=False)+'；parent 的完整行号和身份见 analysis.json。','']

lines += ['## Header 格式与实际文字内容','',
    'R16 四个数据集全部 **0 strict2609 header**。这里的严格 header 指合法相邻 `<src=...><tgt=...>` 加换行；缺失 tgt 或缺 `>` 均无法被该 parser 的 2609 分支识别。它说明输出协议退化，不能证明转写正文用了错误语言。','',
    '| 数据集 | Fix strict / raw rows | R16 strict / raw rows | R16 closed src only | R16 src 缺闭合角括号 | R16 其他 |',
    '|---|---:|---:|---:|---:|---:|']
for ds,d in A['per_dataset'].items():
    b,t=(d['format_diagnostics'][r] for r in ['baseline','target']);p=t['first_header_patterns']
    other=t['raw_rows']-p.get('src_only_closed_no_tgt',0)-p.get('src_header_without_closing_angle',0)
    lines.append(f"| {ds} | {b['strict_2609_header_rows']}/{b['raw_rows']} | {t['strict_2609_header_rows']}/{t['raw_rows']} | {p.get('src_only_closed_no_tgt',0)} | {p.get('src_header_without_closing_angle',0)} | {other} |")
lines += ['',
    '所有 raw→clean 均以两次部署 commit SHA 一致的 parser 精确重放：每侧 6293/6293 行一致。大部分非标准头后的正文被 fallback 保留下来；因此不能把 6293 个 header 失败计为 6293 个识别失败。高精度“正文首词粘在 malformed src token 上，被 fallback 清理一起吞掉”的机械 signature 仅 R16 OpenML 2 行，其余数据集为 0；这只是特定 signature 的下界。','']
lines += ['| 数据集 | Fix empty clean | R16 empty clean | Fix raw header-only | R16 raw header-only | 非空参考 |',
          '|---|---:|---:|---:|---:|---|']
for ds,d in empty.items():
    lines.append(f"| {ds} | {d['baseline']['rows']} | {d['target']['rows']} | {d['baseline']['raw_header_only_rows']} | {d['target']['raw_header_only_rows']} | 全部有；Earnings 为完整 parent ref |")
lines += ['',
    'raw header-only 采用精确元数据规则：raw 去空白后只能等于 `<src=LANGUAGE`、`<src=LANGUAGE.` 或 `<src=LANGUAGE>`，排除可能与正文黏连的其他 token。Earnings 的 3 个 empty clean chunk 全部 raw 也只有 header；OpenML 的 1 个亦然。完整明细另存 `raw_body_diagnostics.json`。','']
for c in empty['earning_chunk_verb']['target']['cases']:
    lines.append(f"- Earnings ID `{c['id']}`；parent `{Path(c['parent_audio_path']).name}`；line {c['line']}；seg_index {c['seg_index']}；音频区间 {c['seg_start']}–{c['seg_end']} s；raw `{c['raw']}`，clean 空；非空完整 parent reference。")
lines += ['']
for ds,d in A['per_dataset'].items():
    for c in d['format_diagnostics']['target']['glued_transcription_token_removed']:
        lines += [f"- {ds} ID `{c['id']}`，line {c['source_line']}：删除的首 token `{c['removed_token']}`，其中正文后缀 `{c['consumed_transcription_suffix']}`。官方 {c['n_err']}/{c['n_ref']}；保留该正文后缀的诊断 rescore {c['diagnostic_prefix_preserved']['n_err']}/{c['diagnostic_prefix_preserved']['n_ref']}。",'  - raw：'+json.dumps(c['raw_excerpt'],ensure_ascii=False),'  - clean：'+json.dumps(c['clean_excerpt'],ensure_ascii=False),'']
lines += ['## OpenML 括号 normalizer 的敏感性','',
    '两模型的 reference 与 hypothesis 采用同一操作：去掉 ASCII `(` 和 `)`，保留括号内所有词，再使用原语言 normalizer/scorer。已核查两侧各 393 个 raw gts，含 `(` 或 `)` 的条数均为 **0**，因此 reference 变换后逐字节不变，现有重算与两侧 reference+hypothesis 同操作严格等价。它是评分敏感性诊断，不覆盖官方验证。','',
    '| 范围 | 单位数 | 官方 Δerrors | 保留括号正文后的诊断 Δerrors | 两者差值 |',
    '|---|---:|---:|---:|---:|']
for status,g in groups.items():
    lines.append(f"| {status} | {g['units']} | {g['official_delta_n_err']:+d} | {g['diagnostic_delta_n_err']:+d} | {g['differential_parenthesis_error_effect']:+d} |")
lines += ['',
    '全 OpenML：Fix 官方 461/5630 → 诊断 454/5630；R16 官方 496/5633 → 诊断 479/5632。净增错误从官方 +35 变为诊断 +25，两侧括号评分的差异对应 10 个净错误。官方 p_err 差为 +0.616978 pp，配对诊断差为 '+f"{100*(479/5632-454/5630):+.6f}"+' pp。分母也略变，不能直接把“10/某个固定分母”当作唯一因果贡献。','']
for c in specific:
    lines += [f"### OpenML ID {c['stable_id']}",'',f"language metadata `{c['language']}`；Fix/R16 dump line `{c['baseline_lines'][0]}/{c['target_lines'][0]}`；官方 errors `{c['baseline']['n_err']}/{c['baseline']['n_ref']} → {c['target']['n_err']}/{c['target']['n_ref']}`。",'',
        '- Ref：'+json.dumps(c['reference'],ensure_ascii=False),'- Fix clean：'+json.dumps(c['fix_clean'],ensure_ascii=False),'- R16 clean：'+json.dumps(c['r16_clean'],ensure_ascii=False),
        '- R16 raw：'+json.dumps(c['r16_raw'],ensure_ascii=False),'- R16 normalized：'+json.dumps(c['r16_normalized']['normalized_hypothesis'],ensure_ascii=False),'',c['interpretation'],'']
lines += ['## 其他完整验证样例','']
for ds in ['ls_clean','ls_other']:
    lines += [f'### {ds}','']
    for c in A['per_dataset'][ds]['examples']:
        if c['category']=='persistent':continue
        e=c['excerpt'];b,t=c['baseline'],c['target']
        lines += [f"- **{c['category']}**：ID `{c['stable_id']}`；Fix/R16 line `{c['baseline_lines']}/{c['target_lines']}`；完整 errors `{b['n_err']}/{b['n_ref']} → {t['n_err']}/{t['n_ref']}`；S/D/I `{b['n_sub']}/{b['n_del']}/{b['n_ins']} → {t['n_sub']}/{t['n_del']}/{t['n_ins']}`。",'  - Ref：'+json.dumps(e['reference'],ensure_ascii=False),'  - Fix：'+json.dumps(e['fix'],ensure_ascii=False),'  - R16：'+json.dumps(e['r16'],ensure_ascii=False),'']
lines += ['## 解释边界与可复核产物','',
    '这份结果说明 step580 时 R16 的完整验证转写与协议表现；同一步并不等于同 fresh-data exposure。R16 当前 replay16/fresh112/buffer400，首 50 次继承更新采用过 replay12/fresh116 和 ReplayLossNormalizer；Fix fresh128。恢复会重置 replay buffer，输入重建也未完整固定随机状态。因此这组观察不能单独确证 replay 的因果效果，也不是干净的 replay-only 单因素消融。','',
    '本分析不改变任何训练状态或原始来源。`analysis.json` 含总体、全部 parent、header、括号诊断及 reviewed cases；`matched_units_step580.jsonl` 含 5958 个逐单位的配对、S/D/I、分类、行号与参考哈希；`source_manifest.json` 与 `source_snapshot/` 提供文件 SHA 和冻结来源；`scorer_snapshot/` 保留两次实际部署的 scorer/normalizer；`latest_common_step_evidence.json` 固定为何选择 580。','']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'parenthesis_groups':groups,'empty_output_counts':{ds:{r:v['rows'] for r,v in roles.items()} for ds,roles in empty.items()},'report':str(ROOT/'report.md'),'analysis':str(ROOT/'analysis.json'),'figure':str(ROOT/'step580_error_decomposition.png')},ensure_ascii=False,indent=2))
