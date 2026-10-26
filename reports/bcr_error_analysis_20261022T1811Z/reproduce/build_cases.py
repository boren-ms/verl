from pathlib import Path
import json, shutil, hashlib, base64, html, difflib, collections

ROOT = Path('/home/boren/code/verl')
WORK = ROOT / 'tmp/experiment_comparisons/20261022T1811Z_bcr_error_analysis'
OUT = ROOT / 'reports/bcr_error_analysis_20261022T1811Z'
CONFIGS = [
    ('R16', 580, ROOT / 'tmp/experiment_comparisons/20261022T073852Z_bcr_rolling/examples/step_580__5ff1c2c0449461fc/matched_validation_cases.jsonl'),
    ('R64', 440, WORK / 'r64/matched_validation_cases.jsonl'),
]
cache, sources, records, summaries = {}, [], [], {}

def read_dump(path, run, role):
    path = Path(path)
    if path not in cache:
        b = path.read_bytes()
        target = OUT / 'sources/dumps' / run / role / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b)
        sources.append({'original_path': str(path), 'frozen_path': str(target.relative_to(OUT)), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)})
        cache[path] = [json.loads(l) for l in b.decode().splitlines() if l.strip()]
    return cache[path]

def text(row):
    value = row.get('clean_output')
    return value if isinstance(value, str) else ''

def excerpt(a, b):
    # Display-only string comparison; all scores come from official logged counts.
    aa, bb = a.split(), b.split()
    op = next((x for x in difflib.SequenceMatcher(None, [x.lower() for x in aa], [x.lower() for x in bb], autojunk=False).get_opcodes() if x[0] != 'equal'), None)
    if op is None: return a[:350], b[:350]
    _, i, j, k, l = op
    return ' '.join(aa[max(0, i-10):min(len(aa), max(j, i+1)+24)]), ' '.join(bb[max(0, k-10):min(len(bb), max(l, k+1)+24)])

for run, step, pairfile in CONFIGS:
    summary_path = (ROOT / 'tmp/experiment_comparisons/20261022T073852Z_bcr_rolling/examples/step_580__5ff1c2c0449461fc/examples_summary_snapshot.json') if run == 'R16' else WORK / 'r64/examples_evidence.json'
    summary = json.loads(summary_path.read_text())
    summaries[run] = summary
    (OUT / f'{run.lower()}_examples_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with pairfile.open() as f:
        for line in f:
            c = json.loads(line)
            a = read_dump(c['baseline_local_path'], run, 'Fix')
            b = read_dump(c['target_local_path'], run, run)
            aa = [a[i-1] for i in c['baseline_lines']]
            bb = [b[i-1] for i in c['target_lines']]
            aa.sort(key=lambda r: r.get('seg_index') if r.get('seg_index') is not None else 0)
            bb.sort(key=lambda r: r.get('seg_index') if r.get('seg_index') is not None else 0)
            ref = aa[0]['gts']
            ref = ref if isinstance(ref, str) else json.dumps(ref, ensure_ascii=False)
            ta, tb = ' '.join(text(x) for x in aa).strip(), ' '.join(text(x) for x in bb).strip()
            ea, eb = excerpt(ta, tb)
            records.append({'run': run, 'step': step, 'dataset': c['dataset'], 'id': c['stable_identity'], 'category': c['category'],
                            'baseline_n_err': c['baseline_n_err'], 'target_n_err': c['target_n_err'], 'baseline_n_ref': c['baseline_n_ref'], 'target_n_ref': c['target_n_ref'], 'delta_n_err': c['delta_n_err'],
                            'language': aa[0].get('language'), 'reference': ref, 'baseline': ta, 'target': tb,
                            'baseline_excerpt': ea, 'target_excerpt': eb,
                            'baseline_raw': '\n'.join(x['output'] for x in aa), 'target_raw': '\n'.join(x['output'] for x in bb),
                            'baseline_lines': c['baseline_lines'], 'target_lines': c['target_lines'], 'segment_count': c.get('segment_count', 1),
                            'baseline_source': f"sources/dumps/{run}/Fix/{Path(c['baseline_local_path']).name}",
                            'target_source': f"sources/dumps/{run}/{run}/{Path(c['target_local_path']).name}"})

records.sort(key=lambda r: (r['run'], -r['delta_n_err'], r['dataset'], r['id']))
(OUT / 'case_data.json').write_text(json.dumps(records, ensure_ascii=False, separators=(',', ':')))
(OUT / 'sources/dump_manifest.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2))
scalar = json.loads((OUT / 'scalar_analysis.json').read_text())
rows = []
for ds in ['earning_chunk_verb', 'openml_bad', 'ls_clean', 'ls_other']:
    cells = [f'<td>{ds}</td>']
    for run in ['R16', 'R64']:
        r = next(x for x in scalar['pairwise_latest'] if x['dataset'] == ds and x['comparison'] == run + '_vs_Fix')
        count = summaries[run]['per_dataset'][ds]['delta_n_err']
        cells += [f"<td>{100*r['fix']:.4f}% → {100*r['target']:.4f}%</td>", f"<td class={'bad' if r['delta_pp']>0 else 'good'}>{r['delta_pp']:+.4f} pp / {count:+d} errors</td>"]
    rows.append('<tr>' + ''.join(cells) + '</tr>')

def picture(name):
    b = base64.b64encode((OUT / name).read_bytes()).decode()
    return f'<img class="chart" alt="{name}" src="data:image/png;base64,{b}">'

payload = json.dumps(records, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BCR 结果与配对错误分析</title>
<style>
:root{color-scheme:light;--ink:#20324a;--muted:#617084;--line:#dfe6ec;--bg:#f3f6f9;--red:#b23a22;--green:#087569}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,"Microsoft YaHei",sans-serif;font-size:16px;line-height:1.65}main{max-width:1300px;margin:auto;padding:30px 28px 70px}h1{font-size:34px;line-height:1.25;margin:6px 0 18px}h2{font-size:23px;margin:30px 0 14px}p{margin:10px 0}.eyebrow{letter-spacing:.1em;color:#138477;font-weight:700;font-size:13px}.muted,small{color:var(--muted)}.panel,.case{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin:16px 0}.lead{border-left:5px solid #138477}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.grid .panel{margin:0}.big{font-size:26px;font-weight:750}.chart{max-width:100%;height:auto;display:block;background:white;border-radius:10px;margin:15px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:12px 10px}th{background:#eef3f7}td:first-child{font-weight:600}.tablewrap{overflow:auto}.bad{color:var(--red)}.good{color:var(--green)}a{color:#12669a}.controls{display:flex;gap:10px;flex-wrap:wrap;position:sticky;top:0;background:#f3f6f9f5;padding:12px 0;z-index:2}select,input,button{font:inherit;padding:9px 12px;border:1px solid #cbd6df;border-radius:8px;background:white;color:var(--ink)}input{flex:1;min-width:250px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.case{padding:18px 22px}.case h3{font-size:16px;margin:0 0 8px;overflow-wrap:anywhere}.label{display:inline-block;border-radius:5px;background:#edf2f7;padding:2px 7px;font-size:12px;margin-right:6px}.textrow{display:grid;grid-template-columns:100px 1fr;gap:10px;margin:8px 0}.txt{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace;font-size:14px;line-height:1.55}details{margin-top:12px}summary{cursor:pointer;color:#12669a}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f5f7f9;padding:12px;border-radius:7px}code{background:#eef2f5;padding:1px 4px;border-radius:4px}.pager{display:flex;gap:12px;align-items:center}.notice{font-size:14px;padding:12px 16px;background:#fff7e8;border-radius:8px}footer{margin-top:36px;font-size:13px;color:var(--muted)}@media(max-width:780px){main{padding:22px 14px}.grid{grid-template-columns:1fr}h1{font-size:27px}.textrow{grid-template-columns:1fr;gap:2px}.panel{padding:16px}table{min-width:800px}}
</style><main>
<div class="eyebrow">VERL / REMAX / INPUT REPLAY</div><h1>BCR 结果与配对错误分析</h1>
<p class="muted">2026-10-22 · 固定分析切片：R16/Fix 截至 10:36 PDT；R64 截至 10:49 PDT。完整文本在本轮重新取证。直接基线为 remax_dev_base0_v1h2a_fix。</p>
<div class="panel lead"><p><strong>R16 有持续退化；R64 呈现数据集取舍和近期收益回吐。</strong>R16 在 step580 四个验证集都差于 Fix；R64 在 step440 的 Earnings、OpenML、LS clean 略差，LS other 更好。R64 的 OpenML 在 step80–440 的 19 次共同检查中，16 次更好、2 次更差、1 次相等。</p><p>错误文本显示 R16 主要增加漏词，并伴随广泛的输出头格式退化。OpenML 的部分官方差距来自括号归一化约定。曝光量、困难样本筛选、训练早期代码和重启均存在差异，这些轨迹尚不能单独确立 replay 比例的因果效果。</p></div>
<div class="grid"><div class="panel"><div class="muted">每对比较覆盖</div><div class="big">5,958 / 5,958</div><small>6 Earnings parents + 393 OpenML + 2620 clean + 2939 other；全部稳定 ID 和 reference 配对。</small></div><div class="panel"><div class="muted">R16 · Earnings @580</div><div class="big bad">+112 词错误</div><small>其中删除错误净增90；5/6 parent 恶化，1/6持平。</small></div><div class="panel"><div class="muted">R64 · LS other @440</div><div class="big good">−14 词错误</div><small>2.6930% → 2.6665%；不能概括为 R64 全面更差。</small></div></div>
<h2>相同步数：官方 validation p_err</h2><p>数值越低越好；Δ 为 target − Fix，单位为百分点。每一对使用自身最新共同步数，未跨步插值。</p><div class="panel tablewrap"><table><thead><tr><th>数据集</th><th>Fix → R16 @580</th><th>R16变化</th><th>Fix → R64 @440</th><th>R64变化</th></tr></thead><tbody>__ROWS__</tbody></table></div>
__DIFF__
<details><summary>查看完整验证轨迹和早期波动</summary>__VAL____FULLDIFF__</details>
<h2>最能解释差距的配对案例</h2>
<div class="panel"><p><strong>R16 · Earnings 的连续正文缺失：</strong>c6ef… parent 的官方错误63→106。Fix在seg11、12写出“Slower than expected”及“COVID-19 resurgence … continuing to closely monitor … precautionary measures …”，R16两块raw都只有<code>&lt;src=English.</code>，随后直接恢复“To ensure …”。另一parent647…的“a stronger performance next year”整段在R16 raw中也缺失。这两个parent占净增112错误中的79个；不是括号normalizer删掉已经生成的正文。</p>
<p><strong>R16 · OpenML 的空正文：</strong><code>common_voice_es_19741241</code>的reference/Fix为“Bacterial charity work leads to population-wide resistance.”；R16只输出<code>&lt;src=Spanish.</code>，错误0→8。</p>
<p><strong>R16/R64 · OpenML 的括号评分敏感性：</strong><code>8031031497862620363</code>中Fix为“50 km, 31 millas”，BCR为“50 km (31 millas)”，官方错误1→5。括号内的词仍在输出中。对两模型和reference统一保留括号正文后，R16全OpenML净差+35→诊断+25，R64净差+7→诊断+2；官方p_err继续保留。</p>
<p><strong>R64 · LS clean 的局部退化：</strong><code>7176-92135-0037</code>中reference/Fix为“introduce a meal on the stage”，R64写成“introduce Amiel on the stage”，错误0→2。</p>
<p><strong>R64 · LS other 的改善对照：</strong><code>2609-169640-0000</code>中Fix的“see us for them”被R64改为“sea is full of them”，完整句错误7→3。它仍有其他错误，属于持续困难但改善，不能归为全句恢复正确。下方搜索可查看这些案例的完整文本和原始行号。</p></div>
<h2>为什么会出现这些结果</h2>
<div class="panel"><p><strong>同一步数的新样本曝光不一致。</strong>@440 的 Fix / R16 / R64 fresh prompt presentations 为 56,320 / 49,480 / 28,160，replay 为 0 / 6,780 / 28,032。R64 每步新输入约为基线的一半。@580 的 R16 为65,160 fresh，Fix为74,240。这是可确认的预算差异；它不能独自解释R64早期OpenML的明显收益，也不能把总presentation当成独立音频数。</p>
<p><strong>R16 出现广泛的格式漂移，而相对 reward 的语言分量在观察窗口没有奖惩。</strong>step580 的6,293个原始验证行中，R16没有一个满足完整的2609 src/tgt头格式。训练401–440窗口的sampled lang、sampled−greedy lang、lang vote均为0。parser需要完整tgt字段，未闭合的src头可令lang为0，而正文仍可被清理后评分。因此这是格式问题的证据，不代表所有正文都用了错误语言。若sample和greedy都失败，相对lang奖励为0；若sample恢复正确tgt，仍可获得正奖励，不能称为不可恢复死锁。</p>
<p><strong>困难池筛选的是低greedy word分数，不是学习价值。</strong>当前是全局保留低分输入，再用当前策略重新生成。没有source配额、TTL、cooldown或max-use，buffer不随checkpoint保存。401–440窗口R64已重放的2,560个prompt中有2,314个重新留在困难池，约90.4%。这说明困难成员在重打分后大多仍属低分集合；尚无buffer的source/ID/使用次数快照，不能据此断言脏样本垄断或同一音频被反复使用了多少次。</p>
<p><strong>相对reward提高不保证绝对转写更准。</strong>401–440精确40个共同训练步，R16 raw word均值低于Fix，但最终相对reward更高。两者分别度量绝对sampled文本质量与sampled相对greedy的差分，且各run的训练输入混合不同，不能相互替代。</p>
<p><strong>现有实验还含实现与恢复差异。</strong>R16前50步使用replay12和ReplayLossNormalizer，51步后replay16并恢复原actor loss；R64 replay64。保留模型轨迹已排除回滚分支，但buffer重置、dataloader恢复及早期实现差异仍限制因果归因。单个seed的细小LS差距也不足以建立稳定收益排序。</p></div>
__REWARD__
<p class="notice">文本案例中的n_err/n_ref以原始验证日志为准。OpenML的compound归一化使两run的有效n_ref可能不同；保留两侧分母。括号保词等反事实诊断只帮助解释scorer敏感性，不替换官方结果。完整分析与额外诊断见<a href="report.md">report.md</a>，可复算数据见<a href="evidence.json">evidence.json</a>。</p>
<h2>逐条检查：退化、改善和持续困难</h2><p>每个比较均覆盖完整验证集。Earnings按parent录音聚合，并保留原始chunk行号。默认显示退化案例；可筛选改善作为对照。展示片段按字符串差异定位，仅用于阅读，不参与打分。</p>
<div class="controls"><select id="run"><option value="R16">R16 vs Fix @580</option><option value="R64">R64 vs Fix @440</option><option value="">两个比较</option></select><select id="dataset"><option value="">所有数据集</option><option>earning_chunk_verb</option><option>openml_bad</option><option>ls_clean</option><option>ls_other</option></select><select id="category"><option value="worse">所有退化</option><option value="newly_failed">新增失败</option><option value="worsening_persistent">持续错误且恶化</option><option value="improved">改善（含恢复正确）</option><option value="persistent">持续错误且持平</option><option value="unchanged_correct">保持正确</option><option value="">全部</option></select><input id="search" placeholder="搜索 ID / reference / output / language" aria-label="搜索案例"></div>
<p id="count" class="muted"></p><div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button><a href="case_data.json" download>下载全部案例 JSON</a></div><div id="cases"></div><div class="pager"><button id="top">回到筛选器</button></div>
<footer>所有病例基于同一reference的稳定identity配对；没有按W&amp;B表slot配对。选择表不能估计全量错误比例。报告没有跨数据集平均分，也没有启动新实验。原始dump和SHA256清单保存在sources/dumps及sources/dump_manifest.json。</footer>
</main><script id="payload" type="application/json">__DATA__</script><script>
const DATA=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id), SIZE=40;
const NAMES={newly_failed:'新增失败',worsening_persistent:'持续错误且恶化',improved:'改善',persistent:'持续错误且持平',unchanged_correct:'保持正确',unknown:'未知'};
let PAGE=0,FILTERED=[];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function filter(){const run=$('run').value,ds=$('dataset').value,cat=$('category').value,q=$('search').value.toLocaleLowerCase();FILTERED=DATA.filter(r=>(!run||r.run===run)&&(!ds||r.dataset===ds)&&(!cat||(cat==='worse'?['newly_failed','worsening_persistent'].includes(r.category):r.category===cat))&&(!q||[r.id,r.reference,r.baseline,r.target,r.language].join(' ').toLocaleLowerCase().includes(q)));PAGE=0;render()}
function render(){const total=FILTERED.length,pages=Math.max(1,Math.ceil(total/SIZE));PAGE=Math.max(0,Math.min(PAGE,pages-1));$('count').textContent=`匹配 ${total.toLocaleString()} 个评分单位 / 全部 ${DATA.length.toLocaleString()} 个配对记录（两个比较）`;$('page').textContent=`${PAGE+1} / ${pages}`;$('prev').disabled=PAGE===0;$('next').disabled=PAGE>=pages-1;
$('cases').innerHTML=FILTERED.slice(PAGE*SIZE,(PAGE+1)*SIZE).map(r=>`<article class="case"><h3>${esc(r.dataset)} · ${esc(r.id)}</h3><span class="label">${esc(r.run)} @${r.step}</span><span class="label">${esc(NAMES[r.category]||r.category)}</span><span class="label">${esc(r.language)}</span><p><strong class="${r.delta_n_err>0?'bad':r.delta_n_err<0?'good':''}">错误 ${r.baseline_n_err} → ${r.target_n_err}（${r.delta_n_err>=0?'+':''}${r.delta_n_err}）</strong> · n_ref ${r.baseline_n_ref} → ${r.target_n_ref} · ${r.segment_count} chunk(s)</p><div class="textrow"><strong>Reference</strong><div class="txt">${esc(r.reference.length>550?r.reference.slice(0,550)+' … [完整参考见展开]':r.reference)}</div></div><div class="textrow"><strong>Fix 片段</strong><div class="txt">${esc(r.baseline_excerpt)}</div></div><div class="textrow"><strong>${esc(r.run)} 片段</strong><div class="txt">${esc(r.target_excerpt)}</div></div><details><summary>完整参考、正文、raw output 和证据行号</summary><p>Fix: <a href="${esc(r.baseline_source)}">${esc(r.baseline_source)}</a> · JSONL lines ${esc(r.baseline_lines.join(', '))}</p><p>${esc(r.run)}: <a href="${esc(r.target_source)}">${esc(r.target_source)}</a> · JSONL lines ${esc(r.target_lines.join(', '))}</p><strong>Reference</strong><pre>${esc(r.reference)}</pre><strong>Fix clean_output</strong><pre>${esc(r.baseline)}</pre><strong>${esc(r.run)} clean_output</strong><pre>${esc(r.target)}</pre><strong>Fix raw output</strong><pre>${esc(r.baseline_raw)}</pre><strong>${esc(r.run)} raw output</strong><pre>${esc(r.target_raw)}</pre></details></article>`).join('')||'<div class="panel">当前筛选无匹配案例。</div>'}
['run','dataset','category'].forEach(x=>$(x).addEventListener('change',filter));let timer;$('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(filter,150)});$('prev').onclick=()=>{PAGE--;render()};$('next').onclick=()=>{PAGE++;render()};$('top').onclick=()=>$('run').scrollIntoView({behavior:'smooth',block:'start'});filter();
</script></html>'''
result = template.replace('__ROWS__', ''.join(rows)).replace('__DIFF__', picture('difference_recent.png')).replace('__VAL__', picture('validation.png')).replace('__FULLDIFF__', picture('difference.png')).replace('__REWARD__', picture('reward.png')).replace('__DATA__', payload)
(OUT / 'index.html').write_text(result)
print(json.dumps({'case_count': len(records), 'run_counts': dict(collections.Counter(r['run'] for r in records)), 'html_bytes': len(result.encode()), 'dump_count': len(sources)}, ensure_ascii=False))
