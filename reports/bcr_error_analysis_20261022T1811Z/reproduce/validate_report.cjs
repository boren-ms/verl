const fs=require('fs'),vm=require('vm'),crypto=require('crypto'),path=require('path');
const root='/home/boren/code/verl/reports/bcr_error_analysis_20261022T1811Z';
const checks=[];function ok(name,v){if(!v)throw Error(name);checks.push({name,passed:true});}
const records=JSON.parse(fs.readFileSync(path.join(root,'case_data.json'),'utf8'));
ok('Both comparisons have 5958 stable matched scoring units',records.length===11916&&['R16','R64'].every(run=>records.filter(r=>r.run===run).length===5958));
const datasets={earning_chunk_verb:6,openml_bad:393,ls_clean:2620,ls_other:2939};
const scalar=JSON.parse(fs.readFileSync(path.join(root,'scalar_analysis.json'),'utf8'));
for(const run of ['R16','R64'])for(const [ds,n]of Object.entries(datasets)){
 const rr=records.filter(r=>r.run===run&&r.dataset===ds);ok(`${run}/${ds} unique complete IDs`,rr.length===n&&new Set(rr.map(r=>r.id)).size===n);
 const count=side=>rr.reduce((a,r)=>a+r[side],0),m=scalar.pairwise_latest.find(r=>r.comparison===run+'_vs_Fix'&&r.dataset===ds);
 ok(`${run}/${ds} both official rates reproduced`,Math.abs(count('baseline_n_err')/count('baseline_n_ref')-m.fix)<1e-12&&Math.abs(count('target_n_err')/count('target_n_ref')-m.target)<1e-12);
 ok(`${run}/${ds} expected exact step`,rr.every(r=>r.step===(run==='R16'?580:440)));
}
const manifest=JSON.parse(fs.readFileSync(path.join(root,'sources/dump_manifest.json'),'utf8'));
ok('Sixteen original dumps frozen',manifest.length===16);
for(const row of manifest){const b=fs.readFileSync(path.join(root,row.frozen_path));ok('Frozen SHA '+row.frozen_path,crypto.createHash('sha256').update(b).digest('hex')===row.sha256);}
const html=fs.readFileSync(path.join(root,'index.html'),'utf8'),payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1],script=html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
const elements={};function el(id){if(!elements[id])elements[id]={value:'',textContent:'',innerHTML:'',addEventListener(){},scrollIntoView(){}};return elements[id];}
el('payload').textContent=payload;el('run').value='R16';el('category').value='worse';
const ctx=vm.createContext({document:{getElementById:el},setTimeout,clearTimeout});vm.runInContext(script,ctx,{timeout:15000});
ok('Initial R16 regression filter has185 units',vm.runInContext('FILTERED.length',ctx)===185);
ok('Initial render paginates40 cards',(el('cases').innerHTML.match(/<article class="case">/g)||[]).length===40);
el('run').value='R64';vm.runInContext('filter()',ctx);ok('R64 regression filter has139 units',vm.runInContext('FILTERED.length',ctx)===139);
el('dataset').value='ls_other';el('category').value='improved';vm.runInContext('filter()',ctx);ok('R64 ls_other85 improved units',vm.runInContext('FILTERED.length',ctx)===85);
el('search').value='8188-269290-0003';vm.runInContext('filter()',ctx);ok('Search resolves exact improved example',vm.runInContext('FILTERED.length',ctx)===1&&el('cases').innerHTML.includes('half past eight'));
el('search').value='nonexistent-identity-filter-test';vm.runInContext('filter()',ctx);ok('Empty search state renders',vm.runInContext('FILTERED.length',ctx)===0&&el('cases').innerHTML.includes('无匹配案例'));
ok('HTML has4 embedded scientific figures',(html.match(/src="data:image\/png;base64,/g)||[]).length===4);
ok('No network chart dependency',!html.includes('cdn.plot.ly')&&!html.includes('cdn.jsdelivr'));
fs.writeFileSync(path.join(root,'validation_checks.json'),JSON.stringify({status:'passed',checks,scope:'Source hashes, official aggregates and functional JS filters via Node VM DOM mocks. Not an actual browser rendering test.'},null,2));
process.stdout.write(JSON.stringify({status:'passed',checks:checks.length}));
