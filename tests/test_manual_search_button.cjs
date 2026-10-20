const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element(tagName = 'div') {
  return {tagName, disabled: false, dataset: {}, textContent: '', children: [],
    append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; }};
}
let keyAvailable = true, blocked = false, fail = false, posts = 0, analyses = 0, fullAnalyses = 0, finish;
const record = {id: 1, text: 'Afirmação factual.', status: 'aguardando pesquisa',
  explanation: '', offset_s: 12, sources: [], evidence_count: 0, searched_at: null};
const context = vm.createContext({
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    createElement: element
  },
  setInterval() {}, confirm() { return false; },
  fetch: async (url, options) => {
    if (url === '/api/claims/1/analyze-full') {
      fullAnalyses++;
      record.full_assessment = {verdict: 'inconclusivo', explanation: 'Faltou contexto.', complete: true,
        assessed_at: '2026-10-20T18:02:00Z', citations: []};
      record.full_documents = [{source_id: 0, title: 'Artigo completo', url: 'https://example.org/article',
        status: 'ok', published_date: '2025-06-12', fetched_at: '2026-10-20T18:02:00Z'}];
      return {ok: true, json: async () => ({analyzed: true})};
    }
    if (url === '/api/claims/1/analyze') {
      analyses++;
      record.assessment = {verdict: 'fato', explanation: 'Sugestão preliminar.',
        assessed_at: '2026-10-20T18:01:00Z', citations: [{source_id: 0, quote: 'Trecho literal'}]};
      return {ok: true, json: async () => ({analyzed: true})};
    }
    if (url === '/api/claims/1/search') {
      posts++;
      assert.equal(options.method, 'POST');
      if (fail) return {ok: false, json: async () => ({detail: 'Falha Brave. Tentativa contabilizada.'})};
      await new Promise(resolve => { finish = resolve; });
      record.searched_at = '2026-10-20T18:00:00Z';
      record.status = 'inconclusivo';
      record.sources = [{title: 'Fonte', url: 'https://example.org/source', snippet: '<b>Trecho literal</b>'}];
      record.evidence_count = 1;
      return {ok: true, json: async () => ({searched: true})};
    }
    return {ok: true, json: async () => url.endsWith('/feed') ? {segments: [], claims: [record]} : {
      running: false, can_clear: true, configured: {search: keyAvailable}, local_llm: {state: 'pronta', ready: true},
      brave_quota: {used: posts, limit: 1000, remaining: 1000 - posts, blocked, window_days: 32}
    }};
  }
});
const html = fs.readFileSync('live.html', 'utf8');
vm.runInContext(fs.readFileSync('session-panel.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('subject-panel.js', 'utf8'), context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
const refresh = () => vm.runInContext('refresh()', context);
const article = () => elements.get('claims').children[0];
const button = () => article().children.find(node => node.tagName === 'button');

(async () => {
  await refresh(); await refresh();
  assert.equal(posts, 0, 'polling must never start searches automatically');
  const initial = button();
  const searching = initial.onclick();
  await initial.onclick();
  assert.equal(posts, 1, 'double click must not send a second search');
  await refresh();
  assert.equal(button().disabled, true);
  assert.match(button().textContent, /Pesquisando/);
  finish();
  await searching;
  assert.equal(button().disabled, true);
  assert.match(button().textContent, /concluída/);
  const source = article().children.find(node => node.tagName === 'ul').children[0];
  assert.equal(source.children[0].rel, 'noopener noreferrer');
  assert.equal(source.children[1].textContent, '<b>Trecho literal</b>', 'snippets must be text, not injected HTML');
  const analysisButton = article().children.find(node => node.tagName === 'button' && node.textContent === 'Analisar com IA');
  assert.ok(analysisButton && !analysisButton.disabled);
  await analysisButton.onclick();
  assert.equal(analyses, 1);
  assert.equal(posts, 1, 'local analysis must not start a new Brave search');
  const fullButton = article().children.find(node => node.tagName === 'button' && node.textContent === 'Analisar fontes completas');
  assert.ok(fullButton && !fullButton.disabled);
  await fullButton.onclick();
  assert.equal(fullAnalyses, 1);
  assert.equal(posts, 1, 'full-source analysis must not make a Brave search');
  const fullBadge = vm.runInContext(`assessmentBadge(${JSON.stringify(record)})`, context);
  assert.equal(fullBadge.className, 'pill inconclusivo', 'full-source result must override snippet suggestion');
  assert.ok(article().children.some(node => node.tagName === 'button' && node.textContent === 'Fontes analisadas' && node.disabled));
  for (const [verdict, icon, css] of [['fato', '✓', 'fato'], ['fake', '⚠', 'fake'], ['inconclusivo', '?', 'inconclusivo']]) {
    const badge = vm.runInContext(`assessmentBadge({assessment:{verdict:'${verdict}'}})`, context);
    assert.ok(badge.textContent.includes(icon));
    assert.ok(badge.textContent.includes('Sugestão'));
    assert.equal(badge.className, 'pill ' + css);
  }
  assert.ok(html.includes('ausência de evidências não comprova falsidade'));
  assert.match(html, /\.fato\{background:#d9f7e8/);
  assert.match(html, /\.fake\{background:#ffe1e5/);
  record.searched_at = null;
  record.assessment = null;
  record.full_assessment = null;
  record.full_documents = [];
  blocked = true;
  await refresh();
  assert.equal(button().disabled, true);
  await button().onclick();
  assert.equal(posts, 1);
  blocked = false; keyAvailable = false;
  await refresh();
  assert.equal(button().disabled, true);
  keyAvailable = true; fail = true;
  await refresh();
  await button().onclick();
  assert.equal(posts, 2);
  assert.equal(button().disabled, false, 'failure must allow another explicit attempt');
  assert.ok(article().children.some(node => node.textContent.includes('Tentativa contabilizada')));
  console.log('Manual search UI: explicit click, duplicate prevention, quota/key blocks, snippets and retry passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
