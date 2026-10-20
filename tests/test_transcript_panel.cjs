const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function element(tagName = 'div') {
  return {tagName, dataset: {}, children: [], textContent: '',
    append(...items) { this.children.push(...items); },
    replaceChildren() { this.children = []; }};
}
const context = vm.createContext({
  document: {
    createElement: element,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    }
  },
  setInterval() {},
  fetch: async url => ({ok: true, json: async () => url.endsWith('/feed') ? {segments: [], claims: []} : {
    running: false, can_clear: true, configured: {}, local_llm: {}, brave_quota: {}
  }})
});
vm.runInContext(fs.readFileSync('session-panel.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('subject-panel.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('live.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1], context);
context.feed = {
  segments: [{id: 1, offset_s: 12, text: 'Original'}, {id: 2, offset_s: 24, text: 'Legado'}],
  paragraphs: [{id: 5, at: '2026-10-20T12:00:00Z', start_s: 0, offset_s: 12,
    text: 'Texto revisado.', finalized: 1, originals: [{id: 1, offset_s: 12, text: '<script>fala</script>'}]}]
};
const items = vm.runInContext('transcriptItems(feed)', context);
assert.equal(items.length, 2, 'Grouped originals must not appear twice; legacy transcripts remain visible');
assert.equal(items[0].text, 'Legado');
let article = vm.runInContext('paragraph(feed.paragraphs[0])', context);
assert.match(article.children[0].textContent, /0–12s/);
assert.equal(article.children[1].textContent, 'Texto revisado.');
let details = article.children.find(node => node.tagName === 'details');
assert.equal(details.children[0].textContent, 'Ver transcrição original');
assert.match(details.children[1].textContent, /<script>fala<\/script>/);
details.open = true;
details.ontoggle();
article = vm.runInContext('paragraph(feed.paragraphs[0])', context);
details = article.children.find(node => node.tagName === 'details');
assert.equal(details.open, true, 'Original transcript must stay expanded across polling updates');
console.log('Transcript panel: revised paragraphs, originals, timestamps, legacy data and polling state passed.');
