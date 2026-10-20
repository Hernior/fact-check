const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const elements = new Map();
function element() {
  return {disabled: false, dataset: {}, textContent: '', append() {}, replaceChildren() {}};
}
let canClear = true;
let confirmed = false;
let confirmations = 0;
const deletions = [];
const context = vm.createContext({
  document: {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    createElement: element
  },
  setInterval() {},
  confirm(message) { confirmations++; assert.match(message, /Brave/); return confirmed; },
  fetch: async (url, options) => {
    if (url === '/api/clear') {
      assert.equal(options.method, 'POST');
      assert.deepEqual(JSON.parse(options.body), {confirmed: true});
      deletions.push(url);
      return {ok: true, json: async () => ({cleared: true})};
    }
    return {ok: true, json: async () => url.endsWith('/feed') ? {segments: [], claims: []} : {
      running: !canClear, can_clear: canClear, local_llm: {state: 'pronta'}, configured: {},
      brave_quota: {used: 1, limit: 1000, remaining: 999, window_days: 32}, segments: 0, claims: 0
    }};
  }
});
const html = fs.readFileSync('live.html', 'utf8');
vm.runInContext(fs.readFileSync('session-panel.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('subject-panel.js', 'utf8'), context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);

(async () => {
  await vm.runInContext('refresh()', context);
  const button = elements.get('clear');
  await button.onclick();
  assert.equal(confirmations, 1);
  assert.equal(deletions.length, 0, 'cancelling confirmation must not send a deletion');
  canClear = false;
  await vm.runInContext('refresh()', context);
  assert.equal(button.disabled, true);
  confirmed = true;
  await button.onclick();
  assert.equal(deletions.length, 0, 'active capture must block the button');
  canClear = true;
  await vm.runInContext('refresh()', context);
  elements.get('capture').dataset.busy = 'true';
  await button.onclick();
  assert.equal(deletions.length, 0, 'tab selection/setup must block deletion before server capture starts');
  delete elements.get('capture').dataset.busy;
  await button.onclick();
  assert.equal(deletions.length, 1);
  assert.match(elements.get('clear-message').textContent, /Brave foi preservado/);
  console.log('Clear button: cancellation, active capture block and confirmed deletion passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
