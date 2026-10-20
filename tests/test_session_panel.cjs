const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = new Map();
const element = (name, cl, text) => ({textContent: text, children: [], hidden: false, value: '',
  append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; }});
let refreshes = 0;
const context = vm.createContext({tag: element, refresh() { refreshes++; }, $: id => {
  if (!elements.has(id)) elements.set(id, element());
  return elements.get(id);
}});
vm.runInContext(fs.readFileSync('session-panel.js', 'utf8'), context);
context.status = {running: true, progress: {received: 10, received_seconds: 120, transcribed_seconds: 60,
  analyzed_seconds: 24, pending: 8, delay_seconds: 96, failed: 0}};
vm.runInContext('renderProgress(status)', context);
assert.equal(elements.get('progress-transcription').value, 50);
assert.equal(elements.get('progress-analysis').value, 20);
assert.match(elements.get('progress-message').textContent, /8 bloco/);
assert.equal(elements.get('progress-spinner').hidden, false);
context.status.running = false; context.status.stopping = true;
vm.runInContext('renderProgress(status)', context);
assert.match(elements.get('progress-message').textContent, /concluindo a fila/);
context.sessions = [{id: 'new', archived: 0, started: '2026-10-20T10:00:00Z'},
  {id: 'old', archived: 1, started: '2026-10-20T09:00:00Z', segments: 9, claims: 2}];
vm.runInContext('renderSessions(sessions)', context);
const select = elements.get('session-history');
assert.equal(select.children.length, 2);
select.value = 'old'; select.onchange();
assert.equal(vm.runInContext('feedPath()', context), 'feed?session_id=old');
vm.runInContext('renderSessions(sessions)', context);
assert.match(elements.get('session-view-label').textContent, /arquivada/);
vm.runInContext('resetSessionView()', context);
assert.equal(vm.runInContext('feedPath()', context), 'feed');
assert.equal(refreshes, 1);
console.log('Session panel: stage percentages, backlog, stop feedback, archive navigation and reset passed.');
