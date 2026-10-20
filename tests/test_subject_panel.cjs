const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
function tag(tagName, className, textContent) {
  return {tagName, className, textContent, dataset: {}, children: [], value: '',
    append(...nodes) { this.children.push(...nodes); },
    setAttribute(key, value) { this[key] = value; }, blur() {}};
}
elements.set('participants', {value: 'Lula, Flávio Bolsonaro; Outra pessoa'});
let posts = [], release;
const context = vm.createContext({tag, $: id => elements.get(id), clearing: false,
  refresh() {},
  api(path, options) {
    posts.push({path, body: JSON.parse(options.body)});
    return new Promise(resolve => { release = resolve; });
  }
});
vm.runInContext(fs.readFileSync('subject-panel.js', 'utf8'), context);
assert.deepEqual(Array.from(vm.runInContext('captureParticipants()', context)), ['Lula', 'Flávio Bolsonaro', 'Outra pessoa']);
context.claim = {id: 1, subject_resolution: {original_text: 'Você recebeu 134 milhões.', reference: 'Você',
  participants: ['Lula', 'Flávio Bolsonaro'], context: 'Contexto original.'}};
context.root = tag('article');
assert.equal(vm.runInContext('renderSubject(claim, root)', context), true);
const names = context.root.children.filter(node => node.tagName === 'button');
assert.deepEqual(names.map(node => node.textContent), ['Lula', 'Flávio Bolsonaro']);
const input = context.root.children.find(node => node.tagName === 'form').children[0];
input.value = 'Pessoa digitada'; input.oninput();
context.root = tag('article');
vm.runInContext('renderSubject(claim, root)', context);
assert.equal(context.root.children.find(node => node.tagName === 'form').children[0].value, 'Pessoa digitada');
(async () => {
  const selecting = names[1].onclick();
  await names[1].onclick();
  assert.equal(posts.length, 1, 'Repeated clicks must not submit twice');
  assert.equal(posts[0].path, 'claims/1/subject');
  assert.equal(posts[0].body.subject, 'Flávio Bolsonaro');
  assert.ok(!posts.some(p => /search|stop/.test(p.path)), 'Subject selection must neither stop capture nor search');
  release({resolved: true}); await selecting;
  console.log('Subject panel: participant choices, free text, preserved drafts and isolated confirmation passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
