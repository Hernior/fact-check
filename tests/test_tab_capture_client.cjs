const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
const uploads = [], stopped = [], handlers = {};
let node, releaseFirst;
const track = {stop() {}, getSettings() { return {displaySurface: 'browser'}; }};
const stream = {getTracks: () => [track], getVideoTracks: () => [track], getAudioTracks: () => [track]};
const context = vm.createContext({
  document: {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {dataset: {}, addEventListener(event, handler) { handlers[id] = handler; }});
    return elements.get(id);
  }},
  navigator: {mediaDevices: {getDisplayMedia: async () => stream}},
  AudioContext: class {
    constructor() { this.sampleRate = 16000; this.audioWorklet = {addModule: async () => {}}; }
    async resume() {} async close() {}
    createMediaStreamSource() { return {connect() {}}; }
  },
  AudioWorkletNode: class {
    constructor() {
      node = this;
      this.port = {postMessage: message => {
        assert.equal(message.type, 'flush');
        node.port.onmessage({data: new ArrayBuffer(4)});
        node.port.onmessage({data: {type: 'flushed'}});
      }};
    }
    connect() {}
  },
  MediaStream: class {}, AbortController, setTimeout, clearTimeout, setInterval() {},
  window: {addEventListener() {}}, captureParticipants: () => ['Ana', 'João'], resetSessionView() {}, refresh() {},
  fetch: async (url, options) => {
    if (url.endsWith('/start')) {
      assert.deepEqual(JSON.parse(options.body).participants, ['Ana', 'João']);
      return {ok: true, json: async () => ({session: 'test', sample_rate: 16000, chunk_seconds: 12})};
    }
    if (url.includes('/audio?')) {
      uploads.push({url, size: options.body.byteLength});
      if (uploads.length === 1) await new Promise(resolve => { releaseFirst = resolve; });
      return {ok: true, json: async () => ({accepted: true})};
    }
    if (url.includes('/stop?')) {
      stopped.push(url);
      return {ok: true, json: async () => ({stopped: true, processing: true})};
    }
    throw Error(url);
  }
});
vm.runInContext(fs.readFileSync('tab-capture.js', 'utf8'), context);
(async () => {
  await elements.get('capture').onclick();
  node.port.onmessage({data: new ArrayBuffer(32000)});
  await new Promise(resolve => setImmediate(resolve));
  handlers.stop();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(stopped.length, 0, 'Stop must wait for outstanding audio uploads');
  releaseFirst();
  for (let i = 0; i < 20 && !stopped.length; i++) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(uploads.map(u => u.size), [32000, 4]);
  assert.match(uploads[0].url, /sequence=0/);
  assert.match(uploads[1].url, /sequence=1/);
  assert.equal(stopped.length, 1);
  assert.match(elements.get('capture-message').textContent, /fila continua/);
  console.log('Capture client: participants, ordered uploads, final partial block and stop-after-commit passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
