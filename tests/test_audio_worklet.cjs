const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let Processor;
const chunks = [];
vm.runInNewContext(fs.readFileSync('audio-worklet.js', 'utf8'), {
  AudioWorkletProcessor: class {
    constructor() { this.port = {postMessage: buffer => chunks.push(buffer)}; }
  },
  registerProcessor: (_, implementation) => { Processor = implementation; }
});
const processor = new Processor({processorOptions: {chunkSamples: 4}});
processor.process([[new Float32Array([1, -1]), new Float32Array([0, 0])]]);
assert.equal(chunks.length, 0, 'partial chunks must not upload');
processor.process([[new Float32Array([3, -3]), new Float32Array([3, -3])]]);
assert.equal(chunks.length, 1);
const data = new DataView(chunks[0]);
assert.deepEqual([0, 2, 4, 6].map(offset => data.getInt16(offset, true)), [16384, -16384, 32767, -32768]);
processor.process([[new Float32Array(4)]]);
assert.equal(chunks.length, 2, 'recording must continue after each complete chunk');
assert.equal(processor.process([[]]), true, 'missing audio must not stop the processor');
processor.process([[new Float32Array([0.5, -0.5])]]);
processor.port.onmessage({data: {type: 'flush'}});
assert.equal(chunks[2].byteLength, 4, 'stop must send the final partial block');
assert.equal(chunks[3].type, 'flushed', 'flush acknowledgement follows the audio');
processor.process([[new Float32Array(4)]]);
assert.equal(chunks.length, 4, 'flushed processor must not generate more audio');
console.log('PCM: stereo downmix, clipping, little endian and chunk boundaries passed.');
