// Chrome resamples the captured stream to the AudioContext's 16 kHz rate.
class TabAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = new ArrayBuffer(options.processorOptions.chunkSamples * 2);
    this.view = new DataView(this.buffer);
    this.position = 0;
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data?.type !== 'flush') return;
      this.stopped = true;
      if (this.position) this.port.postMessage(this.buffer.slice(0, this.position));
      this.position = 0;
      this.port.postMessage({type: 'flushed'});
    };
  }

  process(inputs) {
    if (this.stopped) return true;
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i];
      value = Math.max(-1, Math.min(1, value / channels.length));
      this.view.setInt16(this.position, Math.round(value * (value < 0 ? 32768 : 32767)), true);
      this.position += 2;
      if (this.position === this.buffer.byteLength) {
        const size = this.buffer.byteLength;
        this.port.postMessage(this.buffer, [this.buffer]);
        this.buffer = new ArrayBuffer(size);
        this.view = new DataView(this.buffer);
        this.position = 0;
      }
    }
    // Outputs stay silent: the source tab already plays its audio.
    return true;
  }
}
registerProcessor('tab-audio', TabAudioProcessor);
