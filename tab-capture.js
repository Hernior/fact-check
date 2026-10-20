(() => {
  const button = document.getElementById('capture');
  const message = document.getElementById('capture-message');
  let capture = null;
  let starting = false;

  async function post(path, options = {}) {
    const response = await fetch('/api/tab/' + path, {method: 'POST', ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Falha na captura.');
    return body;
  }

  async function finish(current, error = '') {
    if (!current || current.stopping) return;
    current.stopping = true;
    try {
      if (current.node) {
        current.flushing = true;
        await new Promise(resolve => {
          const timeout = setTimeout(resolve, 2000);
          current.flushed = () => { clearTimeout(timeout); resolve(); };
          current.node.port.postMessage({type: 'flush'});
        });
        current.flushing = false;
        current.node.port.onmessage = null;
      }
      current.stream.getTracks().forEach(track => track.stop());
      if (current.context) await current.context.close().catch(() => {});
      await current.sending;
    } catch (failure) {
      error = error || 'Falha ao enviar áudio. Os blocos confirmados pelo servidor estão preservados.';
    }
    try {
      if (current.session) await post('stop?session=' + encodeURIComponent(current.session));
    } catch (failure) {
      error = error || failure.message;
    } finally {
      if (capture === current) capture = null;
      delete button.dataset.busy;
      message.textContent = error || 'Captura encerrada. O áudio recebido está salvo e a fila continua sendo processada. Acompanhe o progresso abaixo.';
      button.disabled = false;
      refresh();
    }
  }

  button.onclick = async () => {
    if (capture || starting) return;
    const participants = captureParticipants();
    starting = true;
    button.dataset.busy = 'true';
    button.disabled = true;
    message.textContent = '';
    let current = null;
    try {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        throw new Error('Abra o painel no Chrome em http://localhost:8000 ou http://127.0.0.1:8000.');
      }
      // This must run directly from the click, before any asynchronous HTTP call.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {displaySurface: 'browser'}, audio: true,
        selfBrowserSurface: 'exclude', systemAudio: 'exclude',
        surfaceSwitching: 'exclude', preferCurrentTab: false
      });
      current = {stream, controller: new AbortController(), stopping: false, sending: Promise.resolve(), sequence: 0};
      capture = current;
      if (stream.getVideoTracks()[0]?.getSettings().displaySurface !== 'browser') {
        throw new Error('Selecione uma aba do Chrome, em vez de uma janela ou tela.');
      }
      if (!stream.getAudioTracks().length) {
        throw new Error('Nenhum áudio compartilhado. Escolha a aba e marque “Compartilhar áudio da aba”.');
      }
      for (const track of stream.getTracks()) {
        track.onended = () => finish(current);
      }
      const context = current.context = new AudioContext({sampleRate: 16000});
      await context.resume();
      await context.audioWorklet.addModule('/audio-worklet.js');
      if (current.stopping) return;
      const session = await post('start', {headers: {'Content-Type': 'application/json'}, body: JSON.stringify({participants})});
      current.session = session.session;
      resetSessionView();
      if (current.stopping) {
        await post('stop?session=' + encodeURIComponent(session.session));
        return;
      }
      if (context.sampleRate !== session.sample_rate) throw new Error('Taxa de áudio incompatível com o Whisper.');
      const node = current.node = new AudioWorkletNode(context, 'tab-audio', {
        processorOptions: {chunkSamples: session.sample_rate * session.chunk_seconds}
      });
      const source = context.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
      source.connect(node);
      node.connect(context.destination);
      let pending = 0;
      node.port.onmessage = event => {
        if (event.data?.type === 'flushed') { current.flushed?.(); return; }
        if (current.stopping && !current.flushing) return;
        pending++;
        const sequence = current.sequence++;
        current.sending = current.sending.then(async () => {
          const timeout = setTimeout(() => current.controller.abort(), 20000);
          try { await post('audio?session=' + encodeURIComponent(current.session) + '&sequence=' + sequence, {
            headers: {'Content-Type': 'application/octet-stream'},
            body: event.data, signal: current.controller.signal
          }); } finally { clearTimeout(timeout); }
          pending--;
        });
        current.sending.catch(error => {
          if (!current.stopping) finish(current, error.message);
        });
      };
      node.onprocessorerror = () => finish(current, 'Falha no processamento do áudio. Inicie novamente.');
      message.textContent = `Capturando a aba escolhida em blocos de ${session.chunk_seconds}s. Transcrição e identificação de afirmações locais, sem APIs pagas. Mantenha o painel aberto.`;
      refresh();
    } catch (error) {
      const detail = error.name === 'NotAllowedError' ? 'Compartilhamento cancelado ou não autorizado.' : error.message;
      if (current) await finish(current, detail);
      else message.textContent = detail;
    } finally {
      starting = false;
      if (!capture) {
        delete button.dataset.busy;
        button.disabled = false;
      }
    }
  };

  document.getElementById('stop').addEventListener('click', () => {
    if (capture) finish(capture);
  });
  window.addEventListener('pagehide', () => {
    if (!capture) return;
    capture.stream.getTracks().forEach(track => track.stop());
    if (capture.session) navigator.sendBeacon('/api/tab/stop?session=' + encodeURIComponent(capture.session));
  });
  // Detect server-side failures/stops, including a stop from another panel.
  setInterval(async () => {
    if (!capture?.session || capture.stopping) return;
    const current = capture;
    try {
      const response = await fetch('/api/status');
      if (!response.ok) throw new Error('Não foi possível consultar o servidor.');
      const status = await response.json();
      if (!status.running || status.source !== 'tab') await finish(current, status.error || 'Captura encerrada pelo servidor.');
    } catch (error) {
      await finish(current, error.message);
    }
  }, 3000);
})();
