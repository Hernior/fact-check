let viewedSession = '', sessionsSignature = '';

function feedPath() {
  return 'feed' + (viewedSession ? '?session_id=' + encodeURIComponent(viewedSession) : '');
}

function resetSessionView() {
  viewedSession = '';
  $('session-history').value = '';
}

function renderSessions(items) {
  if (!Array.isArray(items)) return;
  $('session-view-label').textContent = viewedSession ? 'Consultando uma sessão arquivada. A captura atual continua independente.' : 'Exibindo a sessão atual.';
  const signature = JSON.stringify(items);
  if (signature === sessionsSignature) return;
  sessionsSignature = signature;
  const select = $('session-history');
  select.replaceChildren();
  const current = tag('option', '', 'Sessão atual'); current.value = '';
  select.append(current);
  for (const item of items.filter(s => s.archived)) {
    const option = tag('option', '', (item.source === 'legacy' ? 'Histórico anterior' : new Date(item.started).toLocaleString('pt-BR')) +
      ` · ${item.segments} trechos · ${item.claims} afirmações`);
    option.value = item.id; select.append(option);
  }
  select.value = viewedSession;
  select.onchange = () => { viewedSession = select.value; refresh(); };
  $('session-view-label').textContent = viewedSession ? 'Consultando uma sessão arquivada. A captura atual continua independente.' : 'Exibindo a sessão atual.';
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  return Math.floor(seconds / 60) + 'min ' + seconds % 60 + 's';
}

function renderProgress(status) {
  const data = status.progress;
  $('capture-progress').hidden = !data;
  if (!data) return;
  const received = data.received_seconds;
  const percent = value => received ? Math.min(100, Math.floor(value / received * 100)) : 0;
  $('progress-received').textContent = `Áudio salvo: ${duration(received)} · ${data.received} blocos`;
  $('progress-transcription').value = percent(data.transcribed_seconds);
  $('progress-transcription-label').textContent = `Transcrição: ${duration(data.transcribed_seconds)} de ${duration(received)} (${percent(data.transcribed_seconds)}%)`;
  $('progress-analysis').value = percent(data.analyzed_seconds);
  $('progress-analysis-label').textContent = `Análise automática: ${duration(data.analyzed_seconds)} de ${duration(received)} (${percent(data.analyzed_seconds)}%)`;
  $('progress-spinner').hidden = !status.running && !status.stopping;
  $('progress-message').textContent = data.failed
    ? `${data.failed} bloco(s) com falha. O áudio está preservado; o processamento não foi concluído integralmente.`
    : data.pending ? `${data.pending} bloco(s) pendente(s) · atraso de ${duration(data.delay_seconds)}. ${status.running ? 'Captura continua; o áudio recebido está salvo.' : 'Captura parada; concluindo a fila salva.'}`
    : status.running ? 'Em dia. Aguardando os próximos blocos.' : 'Processamento concluído.';
}
