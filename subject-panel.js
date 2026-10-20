/* Subject choices belong to a single claim; capture continues independently. */
const subjectDrafts = new Map(), subjectErrors = new Map(), resolvingSubjects = new Set();

function captureParticipants() {
  return ($('participants').value || '').split(/[,;\n]/).map(s => s.trim()).filter(Boolean);
}

function renderSubject(x, el) {
  const data = x.subject_resolution;
  if (!data) return false;
  el.append(tag('p', 'tiny', 'Fala original: ' + data.original_text));
  if (data.resolved_at) {
    el.append(tag('p', 'tiny', 'Sujeito: ' + data.subject +
      (data.resolved_by === 'user' ? ' · informado por você para esta afirmação' : ' · identificado no contexto')));
    return false;
  }
  el.append(tag('p', '', 'Quem é “' + data.reference + '” nesta afirmação?'));
  el.append(tag('p', 'tiny', 'A captura continua. A escolha vale somente para esta afirmação e não confirma a alegação.'));
  const pending = resolvingSubjects.has(x.id);
  for (const name of data.participants) {
    const button = tag('button', '', name);
    button.type = 'button'; button.disabled = pending || clearing;
    button.onclick = () => selectSubject(x.id, name);
    el.append(button);
  }
  const form = tag('form', '', '');
  const input = tag('input', '', '');
  input.type = 'text'; input.placeholder = 'Ou informe outro sujeito';
  input.setAttribute('aria-label', 'Sujeito desta afirmação');
  input.minLength = 2; input.maxLength = 80; input.required = true;
  input.dataset.subjectInput = 'true';
  input.value = subjectDrafts.get(x.id) || '';
  input.disabled = pending || clearing;
  input.oninput = () => subjectDrafts.set(x.id, input.value);
  const submit = tag('button', '', pending ? 'Salvando...' : 'Confirmar sujeito');
  submit.type = 'submit'; submit.disabled = pending || clearing;
  form.onsubmit = event => {
    event.preventDefault();
    input.blur();
    selectSubject(x.id, input.value);
  };
  form.append(input, submit); el.append(form);
  if (subjectErrors.has(x.id)) el.append(tag('p', 'tiny', subjectErrors.get(x.id)));
  const details = tag('details', '', '');
  details.append(tag('summary', '', 'Ver contexto da fala'), tag('p', '', data.context));
  el.append(details);
  return true;
}

async function selectSubject(id, subject) {
  if (resolvingSubjects.has(id) || clearing) return;
  resolvingSubjects.add(id); subjectErrors.delete(id);
  const request = api('claims/' + id + '/subject', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({subject})
  });
  refresh();
  try { await request; subjectDrafts.delete(id); }
  catch (error) { subjectErrors.set(id, error.message); }
  finally { resolvingSubjects.delete(id); await refresh(); }
}
