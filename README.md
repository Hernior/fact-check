# Checagem ao Vivo — live transcription & evidence research MVP

Independent educational/open-source starting point for Brazilian Portuguese fact-checking of YouTube livestreams. **Not a reproduction of the reference site's code, brand, or data.**

## What works

- Captures a Chrome tab selected directly from the dashboard and sends mono 16-kHz PCM audio to local `faster-whisper`. A locally managed Qwen llamafile identifies claims and saves them as **aguardando pesquisa**, with no truth verdict or automatic source search. Transcription and extraction need no YouTube URL, `yt-dlp`, FFmpeg, API key, or paid service. Manual evidence search uses the configured Brave key and its bounded quota.
- Validates a YouTube watch/short URL; uses `yt-dlp` to resolve the media and `ffmpeg` to decode mono 16-kHz audio.
- Processes incoming audio in 12-second chunks using local `faster-whisper`.
- When `OPENAI_API_KEY` is set, extracts up to 3 potentially verifiable factual claims per chunk using an LLM.
- When `BRAVE_SEARCH_API_KEY` is set, searches for source URLs and snippets supporting research for each claim.
- Keeps snippet-only assessments **inconclusive** (never fabricates a verified true/false result). Human review and full-source retrieval are prerequisites for defensible verdicts.
- Saves transcripts, claims, sources, and timestamps to SQLite, displaying updates every 3 seconds in a responsive dashboard.
- Stop/start from the browser. Old session data is retained for now.

## Requirements

Python 3.10+ and Chrome for tab capture. Download `Qwen3.5-0.8B-Q8_0.llamafile` from the [Mozilla quickstart](https://docs.mozilla.ai/llamafile/getting-started/quickstart) into the project directory for local claim extraction. Internet access is needed to install dependencies and download Whisper weights on first use; inference runs locally. The alternative YouTube URL mode additionally requires `ffmpeg` on PATH and access to the stream. `yt-dlp` and YouTube may block some formats or change behavior. CPU processing can lag behind live audio; smaller Whisper models may help.

Whisper is explicitly configured for CPU with `int8`; CUDA, cuBLAS and cuDNN are not required. This avoids automatic GPU selection on machines missing NVIDIA runtime libraries. GPU acceleration would require a separate configuration change.

## Run

```bash
# From the project directory:
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Optional: copy .env.example to .env and fill in the keys
uvicorn server:app --host 127.0.0.1 --port 8000
```

Visit http://127.0.0.1:8000 in Chrome. The server is necessary; opening `live.html` alone or publishing it on GitHub Pages does **not** provide live audio capture/transcription.

`index.html` is a separate manual panel using browser storage; it is not synchronized with the live dashboard or SQLite. `.env.example` is a configuration template; only `.env` is loaded automatically.

### Capturar uma aba pelo painel

1. Abra a transmissão em outra aba do Chrome e inicie a reprodução.
2. No painel, clique em **Capturar áudio da aba**.
3. No seletor do Chrome, escolha uma **aba** e marque **Compartilhar áudio da aba**. Janelas, telas ou seleções sem áudio são recusadas pelo painel.
4. Aguarde o primeiro bloco (12 segundos por padrão) e o processamento do Whisper. Na primeira utilização, o download do modelo pode aumentar a espera.
5. Use **Parar** no painel ou encerre o compartilhamento no Chrome.

A captura permanece na aba escolhida ao alternar entre abas. Mantenha o painel aberto; recarregar ou fechar o painel encerra a captura. O navegador exige permissão a cada início. O vídeo solicitado pelo seletor não é enviado ao servidor; somente o áudio é processado, sem microfone. Use localhost/127.0.0.1 ou HTTPS.

O servidor confirma cada bloco somente depois de gravá-lo no SQLite. A fila fica em disco e não tem o antigo limite de cinco blocos; uma análise lenta acumula atraso sem descartar áudio recebido. Whisper e análise textual trabalham separadamente. `CHUNK_SECONDS` aceita de 1 a 30 segundos. Um modelo Whisper menor ainda pode reduzir o atraso, mas não é necessário trocá-lo apenas por causa da antiga fila cheia.

Ao clicar em **Parar**, o navegador tenta enviar também o último bloco parcial, aguarda o envio dos blocos em trânsito e encerra a entrada de áudio. O servidor continua processando a fila salva. Uma nova captura fica disponível após terminar esse processamento. Fechar/recarregar o navegador abruptamente pode perder áudio ainda não confirmado pelo servidor; os blocos já salvos permanecem preservados. Falhas reais de envio ou de disco são informadas; espaço livre continua necessário.

### Progresso e sessões arquivadas

O painel mostra áudio salvo, barras de transcrição e análise automática, blocos pendentes, atraso em minutos/segundos e indicador de processamento. Os percentuais usam a duração do áudio recebido até o momento: podem diminuir enquanto novos blocos chegam. Depois de parar, chegam a 100% quando todo o trabalho automático foi concluído. Falhas são indicadas separadamente, sem mostrar conclusão integral. A revisão manual de sujeitos e a pesquisa Brave não entram nessas barras.

Iniciar outra captura arquiva a sessão anterior inteira e abre o painel vazio para a nova. Use **Sessão exibida** para consultar os resultados arquivados, incluindo afirmações, sujeitos, fontes e avaliações já salvas. O histórico anterior à implementação também é preservado como **Histórico anterior**. As consultas do feed continuam limitadas aos 80 registros mais recentes por tipo da sessão selecionada; os demais registros permanecem no banco.

As tabelas `capture_sessions` e `capture_audio` guardam as sessões e os blocos PCM; `segments`, `transcript_paragraphs` e `claims` recebem `session_id`. Os IDs das afirmações e suas evidências são preservados. O áudio fica no SQLite até a limpeza confirmada, consumindo cerca de 115 MB por hora, além dos textos e índices. Após reiniciar o Python, blocos ainda pendentes e análises interrompidas são retomados automaticamente; o compartilhamento da aba precisa ser iniciado novamente pelo usuário. Blocos com falha explícita ficam preservados com erro, sem garantia de reprocessamento automático.

A captura de aba usa transcrição e IA locais. **O modo antigo por URL do YouTube ainda usa OpenAI e Brave** automaticamente; na captura de aba, a Brave só é chamada por solicitação manual. O Chrome foi escolhido como navegador pelo usuário; o código de captura usa APIs nativas e não depende de extensão ou serviço proprietário de captura.

### IA local com llamafile: execução automática

Ao iniciar o Uvicorn, o Python inicia o llamafile em segundo plano, sem janela adicional no Windows, e aguarda sua API responder. No Windows, se necessário, o arquivo `.llamafile` é renomeado para `.llamafile.exe`; não é feita outra cópia dos pesos. Em Linux/macOS, conceda permissão de execução ao arquivo antes de iniciar (`chmod +x Qwen3.5-0.8B-Q8_0.llamafile`). O binário/modelo não é incluído no Git.

O processo usa CPU com quatro threads por padrão, contexto de 4.096 tokens e somente uma requisição por vez, escutando em `127.0.0.1:8080`, com um token temporário gerado pelo Python. As chaves OpenAI/Brave não são passadas ao processo filho. As chamadas locais ignoram proxies do ambiente e não seguem redirecionamentos. Não é necessário iniciar o modelo manualmente.

O painel mostra a IA como carregando, pronta ou indisponível. Enquanto o modelo carrega, o Whisper pode transcrever e a análise aguarda. Se o modelo estiver ausente, falhar ou responder em formato inválido, a transcrição é preservada e o processamento sinaliza falha; trechos com falha explícita não são reprocessados automaticamente. A identificação sequencial pode acumular atraso na análise, mas não bloqueia o recebimento de áudio nem o Whisper. O Qwen de 0,8B pode omitir ou classificar incorretamente afirmações; é necessário revisar os resultados.

A seleção continua conservadora quanto à elegibilidade: opiniões, perguntas, fragmentos e casos incertos são descartados. Na captura da aba, a dúvida apenas sobre o sujeito de uma declaração objetiva gera uma pendência para o usuário, conforme o fluxo de participantes abaixo. A primeira chamada local propõe até três trechos literais, preservando nomes, números e negações. O Python elimina duplicatas, perguntas (inclusive quando o modelo omite o ponto de interrogação), opiniões pessoais explícitas e repetições típicas de transcrição incoerente.

Se restarem candidatos, uma segunda chamada ao mesmo modelo local avalia sua elegibilidade. Só são aceitas declarações objetivas com possibilidade de confirmação ou refutação por evidências externas; a identidade do sujeito pode aguardar esclarecimento manual. Opiniões, perguntas, fragmentos, casos incertos sobre a própria alegação, decisões ausentes/conflitantes e respostas inválidas não são registrados como afirmações. Não é feita pesquisa de fontes nessa avaliação. Uma declaração falsa também é elegível se puder ser verificada: esta etapa não determina a verdade.

Isso acrescenta uma chamada local por passagem com candidatos e pode descartar afirmações válidas; os filtros e o modelo de 0,8B não garantem classificação perfeita. A correção vale para novas extrações: registros antigos não são apagados ou reprocessados. Cada registro aceito fica como **aguardando pesquisa** ou **aguardando sujeito**, sem fontes e sem avaliação de veracidade. OpenAI e Brave não são chamadas na extração automática, mesmo que as chaves estejam configuradas; a pesquisa Brave exige um clique separado.

### Continuidade e revisão da transcrição da aba

Após transcrever cada bloco, o Qwen local revisa ortografia, pontuação e erros de reconhecimento usando o parágrafo anterior e o contexto recente da mesma captura. No fluxo persistente, a revisão trabalha com o texto já transcrito e não repete o Whisper, preservando sua capacidade de acompanhar os próximos blocos. Nenhuma dessas etapas utiliza OpenAI ou Brave. O áudio permanece na fila SQLite para preservação e retomada.

Quando houver continuidade, os trechos formam um único parágrafo revisado. As afirmações são extraídas ao encerrar o parágrafo, por mudança de assunto, silêncio, limite de três blocos/36 segundos ou parada da captura. O limite também considera 2.400 caracteres para conter o contexto do modelo; não há garantia de identificar o fim linguístico de todo parágrafo. Parágrafos maiores são analisados em passagens de até três frases, com uma frase de sobreposição, evitando aplicar o limite de três candidatos ao parágrafo inteiro. A parada pode levar alguns segundos para concluir a identificação do último parágrafo. Duplicatas textuais são evitadas dentro da mesma captura.

O painel mostra os parágrafos com intervalo de tempo e **Ver transcrição original**. Os blocos originais continuam intactos em `segments`; a tabela adicional `transcript_paragraphs` guarda texto revisado, originais associados, intervalos e estado de conclusão. A limpeza confirmada remove ambos, preservando a cota Brave. Dados antigos não são reprocessados.

A revisão não verifica veracidade. Instruções ao modelo preservam perguntas, negações e atribuições; validação adicional rejeita remoção/adição de pontos de interrogação, mudanças em números escritos com dígitos e reescritas excessivas. Palavras ambíguas podem continuar incorretas e a IA pode errar nomes ou junções. Falha ou resposta inválida mantém o original visível com aviso. A revisão acrescenta custo de CPU e latência; a capacidade de acompanhar uma transmissão real precisa ser medida no computador de uso.

### Participantes e sujeito por afirmação no debate

Antes de capturar uma aba, informe opcionalmente os participantes separados por vírgula (até 20 nomes, de 2 a 80 caracteres). A lista é copiada no início da sessão e guardada com cada pendência; ela não identifica automaticamente quem fala ou a quem um pronome se refere. O modo alternativo de URL do YouTube mantém seu fluxo anterior.

Na captura de aba, declarações objetivas com pronomes podem passar para revisão de elegibilidade. Frases literais também são apresentadas ao revisor quando o pequeno modelo proponente as omite ou parafraseia; continuam exigindo avaliação semântica como verificáveis. Perguntas, opiniões, insultos e fragmentos não devem virar afirmações. Só um vínculo textual explícito e sem nomes concorrentes permite atribuição automática; na dúvida, o cartão fica **aguardando sujeito**.

O cartão mostra a fala original, o contexto, botões com os participantes e um campo para outro sujeito. A captura continua enquanto você decide. A confirmação substitui a referência somente naquela afirmação, preservando o original e as quantidades; não altera outros cartões, o parágrafo ou as atribuições seguintes. Interrupções e trocas de interlocutor podem exigir novas confirmações. Não há identificação de voz por participante.

Escolher o sujeito não confirma a acusação e não pesquisa. A pesquisa Brave permanece manual e é bloqueada pelo servidor enquanto faltar o sujeito, sem gastar cota. A tabela adicional `claim_subjects` guarda original, referência, participantes, contexto e autoria/data da resolução; ela é removida na limpeza confirmada, preservando o contador Brave. O histórico antigo não é reprocessado. As verificações controladas não garantem que o modelo de 0,8B identificará toda alegação em um debate real.

Ao encerrar o servidor Python normalmente (Ctrl+C), o processo llamafile iniciado por ele é encerrado. Uma porta ocupada gera um aviso; nenhum processo externo é reutilizado ou encerrado. Use uma única instância/worker do Uvicorn. Encerramento forçado do Python pelo sistema pode impedir a limpeza; nesse caso, encerre a instância órfã antes de reiniciar.

Configurações opcionais no `.env`:

```dotenv
LLAMAFILE_PATH=Qwen3.5-0.8B-Q8_0.llamafile
LLAMAFILE_PORT=8080
LLAMAFILE_STARTUP_TIMEOUT=180
LLAMAFILE_REQUEST_TIMEOUT=60
LLAMAFILE_THREADS=4
```

Se a inicialização falhar, consulte o console Python e `data/llamafile.log`. Alterações de configuração exigem reiniciar o servidor.

### Limpar os dados do painel

O botão **Limpar dados** só fica disponível após parar a captura. A confirmação informa que todas as transcrições, afirmações e evidências salvas serão apagadas permanentemente. Cancelar não altera os dados. Durante uma captura ou enquanto seu encerramento estiver pendente, o servidor também recusa a limpeza, mesmo que a chamada seja feita diretamente à API.

A exclusão de sessões, áudio, transcrições, afirmações e evidências ocorre em uma única transação: uma falha mantém todo o histórico. A limpeza também aguarda o processamento da fila terminar. O contador `brave_search_attempts` é preservado, assim como chaves, modelos e arquivos de configuração. A limpeza zera os contadores da sessão e atualiza o painel; não remove o arquivo SQLite nem altera o esquema. O painel manual `index.html`, que usa armazenamento do navegador, é independente e não é limpo por esse botão.

### Optional credentials

- `OPENAI_API_KEY`: claim identification and preliminary source-context evaluation.
- `BRAVE_SEARCH_API_KEY`: search engine results for evidence investigation.
- `WHISPER_MODEL`: `tiny`, `base`, `small` (default), `medium`, etc.; first run downloads weights.
- `CHUNK_SECONDS`: chunk size (default 12).

The server automatically loads `.env` from the directory containing `server.py`, before reading configuration. UTF-8 files with or without a BOM are accepted. Existing terminal/service environment variables take precedence, including variables explicitly set to an empty string. If `.env` does not exist, the server uses the environment and defaults normally. Restart the Python server after editing `.env`.

Alternatively, in Windows PowerShell set `$env:OPENAI_API_KEY="..."` in the terminal running Uvicorn. `.env.example` is a reference and is not loaded. Never expose API keys in browser code or commit them to Git; `.env` is ignored. `/api/status` reports only whether each key is configured, without returning its value. API calls have costs and may be rate-limited. A configured key has not necessarily been validated by its provider.

## Important limitations

### Brave Search: limite estrito de consultas

O servidor permite no máximo **1.000 tentativas em qualquer janela móvel de 32 dias**. Essa janela conservadora cobre um período mensal sem presumir o dia/fuso de renovação dos créditos da conta. Não há zeragem automática na virada do mês: as consultas voltam a ficar disponíveis individualmente quando as respectivas tentativas completam 32 dias. Isso pode liberar a cota depois da renovação na Brave, mas evita liberá-la antes.

O consumo inicial é zero, conforme informado pelo usuário, e a chave é exclusiva deste projeto. Cada tentativa é registrada e confirmada no SQLite **antes** do envio HTTP, inclusive se houver erro, timeout ou cancelamento; não há reembolso local nem repetição automática. A Brave pode contabilizar menos chamadas do que esse controle conservador. O painel mostra consumo e saldo. Ao esgotar a cota, a pesquisa retorna sem evidências e não envia novas chamadas; a transcrição continua.

A reserva usa uma transação SQLite para impedir que chamadas simultâneas ultrapassem o limite, inclusive entre processos que compartilham o mesmo arquivo de banco. Se não for possível registrar a tentativa, a busca não é enviada. Todas as chaves desta instalação compartilham o contador: trocar a chave ou reiniciar o servidor não renova a cota.

O histórico fica na tabela `brave_search_attempts`, em `data/live.sqlite3`, sem chaves nem consultas. **Preserve esse banco e não utilize a chave em outras aplicações, cópias do projeto com bancos independentes ou no playground da Brave**: o contador local não conhece esses consumos. Como proteção adicional na conta, mantenha a recarga automática desativada e o limite de gasto dentro dos créditos gratuitos no painel da Brave.

O `.env` é carregado automaticamente. A captura de aba usa transcrição e identificação de afirmações locais; configurar as chaves não ativa pesquisa automática nesse modo.

### Pesquisa manual por afirmação

Clique em **Pesquisar evidências** na afirmação desejada. A pesquisa usa o texto salvo, até sete resultados da Brave e uma tentativa da cota persistente. Após salvar os resultados, o servidor tenta analisá-los com o Qwen local. Não há chamada OpenAI nem pesquisa adicional nessa análise. O botão de pesquisa fica indisponível sem chave, com cota esgotada ou enquanto a mesma afirmação é pesquisada. Buscas podem ser feitas durante a captura.

Os links e os trechos do buscador aparecem no cartão. Resultados com fontes ficam **inconclusivos**: os trechos não confirmam nem refutam a afirmação e ainda exigem leitura das fontes completas. Uma busca sem resultados fica como **sem evidências**. O resultado e sua data de pesquisa são salvos no SQLite; uma pesquisa concluída, mesmo sem resultados, não é repetida por outro clique nem após reiniciar o servidor.

Falhas não são repetidas automaticamente. A tentativa continua consumida e o cartão informa o erro; outro clique permite tentar novamente. Cliques simultâneos na mesma afirmação não disparam buscas duplicadas na instância única do servidor. Enquanto houver pesquisa em andamento, a limpeza é bloqueada para não excluir a afirmação antes de salvar a resposta. Limpar os dados apaga também o histórico de pesquisas das afirmações, preservando a cota Brave.

A consulta não é truncada: textos com mais de 600 caracteres ou 75 palavras são recusados antes de gastar cota. A leitura do conteúdo das páginas exige a ação separada **Analisar fontes completas**, descrita abaixo. O controle de consultas permanece em 1.000 tentativas por janela móvel de 32 dias.

### Sugestão preliminar de Fato, Fake ou Inconclusivo

A IA local recebe somente a afirmação e os trechos de busca já salvos. Os dados enviados são limitados a sete fontes, títulos de 120 caracteres, URLs de 200 caracteres e trechos de 450 caracteres; partes além desses limites não são analisadas. O cartão mostra uma sugestão com justificativa, data e citações literais ligadas às fontes originais:

- **Verde, ✓ Sugestão: Fato**: os trechos sustentam diretamente a afirmação.
- **Vermelho, ⚠ Sugestão: Fake**: os trechos contradizem diretamente a afirmação.
- **Amarelo, ? Sugestão: Inconclusivo**: evidências ausentes, insuficientes, conflitantes ou sem referências válidas.

O painel exibe permanentemente o aviso: **Análise preliminar gerada por IA, sujeita a erros. “Fato” indica evidências que sustentam a afirmação; “Fake” indica evidências que a contradizem. A ausência de evidências não comprova falsidade. Resultados e trechos de busca podem omitir contexto; consulte as fontes originais.**

Fato/Fake exigem indicação explícita de suficiência, ausência de conflito e uma citação literal válida com relação de apoio/contradição correspondente. URLs ou citações inventadas pelo modelo não são usadas. Referências à ausência ou insuficiência de informações na justificativa/citação são rebaixadas a Inconclusivo; para sugerir Fato, os números da afirmação também devem aparecer nas citações. Essas travas reduzem erros conhecidos, mas não garantem acurácia semântica do Qwen de 0,8B.

A sugestão é guardada separadamente em `claim_assessments`, com modelo, data, justificativa e referências. O resultado bruto da pesquisa em `claims` continua sem um veredito editorial conclusivo. Sem resultados, a sugestão é Inconclusivo, sem chamar a IA. Se a IA estiver indisponível ou falhar, os resultados Brave são preservados, e o botão **Analisar com IA** permite tentar depois, inclusive para pesquisas antigas e com cota Brave esgotada. Essa ação não faz outra consulta Brave. A limpeza aguarda análises em andamento e apaga também suas sugestões, mantendo a cota de consultas.

### Analisar fontes completas

Após pesquisar uma afirmação, clique em **Analisar fontes completas**. O servidor consulta até sete links já salvos, extrai o texto principal das páginas HTML com Trafilatura e reavalia seu conteúdo com o Qwen local. Não há nova chamada Brave nem OpenAI. As dependências estão no `requirements.txt`; atualize o ambiente com `pip install -r requirements.txt` ao instalar esta versão.

O cartão passa a indicar **Análise do texto extraído das fontes**, mostra as datas detectadas (ou "não informada"), o horário de consulta, os erros de acesso e passagens literais vinculadas às fontes. Datas são metadados detectados automaticamente, não datas confirmadas editorialmente. As citações têm posições registradas no texto extraído, permitindo conferir a origem. O HTML original e o texto integral não são reenviados ao navegador a cada atualização do painel.

O download aceita somente HTTPS público na porta 443, com validação de certificado. Endereços privados/locais são bloqueados; cada redirecionamento passa pela mesma validação. A conexão usa o IP público validado e preserva o hostname na verificação TLS, evitando uma segunda resolução DNS no momento da conexão. Não usa cookies de sessão nem executa JavaScript. Páginas bloqueadas, paywalls, HTTP 403, PDFs, outros formatos e conteúdo que exige JavaScript podem ficar indisponíveis; não há tentativa de contornar essas restrições.

Limites desta entrega: três downloads em paralelo, até quatro destinos por cadeia de redirecionamentos, até 2 MB por página, texto extraído de até 120.000 caracteres e tempo de conexão/leitura limitado. Respostas comprimidas que ignorem o pedido de conteúdo sem compressão são recusadas. O texto principal é guardado no SQLite; comentários e menus são excluídos quando o extrator os reconhece.

Por causa do contexto do Qwen, o texto é analisado em até oito blocos por fonte, cada um com no máximo 2.800 caracteres e sobreposição de até 240 caracteres. O painel informa a leitura/análise incompleta. **Se alguma fonte não puder ser lida, o texto exceder esse orçamento, a IA falhar ou surgirem evidências conflitantes, o resultado passa a Inconclusivo**, mesmo que os resumos da Brave tenham produzido outra sugestão. Uma análise completa sem evidências suficientes também permanece Inconclusiva. Blocos irrelevantes não são tratados como prova de falsidade; as classificações continuam preliminares e sujeitas a revisão humana.

A sugestão por texto extraído tem preferência visual sobre a sugestão baseada em resumos, que é preservada separadamente. Textos e metadados ficam em `claim_full_sources`, e o resultado/citações em `claim_full_assessments`. Uma análise concluída é reutilizada; uma leitura incompleta oferece **Tentar fontes novamente**. Enquanto a operação está em andamento, cliques duplicados e a limpeza são bloqueados. **Limpar dados** apaga também esses textos e resultados, preservando o contador Brave.

- Real Chrome tab sharing and live-stream transcription still require an interactive integration check; automated checks do not establish real capture quality, latency, or model accuracy.
- A live stream isn't necessarily at the live edge; processing and verification add delay.
- Audio extraction via `yt-dlp` is fragile and may break after YouTube changes.
- Stream state is in-process (one active stream per Python process), not ready for multiworker deployment. FFmpeg is terminated on stop.
- Model-generated claim extraction can miss facts or make errors; speaker attribution, deduplication, overlap, citation full-text extraction, and fact-checking editorial review remain future work.
- Source search results are **not source validation**. The app deliberately does not proclaim factual verdicts based on snippets.
- `/api` endpoints have no authentication; bind to localhost only. Add auth, rate limits, request quotas, storage policies, observability and resilient work queues before public deployment.

## Architecture

Chrome tab selection -> AudioWorklet (mono PCM 16 kHz) -> committed SQLite audio queue -> independent faster-whisper worker -> committed transcripts -> managed Qwen analysis worker -> SQLite -> progress and session dashboard. No cloud AI/search API calls during automatic extraction. Manual click -> quota reservation -> Brave -> inconclusive search results saved in SQLite.

YouTube URL -> `yt-dlp` -> FFmpeg PCM -> faster-whisper -> LLM claim extraction -> Brave web search -> cautious LLM pre-analysis -> SQLite -> FastAPI JSON -> browser polling.

## Focused checks

```bash
python -m unittest discover -s tests -p test_tab_capture.py -v
python -m unittest discover -s tests -p test_capture_sessions.py -v
node tests/test_session_panel.cjs
node tests/test_tab_capture_client.cjs
node tests/test_audio_worklet.cjs
python -m unittest discover -s tests -p test_brave_quota.py -v
python -m unittest discover -s tests -p test_local_llm.py -v
python -m unittest discover -s tests -p test_clear_data.py -v
node tests/test_clear_button.cjs
python -m unittest discover -s tests -p test_manual_search.py -v
node tests/test_manual_search_button.cjs
python -m unittest discover -s tests -p test_assessment.py -v
python -m unittest discover -s tests -p test_full_sources.py -v
```

The Python checks use test doubles for models and process launch: they validate PCM upload, queue bounds, session isolation, stop/restart, transcript/claim persistence, conservative claim eligibility, local model errors, process ownership, lifecycle cleanup and quota enforcement without invoking paid services. `tests/fixtures/claim_selection.json` contains seven reported false positives, clear verifiable claims (including an incorrect assertion), and ambiguous/subjective samples for real-model spot checks. The JavaScript check validates stereo downmix, signed little-endian PCM, clipping, and block boundaries. These checks do not replace a real Chrome sharing/transcription check or a factual-accuracy benchmark.

## Roadmap

Full article retrieval and content extraction; source trust policy and provenance; fact-check APIs and official datasets; review queue with editorial approval; speaker diarization; automatic reconnection/deduplication; smarter chunk boundaries; WebSocket push; docker-compose; multi-stream worker queue; benchmarks for latency, accuracy, and cost.
