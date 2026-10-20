"""Bounded public HTTPS retrieval, article extraction and full-text assessment."""
import http.client
import ipaddress
import json
import socket
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 2 * 1024 * 1024
MAX_TEXT = 120000
BLOCK_SIZE = 2800
BLOCK_OVERLAP = 240
MAX_BLOCKS = 8


def public_target(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.port not in (None, 443)):
        raise ValueError('Somente páginas HTTPS públicas na porta padrão são aceitas.')
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    ips = [item[4][0] for item in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in ips):
        raise ValueError('Endereço local, privado ou reservado não é permitido.')
    return parsed, ips[0]


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname, address):
        super().__init__(hostname, timeout=10, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        # Pin the validated IP, preserving TLS hostname validation and SNI.
        # The hostname is not resolved a second time during connection.
        raw = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def download_html(url):
    started = time.monotonic()
    for _ in range(4):
        parsed, address = public_target(url)
        connection = PublicHTTPSConnection(parsed.hostname, address)
        try:
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query
            connection.request('GET', path, headers={'User-Agent': 'FactCheckLocal/0.1',
                               'Accept': 'text/html,application/xhtml+xml', 'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader('Location')
                if not location:
                    raise ValueError('Redirecionamento sem destino.')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f'Fonte indisponível (HTTP {response.status}).')
            content_type = response.getheader('Content-Type', '').lower()
            if not any(kind in content_type for kind in ('text/html', 'application/xhtml+xml')):
                raise ValueError('Formato não suportado nesta entrega: somente páginas HTML.')
            if response.getheader('Content-Encoding', 'identity').lower() not in ('identity', ''):
                raise ValueError('Fonte retornou uma codificação de conteúdo não suportada.')
            length = response.getheader('Content-Length')
            if length and int(length) > MAX_BYTES:
                raise ValueError('Página maior que o limite de 2 MB.')
            body = bytearray()
            while True:
                if time.monotonic() - started > 30:
                    raise ValueError('Tempo limite de leitura da fonte atingido.')
                part = response.read1(65536)
                if not part:
                    break
                body.extend(part)
                if len(body) > MAX_BYTES:
                    raise ValueError('Página maior que o limite de 2 MB.')
            return bytes(body), url
        finally:
            connection.close()
    raise ValueError('Número máximo de redirecionamentos excedido.')


def extract_document(html, url):
    from trafilatura import extract
    output = extract(html, url=url, output_format='json', with_metadata=True,
                     include_comments=False, include_tables=True,
                     date_extraction_params={'extensive_search': False, 'original_date': True})
    data = json.loads(output) if output else {}
    text = ' '.join(str(data.get('text') or '').split())
    if len(text) < 200:
        raise ValueError('Texto principal insuficiente ou página bloqueada; não foi possível obter o artigo.')
    if len(text) > MAX_TEXT:
        raise ValueError('Texto extraído maior que o limite de 120.000 caracteres.')
    return {'title': data.get('title') or '', 'published_date': data.get('date'), 'text': text}


def read_source(source, source_id):
    document = {'source_id': source_id, 'url': source.get('url', ''), 'title': source.get('title', ''),
                'published_date': None, 'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'status': 'unavailable', 'text': '', 'error': None}
    try:
        html, final_url = download_html(document['url'])
        extracted = extract_document(html, final_url)
        document.update(extracted, url=final_url, status='ok')
    except Exception as exc:
        document['error'] = str(exc) if isinstance(exc, ValueError) else 'Não foi possível acessar ou extrair a fonte.'
    return document


def text_blocks(text):
    start = 0
    blocks = []
    while start < len(text) and len(blocks) < MAX_BLOCKS:
        end = min(start + BLOCK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(' ', start + BLOCK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        blocks.append(text[start:end])
        if end == len(text):
            return blocks, True
        start = max(start + 1, end - BLOCK_OVERLAP)
        while start < len(text) and not text[start - 1].isspace():
            start += 1
    return blocks, False


async def assess_documents(model, claim, documents, progress=None):
    issues = []
    citations = []
    suggestions = set()
    conflict = False
    for index, document in enumerate(documents):
        if progress is not None:
            progress['message'] = f'Analisando fonte {index + 1}/{len(documents)}'
        if document['status'] != 'ok':
            issues.append('Uma ou mais fontes não puderam ser lidas.')
            continue
        blocks, complete = text_blocks(document['text'])
        document['analysis_complete'] = complete
        document['analyzed_characters'] = 0
        if not complete:
            issues.append('Uma ou mais fontes excederam o limite de texto analisável.')
        for block in blocks:
            try:
                result = await model.assess_claim(claim, [{'title': document['title'], 'url': document['url'],
                    'snippet': block, 'published_date': document['published_date']}], basis='full_text')
                document['analyzed_characters'] = max(document['analyzed_characters'], document['text'].find(block) + len(block))
                conflict = conflict or result.get('conflicting', False)
                if result['verdict'] in {'fato', 'fake'}:
                    suggestions.add(result['verdict'])
                for citation in result['citations']:
                    quote = citation['quote']
                    start = document['text'].casefold().find(quote.casefold())
                    if start >= 0:
                        citations.append({**citation, 'source_id': document['source_id'],
                                          'char_start': start, 'char_end': start + len(quote)})
            except Exception:
                document['analysis_complete'] = False
                issues.append('A IA local não concluiu a análise de todos os textos.')
                break
    verdict = next(iter(suggestions)) if len(suggestions) == 1 and not issues and not conflict else 'inconclusivo'
    if len(suggestions) > 1 or conflict:
        issues.append('Foram identificadas evidências conflitantes.')
    if not suggestions:
        issues.append('Não foram encontradas passagens suficientes para sustentar uma classificação.')
    explanation = ('Sugestão preliminar com base no texto extraído das fontes e nas passagens citadas. '
                   'A leitura automatizada não substitui a revisão humana.')
    if issues:
        explanation = ' '.join(dict.fromkeys(issues)) + ' A ausência de evidências não comprova falsidade.'
    unique = []
    for citation in citations:
        if not any(item['source_id'] == citation['source_id'] and item['quote'] == citation['quote'] for item in unique):
            unique.append(citation)
    # Keep the decisive references visible when many blocks produced citations.
    if verdict in {'fato', 'fake'}:
        expected = 'supports' if verdict == 'fato' else 'contradicts'
        unique.sort(key=lambda item: item['relation'] != expected)
    else:
        first_per_relation = [next((item for item in unique if item['relation'] == relation), None)
                              for relation in ('supports', 'contradicts')]
        first_per_relation = [item for item in first_per_relation if item is not None]
        unique = first_per_relation + [item for item in unique if item not in first_per_relation]
    return {'verdict': verdict, 'explanation': explanation, 'citations': unique[:8],
            'complete': bool(documents) and all(doc['status'] == 'ok' and doc.get('analysis_complete') for doc in documents), 'basis': 'full_text',
            'assessed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'model': model.path.name}
