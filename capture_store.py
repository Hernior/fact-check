"""SQLite-backed PCM queue: acknowledge only committed audio, never discard backlog."""
import asyncio
import json
from contextlib import closing


def schema(database):
    database.executescript('''
        CREATE TABLE IF NOT EXISTS capture_sessions(
          id TEXT PRIMARY KEY, started TEXT, ended TEXT, source TEXT,
          participants TEXT, archived INTEGER DEFAULT 0, status TEXT DEFAULT 'capturing');
        CREATE TABLE IF NOT EXISTS capture_audio(
          id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
          pcm BLOB NOT NULL, samples INTEGER NOT NULL, offset_samples INTEGER NOT NULL,
          status TEXT DEFAULT 'pending', segment_id INTEGER, error TEXT,
          UNIQUE(session_id, sequence));
        CREATE INDEX IF NOT EXISTS capture_audio_queue ON capture_audio(session_id, status, sequence);
    ''')
    for table in ('segments', 'claims', 'transcript_paragraphs'):
        if 'session_id' not in {row[1] for row in database.execute(f'PRAGMA table_info({table})')}:
            database.execute(f'ALTER TABLE {table} ADD COLUMN session_id TEXT')
        database.execute(f'CREATE INDEX IF NOT EXISTS {table}_session ON {table}(session_id, id)')


def new_session(connection, identifier, started, source, participants):
    with closing(connection()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        # Preserve pre-session data as a discoverable archive, without rewriting content.
        if any(db.execute(f'SELECT 1 FROM {table} WHERE session_id IS NULL LIMIT 1').fetchone()
               for table in ('segments', 'claims', 'transcript_paragraphs')):
            db.execute("INSERT OR IGNORE INTO capture_sessions VALUES ('legacy', ?, ?, 'legacy', '[]', 1, 'complete')",
                       (started, started))
            for table in ('segments', 'claims', 'transcript_paragraphs'):
                db.execute(f"UPDATE {table} SET session_id='legacy' WHERE session_id IS NULL")
        db.execute('UPDATE capture_sessions SET archived=1')
        db.execute('INSERT INTO capture_sessions(id,started,source,participants) VALUES (?,?,?,?)',
                   (identifier, started, source, json.dumps(participants, ensure_ascii=False)))


class DiskCapture:
    durable = True

    def __init__(self, connection, identifier):
        self.connection, self.identifier = connection, identifier
        self.aborting = False

    def append(self, audio, sequence=None):
        with closing(self.connection()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            session = db.execute('SELECT ended FROM capture_sessions WHERE id=?', (self.identifier,)).fetchone()
            if session is None or session['ended']:
                raise ValueError('A sessão de captura foi encerrada.')
            last = db.execute('SELECT sequence,offset_samples FROM capture_audio WHERE session_id=? ORDER BY sequence DESC LIMIT 1',
                              (self.identifier,)).fetchone()
            expected = last['sequence'] + 1 if last else 0
            sequence = expected if sequence is None else sequence
            existing = db.execute('SELECT pcm FROM capture_audio WHERE session_id=? AND sequence=?',
                                  (self.identifier, sequence)).fetchone()
            if existing:
                if existing['pcm'] != audio:
                    raise ValueError('Sequência de áudio repetida com conteúdo diferente.')
                return
            if sequence != expected:
                raise ValueError('Sequência de áudio fora de ordem; nenhum bloco foi descartado.')
            samples = len(audio) // 2
            db.execute('INSERT INTO capture_audio(session_id,sequence,pcm,samples,offset_samples) VALUES (?,?,?,?,?)',
                       (self.identifier, sequence, audio, samples, (last['offset_samples'] if last else 0) + samples))

    def close(self, at):
        with closing(self.connection()) as db, db:
            db.execute("UPDATE capture_sessions SET ended=COALESCE(ended,?),status='processing' WHERE id=? AND status!='complete'",
                       (at, self.identifier))

    def pending(self):
        with closing(self.connection()) as db:
            return db.execute("SELECT * FROM capture_audio WHERE session_id=? AND status='pending' ORDER BY sequence LIMIT 1",
                              (self.identifier,)).fetchone()

    def closed(self):
        with closing(self.connection()) as db:
            return bool(db.execute('SELECT ended FROM capture_sessions WHERE id=?', (self.identifier,)).fetchone()['ended'])

    def empty(self):
        with closing(self.connection()) as db:
            return not db.execute("SELECT 1 FROM capture_audio WHERE session_id=? AND status IN ('pending','transcribed') LIMIT 1",
                                  (self.identifier,)).fetchone()

    async def get(self):
        while not self.aborting:
            with closing(self.connection()) as db, db:
                row = db.execute('''SELECT a.id AS chunk_id,a.offset_samples,a.samples,s.* FROM capture_audio a
                    JOIN segments s ON s.id=a.segment_id
                    WHERE a.session_id=? AND a.status='transcribed' ORDER BY a.sequence LIMIT 1''',
                                 (self.identifier,)).fetchone()
                if row:
                    db.execute("UPDATE capture_audio SET status='analyzing' WHERE id=?", (row['chunk_id'],))
                    data = dict(row)
                    data['start_s'] = (data['offset_samples'] - data['samples']) / 16000
                    return data
            if self.closed() and self.empty():
                return None
            await asyncio.sleep(0.05)
        raise asyncio.CancelledError

    def progress(self):
        with closing(self.connection()) as db:
            row = db.execute('''SELECT COUNT(*) AS received,
                COALESCE(SUM(samples),0) AS received_samples,
                COALESCE(SUM(CASE WHEN segment_id IS NOT NULL THEN samples ELSE 0 END),0) AS transcribed_samples,
                COALESCE(SUM(CASE WHEN status='done' THEN samples ELSE 0 END),0) AS analyzed_samples,
                COALESCE(SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),0) AS failed,
                COALESCE(SUM(CASE WHEN status NOT IN ('done','failed') THEN 1 ELSE 0 END),0) AS pending
                FROM capture_audio WHERE session_id=?''', (self.identifier,)).fetchone()
        data = dict(row)
        for key in ('received', 'transcribed', 'analyzed'):
            data[key + '_seconds'] = round(data[key + '_samples'] / 16000, 2)
            del data[key + '_samples']
        data['delay_seconds'] = round(data['received_seconds'] - data['analyzed_seconds'], 2)
        return data
