"""Regression tests for real measurements, isolation, immutable input, and resume."""
from pathlib import Path
import json
import math
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app
from backend.models import FileRecord, DocumentRecord, DatasetVersion, TokenizerRecord
from backend.services.training_service import TrainingService
from src.dataset.compiler import DatasetCompiler
from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.loading import artifact_hash
from src.registry.model_registry import ModelRegistry
from src.evaluation.benchmarks import BenchmarkRunner


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr('backend.services.training_service.SessionLocal', factory)
    def dependency():
        with factory() as db:
            yield db
    old = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = dependency
    yield factory
    app.dependency_overrides.clear()
    app.dependency_overrides.update(old)
    engine.dispose()


def account(client, name):
    assert client.post('/api/v1/auth/register', json={'username':name,'email':name+'@example.test','password':'test-password-123'}).status_code == 201
    response = client.post('/api/v1/auth/login', json={'username_or_email':name,'password':'test-password-123'})
    assert response.status_code == 200
    return response.json()


def test_auth_isolation_refresh_and_logout(isolated):
    a, b = TestClient(app), TestClient(app)
    assert a.get('/api/v1/files/').status_code == 401
    tokens_a = account(a,'alice')
    tokens_b = account(b,'bob')
    # Insert through ORM under each request's scope by calling upload.
    ra = a.post('/api/v1/files/upload', files={'file':('alice.txt',b'Private Alice document','text/plain')})
    assert ra.status_code == 201, ra.text
    file_id = ra.json()['file_id']
    assert a.get(f'/api/v1/files/{file_id}').status_code == 200
    assert b.get(f'/api/v1/files/{file_id}').status_code == 404
    assert b.patch(f'/api/v1/files/{file_id}',json={'training_allowed':True}).status_code == 404
    assert b.delete(f'/api/v1/files/{file_id}').status_code == 404
    # Refresh tokens must not authorize normal resource access.
    c = TestClient(app)
    assert c.get('/api/v1/files/',headers={'Authorization':'Bearer '+tokens_a['refresh_token']}).status_code == 401
    assert a.post('/api/v1/auth/refresh',json={}).status_code == 200
    assert c.post('/api/v1/auth/refresh',json={'refresh_token':tokens_a['refresh_token']}).status_code == 401
    assert a.post('/api/v1/auth/logout').status_code == 200
    assert a.get('/api/v1/files/').status_code == 401


def test_pii_masking_preserves_source(isolated,tmp_path):
    with isolated() as db:
        file = FileRecord(file_id='f',original_name='x.txt',relative_path='x',mime_type='text/plain',size_bytes=1,sha256='a'*64,training_allowed=True)
        doc = DocumentRecord(document_id='d',file=file,text='Contact user@example.com for details',char_count=36,word_count=5)
        db.add(file); db.commit()
        compiler = DatasetCompiler(tmp_path/'compiled', BPETokenizer())
        result = compiler._mask_pii_in_documents([doc])
        assert result[0].text != doc.text
        db.commit(); db.expire_all()
        assert db.get(DocumentRecord,'d').text == 'Contact user@example.com for details'


def test_no_model_never_produces_benchmark_success(isolated):
    runner = BenchmarkRunner('missing')
    with patch.object(runner.registry,'load_model',return_value={'checkpoint_path':None,'tokenizer_path':None}):
        for name in ['bleu','rouge','gsm8k_cot','turkish_knowledge','turkish_summarization','turkish_qa']:
            with pytest.raises(ValueError):
                runner.run_benchmark(name)
    with pytest.raises(ValueError,match='explicit held-out'):
        runner.run_benchmark('perplexity')


def make_artifacts(factory, path):
    from src.dataset.splits import split_for_text
    texts = [f'Türkçe eğitim örneği {i}: modeller veriden öğrenir ve sonuçlar ölçülür.' for i in range(60)]
    tok = BPETokenizer(vocab_size=90,min_frequency=2)
    tok.train([t for t in texts if split_for_text(t)=='train'])
    tokpath = path/'tokenizer'
    tok.save_vocab(tokpath)
    with factory() as db:
        f = FileRecord(file_id='f',original_name='corpus.txt',relative_path='corpus',mime_type='text/plain',size_bytes=1,sha256='b'*64,training_allowed=True)
        docs = [DocumentRecord(document_id=f'd{i}',file=f,text=t,quality_score=1,char_count=len(t),word_count=len(t.split())) for i,t in enumerate(texts)]
        db.add(f); db.add_all(docs); db.commit()
        compiler=DatasetCompiler(path/'compiled',tok,db_session=db)
        result=compiler.compile_dataset(docs,remove_duplicates=False,allow_pii=True)
        hashes={'dataset_sha256':artifact_hash(result['output_path']),'tokenizer_sha256':artifact_hash(tokpath),'split_strategy':'content_sha256_80_10_10'}
        db.add(TokenizerRecord(tokenizer_id='tok',name='tok',vocab_size=len(tok.vocab),storage_path=str(tokpath)))
        db.add(DatasetVersion(dataset_id='ds',name='ds',version='1',compiler_version='1',storage_path=result['output_path'],tokenizer_id='tok',source_file_ids=['f'],num_documents=len(docs),custom_metadata=hashes))
        db.commit()
    return result['output_path']


def test_real_training_reload_perplexity_and_resume(isolated,tmp_path):
    torch.set_num_threads(1)
    dataset_path=make_artifacts(isolated,tmp_path)
    config={'d_model':16,'n_heads':2,'n_layers':1,'d_ff':32,'max_seq_len':16,'batch_size':8,'epochs':1,'lr':0.001,'dropout':0.1,'version':'1.0.0','seed':42}
    with isolated() as db:
        service=TrainingService(db)
        job=service.create_job('tiny','tiny','PRETRAIN','ds','tok',config)
        job_id=job.job_id
    TrainingService._run_training_worker(job_id)
    with isolated() as db:
        from backend.models import TrainingJob
        job=db.get(TrainingJob,job_id)
        assert job.status=='COMPLETED',job.error
        assert job.metrics[-1]['val_loss'] > 0
        assert (Path(job.output_dir)/'resume.pt').exists()
    result=BenchmarkRunner('tiny').run_benchmark('perplexity',dataset_path,max_samples=100)
    assert math.isfinite(result.score) and result.score > 0
    assert 0 < result.samples_evaluated < 100
    assert result.metrics['tokens_evaluated'] > 0
    from src.inference.pipeline import InferencePipeline
    info=ModelRegistry('models').load_model('tiny','1.0.0')
    pipeline=InferencePipeline.from_pretrained(info['checkpoint_path'],info['tokenizer_path'],device='cpu')
    assert isinstance(pipeline.generate('Türkçe',max_new_tokens=2),str)
    # Pause after two batches; resume must reproduce uninterrupted CPU weights.
    class StopAfterTwo:
        count=0
        def is_set(self):
            self.count+=1
            return self.count>2
    with isolated() as db:
        second=TrainingService(db).create_job('resume','resumed','PRETRAIN','ds','tok',config)
        second_id=second.job_id
    TrainingService._run_training_worker(second_id,StopAfterTwo())
    with isolated() as db:
        second=db.get(TrainingJob,second_id)
        assert second.status=='CANCELLED'
        second.config={**second.config,'resume':True,'cancel_requested':False}
        db.commit()
    TrainingService._run_training_worker(second_id)
    resumed=ModelRegistry('models').load_model('resumed','1.0.0',load_weights=True)
    original=ModelRegistry('models').load_model('tiny','1.0.0',load_weights=True)
    for name, value in original['state_dict'].items():
        torch.testing.assert_close(value,resumed['state_dict'][name],rtol=0,atol=0)


def test_training_requires_permissions_and_matching_artifacts(isolated,tmp_path):
    make_artifacts(isolated,tmp_path)
    with isolated() as db:
        service=TrainingService(db)
        with pytest.raises(ValueError):
            service.create_job('bad','bad')
        file=db.get(FileRecord,'f'); file.training_allowed=False; db.commit()
        with pytest.raises(ValueError,match='permission'):
            service.create_job('bad','bad',dataset_id='ds',tokenizer_id='tok')
