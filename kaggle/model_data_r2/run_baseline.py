"""Initial pinned P12 versus Qwen2.5 comparison. No training in this job."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = time.time()
DEADLINE = START + 7200 - 900
ROOT = Path('/kaggle/working/tabcomplete')
OUT = Path('/kaggle/working/model_data_r2_baseline')
COMMIT = '__CHECKOUT_COMMIT__'


def run(args, name, env=None):
    remaining = DEADLINE - time.time()
    if remaining < 30:
        raise TimeoutError('session finalization reserve reached')
    p = subprocess.run(args, cwd=ROOT if ROOT.exists() else None, env={**os.environ, **(env or {})},
                       capture_output=True, text=True, timeout=remaining)
    (OUT / (name + '.log')).write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(name + ' failed; inspect private job log')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20), b''):h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run([sys.executable,'-m','pip','install','-q','transformers==5.5.0','accelerate==1.13.0',
         'opentelemetry-api==1.44.0','opentelemetry-sdk==1.44.0','opentelemetry-exporter-otlp-proto-http==1.44.0',
         'tree-sitter==0.25.2','tree-sitter-language-pack','pydantic','httpx','pyyaml'], 'setup')
    run(['git','clone','--branch','research/model-data-r2','https://github.com/Shlok-Bhakta/tabcomplete.git',str(ROOT)],'clone')
    run(['git','checkout',COMMIT],'checkout')
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from tinycomplete.eval.code_benchmark import load_suite
    from tinycomplete.eval.code_generation import TransformersGenerationProvider, build_prediction_run_metadata, generate_predictions, build_causal_prompt
    from tinycomplete.observability.bootstrap import current_runtime
    env = {'PYTHONPATH':str(ROOT/'src'), 'TABCOMPLETE_OBSERVABILITY_ENABLED':'1',
           'TABCOMPLETE_OBSERVABILITY_MODE':'offline', 'TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT':'1',
           'TABCOMPLETE_OBSERVABILITY_OFFLINE_BUNDLE':str(OUT/'telemetry.jsonl'),
           'TABCOMPLETE_OBSERVABILITY_ARTIFACT_ROOT':str(OUT/'captured'),
           'TOKENIZERS_PARALLELISM':'false', 'TABCOMPLETE_CAMPAIGN_ID':'tabcomplete-model-data-r2'}
    os.environ.update(env)
    matches=list(Path('/kaggle/input').glob('**/P12/model.safetensors'))
    if len(matches)!=1:raise RuntimeError('ambiguous P12 input')
    p12=matches[0].parent
    assert sha(p12/'model.safetensors')=='d4d3fdb8d30ae0f3e4a1342a3d10ead7e0a4363e0f8ca406a8267c726316ac43'
    q25=Path(snapshot_download('Qwen/Qwen2.5-Coder-0.5B',revision='8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']))
    suite=ROOT/'data/benchmarks/code_completion_v2.jsonl'
    assert sha(suite)=='ed28739b302e4c0d3f9f45e859e7ccd68a1e8594a62eb7b13ee8425b92a439a4'
    cases=load_suite(suite)
    class BoundedProvider(TransformersGenerationProvider):
        def generate_detailed(self, prompt, max_new_tokens):
            if time.time() > DEADLINE:
                raise TimeoutError('finalization reserve reached before next request')
            return super().generate_detailed(prompt, max_new_tokens)
    progress={'schema_version':1,'git_sha':COMMIT,'plan_sha256':sha(ROOT/'reports/research/model_data_r2/preregistered_plan.json'),
              'started_unix':START,'status':'running','models':{},'torch':torch.__version__,'transformers':transformers.__version__,
              'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'training_input_tokens':0}
    def save():
        progress['elapsed_seconds']=time.time()-START
        (OUT/'progress.json').write_text(json.dumps(progress,indent=2))
    save()
    for alias,path in [('q35-p12',p12),('q25-coder',q25)]:
        if time.time()>DEADLINE-300:break
        provider=BoundedProvider(str(path),device='cuda:0')
        metadata=build_prediction_run_metadata(suite_path=suite,case_count=len(cases),provider='transformers',model_source=alias,
                    model_revision=sha(path/'model.safetensors'),max_new_tokens=96,workers=1)
        metadata.update(tokenizer_sha256=sha(path/'tokenizer.json'),campaign_id='tabcomplete-model-data-r2',
                        plan_sha256=progress['plan_sha256'],runtime={'torch':torch.__version__,'transformers':transformers.__version__,'precision':'fp16','device':torch.cuda.get_device_name(0)})
        before=time.time()
        predictions=generate_predictions(cases,provider,OUT/(alias+'.jsonl'),run_metadata=metadata,max_new_tokens=96,workers=1)
        sizes=[{'case_id':c.id,'input_tokens':len(provider.tokenizer.encode(build_causal_prompt(c))),
                'output_tokens':p.generated_tokens,'output_characters':len(p.completion),'output_bytes':len(p.completion.encode()),
                'hit_cap':p.hit_token_cap} for c,p in zip(cases,predictions)]
        (OUT/(alias+'-lengths.json')).write_text(json.dumps(sizes,indent=2))
        progress['models'][alias]={'status':'complete','cases':len(predictions),'seconds':time.time()-before,
             'model_class':type(provider.model).__name__,'parameters':sum(p.numel() for p in provider.model.parameters()),
             'vocabulary':len(provider.tokenizer),'tokenizer_class':type(provider.tokenizer).__name__,
             'model_sha256':sha(path/'model.safetensors'),'predictions_sha256':sha(OUT/(alias+'.jsonl'))}
        save()
        del provider
        import gc
        gc.collect();torch.cuda.empty_cache()
    progress['status']='complete' if len(progress['models'])==2 else 'partial'
    save()
    current_runtime().shutdown()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'failure.json').write_text(json.dumps({'type':type(e).__name__,'elapsed_seconds':time.time()-START}))
        raise
