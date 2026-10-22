"""Private set-by-set results backup; checksum-verified upload precedes cache pruning.

No model calls. Original result bytes and Git-tracked files are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'HawkFranklin-Research/mlcagent-runs'
EXCLUDE_DIRS = {'.git', '.control', '.work', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv'}
PRIVATE_DIRS = {'data', 'env', 'private', 'vault', 'keys'}
SECRET_PATTERNS = {
    'hf_token': rb'(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}',
    'github_token': rb'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'api_key': rb'(?<![A-Za-z0-9])sk-(?:or-v1-|proj-|ant-api03-)?[A-Za-z0-9_-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'aws_key': rb'(?<![A-Za-z0-9])AKIA[A-Z0-9]{16}(?![A-Za-z0-9])',
}
PATTERNS = {k: re.compile(v) for k,v in SECRET_PATTERNS.items()}
# Known configured secrets are checked without persisting names, values or hashes.
SECRETS = [v.encode() for k,v in os.environ.items() if len(v)>=16 and
           re.search(r'(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|HF_TOKEN|GITHUB_TOKEN)$', k)]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_bytes(data, name):
    for category, pattern in PATTERNS.items():
        if pattern.search(data):
            raise ValueError(f'Secret-pattern refusal: {category}; file {name}; matched value withheld')
    if any(v in data for v in SECRETS):
        raise ValueError(f'Configured-secret refusal; file {name}; matched value withheld')


def scan_hash(path):
    h=hashlib.sha256();tail=b''
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk);check_bytes(tail+chunk,str(path.relative_to(ROOT)));tail=chunk[-8192:]
    return h.hexdigest()


def sets():
    roots=[p for p in sorted((ROOT/'bench/dev-loop').glob('batch-*')) if p.is_dir() and (p/'batch.json').exists()]
    roots += [ROOT/'bench/comparison/acceptance',ROOT/'bench/comparison/acceptance-02',
              ROOT/'bench/heldout/heldout-01',ROOT/'bio-bench/runs/deepseek-flash/eval-01']
    roots += [p for p in sorted((ROOT/'bench/dev-loop-v2').glob('batch-*')) if p.is_dir() and (p/'batch.json').exists()]
    roots += [ROOT/'bench/automl',ROOT/'bench/automl-heldout']
    for p in roots:
        if not p.is_dir():raise ValueError(f'Missing set: {p}')
        if (p/'batch.json').exists():
            batch=json.loads((p/'batch.json').read_text())
            if batch.get('status')!='finished':raise ValueError(f'Set not finished: {p}')
    return roots


def excluded_file(name):
    return (name.endswith(('.local.md','.local.yaml','.pyc','.pyo','.key','.pem','.incomplete'))
            or name=='.env' or name.startswith('.env.')
            or (name.endswith('.lock') and name not in {'requirements.lock','uv.lock'}))


def inventory(root):
    files=[];omitted=[];prunable=[]
    for dp,dirs,names in os.walk(root,followlinks=False):
        parent=Path(dp)
        for d in list(dirs):
            p=parent/d
            if p.is_symlink() or d in EXCLUDE_DIRS or (parent.name=='.mlca' and d in PRIVATE_DIRS):
                dirs.remove(d)
                omitted.append({'path':str(p.relative_to(ROOT)),'reason':'symlink' if p.is_symlink() else 'excluded_directory'})
                if parent.name=='.mlca' and d in {'data','env'} and not p.is_symlink():
                    prunable.append(p)
                    # Membership is metadata, not a feature table. Relocate it within the archive.
                    if d=='data':
                        for member in sorted(p.rglob('split_manifest.csv')):
                            if member.is_file() and not member.is_symlink():
                                files.append((member,'offload-metadata/'+str(member.relative_to(ROOT))))
                continue
        for name in sorted(names):
            p=parent/name
            if p.is_symlink() or not p.is_file() or excluded_file(name):
                omitted.append({'path':str(p.relative_to(ROOT)),'reason':'symlink_or_excluded_file'});continue
            files.append((p,str(p.relative_to(ROOT))))
    return sorted(files,key=lambda x:x[1]),omitted,prunable


def archive(root,stage):
    rel=str(root.relative_to(ROOT));ident=rel.replace('/','__')
    output=stage/'archives'/f'{ident}.tar.zst';manifest=stage/'manifests'/f'{ident}.csv'
    output.parent.mkdir(parents=True,exist_ok=True);manifest.parent.mkdir(parents=True,exist_ok=True)
    files,omitted,prunable=inventory(root)
    rows=[];before={}
    for p,name in files:
        st=p.stat();before[str(p)]=(st.st_size,st.st_mtime_ns)
        rows.append({'archive_path':name,'source_path':str(p.relative_to(ROOT)),'bytes':st.st_size,'sha256':scan_hash(p)})
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,['archive_path','source_path','bytes','sha256']);w.writeheader();w.writerows(rows)
    packed=io.BytesIO(manifest.read_bytes())
    with output.open('wb') as target:
        compressor=subprocess.Popen(['zstd','-q','-T2','-3','-c'],stdin=subprocess.PIPE,stdout=target)
        try:
            with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                info=tarfile.TarInfo('OFFLOAD_MANIFEST.csv');info.size=len(packed.getbuffer());info.mode=0o644
                tar.addfile(info,packed)
                for p,name in files:
                    if (p.stat().st_size,p.stat().st_mtime_ns)!=before[str(p)]:raise ValueError(f'Source changed: {name}')
                    info=tar.gettarinfo(str(p),arcname=name);info.uid=info.gid=0;info.uname=info.gname=''
                    with p.open('rb') as source:tar.addfile(info,source)
            compressor.stdin.close()
            if compressor.wait()!=0:raise ValueError('Compression failed')
        except BaseException:
            compressor.kill();compressor.wait();raise
    # Independently decompress and compare every member to the source manifest.
    expected={r['archive_path']:r for r in rows};verified=set()
    reader=subprocess.Popen(['zstd','-q','-d','-c',str(output)],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=reader.stdout,mode='r|') as tar:
        for member in tar:
            if not member.isfile():raise ValueError('Unexpected non-regular archive member')
            body=tar.extractfile(member);h=hashlib.sha256()
            for chunk in iter(lambda:body.read(1024*1024),b''):h.update(chunk)
            if member.name=='OFFLOAD_MANIFEST.csv':
                if h.hexdigest()!=sha(manifest):raise ValueError('Embedded manifest differs')
            else:
                row=expected[member.name]
                if member.size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Archive content differs')
                verified.add(member.name)
    if reader.wait()!=0 or verified!=set(expected):raise ValueError('Archive incomplete')
    for p,name in files:
        if (p.stat().st_size,p.stat().st_mtime_ns)!=before[str(p)]:raise ValueError(f'Source changed during packaging: {name}')
    save(stage/'omissions'/f'{ident}.json',omitted)
    return {'set':rel,'archive':str(output.relative_to(stage)),'archive_sha256':sha(output),
            'archive_bytes':output.stat().st_size,'files':len(rows),'uncompressed_bytes':sum(r['bytes'] for r in rows),
            'manifest':str(manifest.relative_to(stage)),'prunable':[str(p.relative_to(ROOT)) for p in prunable],
            'batch_sha256':sha(root/'batch.json') if (root/'batch.json').exists() else None}


def support(stage):
    wanted=['RESULTS.md','PREREGISTRATION.md','AGENTS.md','scripts','automl/README.md','automl/requirements.in','automl/requirements.lock','automl/specs','bio-bench/manifest.yaml','bio-bench/briefs']
    for name in wanted:
        src=ROOT/name
        if not src.exists():continue
        paths=sorted(src.rglob('*')) if src.is_dir() else [src]
        for p in paths:
            if not p.is_file() or p.is_symlink() or excluded_file(p.name) or '__pycache__' in p.parts:continue
            scan_hash(p);dst=stage/'support'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    code=ROOT.parent/'mlca-revized'
    for name in ['benchmarks/manifest.yaml','benchmarks/datasets/SHA256SUMS.txt','rails/LEDGER.md','rails/VERSION','uv.lock','pyproject.toml']:
        p=code/name
        if p.is_file():
            check_bytes(p.read_bytes(),name);dst=stage/'support/mlca-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for p in sorted((code/'rails').glob('RL-*.yaml')):
        dst=stage/'support/mlca-source/rails'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)


def prepare(stage):
    stage.mkdir(parents=True,exist_ok=False)
    roots=sets();records=[]
    for root in roots:
        print('Packaging '+str(root.relative_to(ROOT)),flush=True)
        records.append(archive(root,stage));save(stage/'PACKAGING_INDEX.json',{'created_at':now(),'repo_id':REPO_ID,'sets':records})
        print('Verified '+records[-1]['archive']+' ('+str(records[-1]['files'])+' files)',flush=True)
    support(stage)
    card=(ROOT/'RESULTS.md').read_text()
    card=card.replace('Nothing here is pushed or uploaded yet.','A private archival snapshot is uploaded with checksummed per-set archives. See PACKAGING_INDEX.json and ARCHIVE_POLICY.md.')
    card=card.replace('**Raw folders** stay local until uploaded.', '**Raw results** are retained in private per-set archives; local cleanup is recorded separately.')
    (stage/'README.md').write_text('---\npretty_name: MLCA agent run artifacts\nlanguage:\n- en\ntags:\n- agent-evaluation\n- machine-learning\n- research-artifacts\n---\n\n'+card)
    policy='''# Archive and restoration policy

Private storage snapshot authorized by the author. Contains the 194 main-study
MLCA jobs across 18 sets, plus two AutoML result collections. Exploratory rounds,
smoke/debugging folders, the cancelled reference analyses, source input tables,
private/vault/key stores, Python environments and controller control files are
excluded. No dataset was reanalysed and no model was called for this backup.

Each .tar.zst contains original result paths and OFFLOAD_MANIFEST.csv with every
member's source path, size and SHA-256. Matching CSVs are under manifests/.
PACKAGING_INDEX.json records each set's source and archive identity. Some earlier
sets also contain curated copies: they are retained, not counted as extra runs.
Derived per-row outputs and fitted models are raw results and may be included;
this is a private archive, not an anonymized public dataset or a confidentiality
certification. Do not load untrusted pickle/model files without an isolated environment.

Split-membership CSV metadata originally under .mlca/data is preserved instead
under offload-metadata/<original path>. Other .mlca/data files and .mlca/env are
regenerable and not archived. Inspect omissions/ for excluded paths. Input sources
and checksums are documented in support/. The biomedical input release remains
HawkFranklin-Research/mlcagent-bio-bench at its pinned revision; it is not duplicated.
These archives alone do not make a runnable native project: input partitions,
environments and any excluded vault material must be reconstructed separately.
Hash-chained logs remain byte-identical; their original artifact references are
not rewritten when excluded inputs are absent.

To inspect: zstd -t ARCHIVE.tar.zst; tar --zstd -tf ARCHIVE.tar.zst
To restore: verify SHA256SUMS, then tar --zstd -xf ARCHIVE.tar.zst -C EMPTY_DIRECTORY.
Restore any required membership metadata to its original path separately.
Do not overwrite active studies. Checksums verify preservation, not scientific correctness.

Secret scan: every included source file is scanned for recognized HF/GitHub/API
credentials, private-key headers and configured environment secrets, with overlap
across chunks. Findings stop packaging; matching values are never logged.
This heuristic cannot prove absence of every unknown or obfuscated secret.
Original RESULTs and reports are archival snapshots; no historical counts are altered.
'''
    (stage/'ARCHIVE_POLICY.md').write_text(policy)
    for p in [stage/'README.md',stage/'ARCHIVE_POLICY.md']:
        check_bytes(p.read_bytes(),p.name)
    save(stage/'SECRET_SCAN.json',{'status':'passed','timestamp':now(),'included_files':sum(r['files'] for r in records),'methods':list(PATTERNS)+['configured_environment_secrets'],'matched_values_recorded':False,'limitations':'Heuristic; unknown or encoded credentials may not be recognized.'})
    files=[p for p in sorted(stage.rglob('*')) if p.is_file()]
    with (stage/'SHA256SUMS').open('w') as f:
        for p in files:f.write(sha(p)+'  '+str(p.relative_to(stage))+'\n')
    return records


def publish(stage):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_url, get_token
    from huggingface_hub.utils import get_session
    api=HfApi();api.create_repo(REPO_ID,repo_type='dataset',private=True,exist_ok=True)
    if not api.repo_info(REPO_ID,repo_type='dataset').private:raise ValueError('Destination is not private')
    expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (stage/'SHA256SUMS').read_text().splitlines()}
    expected['SHA256SUMS']=sha(stage/'SHA256SUMS')
    remote={f.path:getattr(f,'lfs',None) for f in api.list_repo_tree(REPO_ID,repo_type='dataset',recursive=True) if hasattr(f,'path')}
    # Upload archives separately; batch the small documentation/manifest files.
    metadata=[]
    for name in sorted(expected, key=lambda n:(not n.startswith('archives/'),n)):
        p=stage/name
        if sha(p)!=expected[name]:raise ValueError('Staged file changed')
        if name.startswith('archives/'):
            if getattr(remote.get(name),'sha256',None)==expected[name]:
                print('Already uploaded '+name,flush=True)
                continue
            api.upload_file(path_or_fileobj=p,path_in_repo=name,repo_id=REPO_ID,repo_type='dataset',commit_message='Archive '+name)
            print('Uploaded '+name,flush=True)
        else:
            metadata.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=p))
    api.create_commit(REPO_ID,repo_type='dataset',operations=metadata,commit_message='Add dataset card, manifests, checksums and protocol snapshots')
    print('Uploaded documentation and manifests',flush=True)
    info=api.repo_info(REPO_ID,repo_type='dataset');revision=info.sha
    verified=[];token=get_token()
    for name,digest in expected.items():
        url=hf_hub_url(REPO_ID,name,repo_type='dataset',revision=revision)
        h=hashlib.sha256();size=0
        with get_session().stream('GET',url,headers={'Authorization':'Bearer '+token} if token else {},follow_redirects=True,timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=1024*1024):h.update(chunk);size+=len(chunk)
        if h.hexdigest()!=digest:raise ValueError('Remote checksum mismatch: '+name)
        verified.append({'path':name,'sha256':digest,'bytes':size})
        if name.startswith('archives/'):print('Remote SHA-256 verified '+name,flush=True)
    receipt={'repo_id':REPO_ID,'private':info.private,'revision':revision,'verified_at':now(),'files':verified,'status':'all_remote_bytes_verified'}
    save(stage.parent/(stage.name+'-REMOTE_VERIFIED.json'),receipt)
    return receipt


def prune(stage):
    receipt_path=stage.parent/(stage.name+'-REMOTE_VERIFIED.json')
    receipt=json.loads(receipt_path.read_text());index=json.loads((stage/'PACKAGING_INDEX.json').read_text())
    if receipt['status']!='all_remote_bytes_verified' or receipt['private'] is not True:raise ValueError('Missing verified private backup')
    from huggingface_hub import HfApi
    if not HfApi().repo_info(REPO_ID,repo_type='dataset',revision=receipt['revision']).private:raise ValueError('Remote visibility changed')
    verified={r['path']:r['sha256'] for r in receipt['files']}
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    targets=[]
    for entry in index['sets']:
        if verified.get(entry['archive'])!=entry['archive_sha256']:raise ValueError('Set archive not verified')
        source=ROOT/entry['set']
        if entry['batch_sha256'] is not None:
            batch=json.loads((source/'batch.json').read_text())
            if batch['status']!='finished' or sha(source/'batch.json')!=entry['batch_sha256']:raise ValueError('Batch changed after packaging')
        for name in entry['prunable']:
            p=ROOT/name
            if p.is_symlink() or not p.exists():continue
            if p.name not in {'data','env'} or p.parent.name!='.mlca':raise ValueError('Unsafe prune target')
            if ROOT not in p.resolve().parents:raise ValueError('Prune escapes repository')
            if any(t==name or t.startswith(name+'/') for t in tracked):raise ValueError('Prune would remove tracked files')
            targets.append(p)
    before=shutil.disk_usage(ROOT).free;deleted=[]
    for p in targets:
        # Only verified completed-set regenerable copies are removed, never raw results.
        def onerror(fn,path,exc):
            os.chmod(path,stat.S_IRWXU);fn(path)
        shutil.rmtree(p,onerror=onerror);deleted.append(str(p.relative_to(ROOT)))
    result={'timestamp':now(),'repo_id':REPO_ID,'verified_revision':receipt['revision'],'removed_regenerable_directories':deleted,
            'disk_free_before_bytes':before,'disk_free_after_bytes':shutil.disk_usage(ROOT).free,
            'raw_results_removed':False,'original_inputs_removed':False,'tracked_files_removed':False}
    save(stage.parent/(stage.name+'-LOCAL_CLEANUP.json'),result)
    print(json.dumps({'pruned_directories':len(deleted),'free_gib':result['disk_free_after_bytes']/2**30,'recovered_gib':(result['disk_free_after_bytes']-before)/2**30}),flush=True)


def prune_raw(stage):
    """Remove only untracked result bytes already verified in the remote archives."""
    receipt=json.loads((stage.parent/(stage.name+'-REMOTE_VERIFIED.json')).read_text())
    index=json.loads((stage/'PACKAGING_INDEX.json').read_text())
    if receipt['status']!='all_remote_bytes_verified' or receipt['private'] is not True:
        raise ValueError('Missing verified private backup')
    from huggingface_hub import HfApi
    if not HfApi().repo_info(REPO_ID,repo_type='dataset',revision=receipt['revision']).private:
        raise ValueError('Remote visibility changed')
    remote={r['path']:r['sha256'] for r in receipt['files']}
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    candidates=[];already_absent=[]
    for entry in index['sets']:
        if remote.get(entry['archive'])!=entry['archive_sha256']:
            raise ValueError('Archive not verified')
        source=ROOT/entry['set']
        if entry['batch_sha256'] and sha(source/'batch.json')!=entry['batch_sha256']:
            raise ValueError('Batch changed')
        manifest=stage/entry['manifest']
        if sha(manifest)!=remote.get(entry['manifest']):
            raise ValueError('Manifest changed')
        with manifest.open(newline='') as f:
            for row in csv.DictReader(f):
                name=row['source_path'];p=ROOT/name
                if name in tracked:continue
                if not p.exists():
                    if '/.mlca/data/' not in name:already_absent.append(row)
                    continue
                if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(source.resolve()):
                    raise ValueError('Unsafe result cleanup target')
                if any(part in {'private','vault','keys'} for part in p.relative_to(ROOT).parts):
                    raise ValueError('Private material is never a raw-result cleanup target')
                if sha(p)!=row['sha256']:raise ValueError('Result changed after backup: '+name)
                st=p.stat()
                candidates.append((p,st.st_size,st.st_mtime_ns,row['sha256']))
    before=shutil.disk_usage(ROOT).free;deleted=[]
    result={'timestamp':now(),'repo_id':REPO_ID,'verified_revision':receipt['revision'],
            'removed_untracked_results':deleted,'already_absent_verified_untracked_results':already_absent,
            'disk_free_before_bytes':before,'disk_free_after_bytes':None,'tracked_files_removed':False,
            'original_inputs_removed':False,'private_or_vault_material_removed':False}
    log=stage.parent/(stage.name+'-RAW_CLEANUP.json')
    result['status']='in_progress';save(log,result)
    try:
        for p,size,mtime,digest in candidates:
            if (p.stat().st_size,p.stat().st_mtime_ns)!=(size,mtime):
                raise ValueError('Result changed during cleanup')
            parent_mode=stat.S_IMODE(p.parent.stat().st_mode)
            try:
                p.parent.chmod(parent_mode | stat.S_IWUSR)
                p.unlink()
            finally:
                p.parent.chmod(parent_mode)
            deleted.append({'path':str(p.relative_to(ROOT)),'bytes':size,'sha256':digest})
        result['status']='complete'
    finally:
        result['disk_free_after_bytes']=shutil.disk_usage(ROOT).free
        save(log,result)
    print(json.dumps({'removed_untracked_files':len(deleted),'removed_bytes':sum(r['bytes'] for r in deleted),
                     'recovered_gib':(result['disk_free_after_bytes']-before)/2**30}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['prepare','publish','prune','prune-raw']);parser.add_argument('--stage',type=Path,required=True);args=parser.parse_args()
    stage=args.stage.resolve()
    if args.operation=='prepare':prepare(stage)
    elif args.operation=='publish':publish(stage)
    elif args.operation=='prune':prune(stage)
    else:prune_raw(stage)


if __name__=='__main__':main()
