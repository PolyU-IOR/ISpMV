#!/usr/bin/env python3
"""Audit public SDK content and compile consumers; never execute GPU work."""
import argparse,hashlib,json,os,re,shutil,subprocess,tarfile,tempfile
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity():return json.loads((ROOT/'manifest.json').read_text())
def run(argv,**kw):return subprocess.run(list(map(str,argv)),check=True,**kw)
def check_repository():
    allowed={'.github','.gitattributes','.gitignore','LICENSE','LICENSE-BINARY.txt','README.md','manifest.json','benchmarks','cmake','docs','examples','include','licenses'}
    if (ROOT/'.git').exists():
        names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT,text=True).split('\0')
        files=[ROOT/n for n in set(names) if n and (ROOT/n).is_file()]
    else:files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts]
    secrets=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    private=re.compile(rb'(?:/home/|/Users/|[A-Z]:[\\/]project)',re.I)
    for p in files:
        rel=p.relative_to(ROOT);need(rel.parts[0] in allowed,'Unlisted public path: '+str(rel))
        need(p.suffix not in ('.cu','.cuh','.a','.o','.so','.ptx','.bundle','.gz'),'Implementation/binary in source repository: '+str(rel))
        data=p.read_bytes()
        images={'docs/images/overall-performance.png','docs/images/three-suites.png'}
        is_image=rel.as_posix() in images
        need(len(data)<(2 if is_image else 1)*1024*1024,'Oversized public file: '+str(rel))
        if p.suffix=='.png':need(is_image and data.startswith(b'\x89PNG\r\n\x1a\n'),'Unlisted/invalid documentation image: '+str(rel))
        # This audit script necessarily spells the patterns it rejects.
        if rel.as_posix()!='.github/check.py':
            need(not secrets.search(data),'Possible credential: '+str(rel));need(not private.search(data),'Private identifier/path: '+str(rel))
        if p.suffix=='.json':json.loads(data)
        if p.suffix=='.md':
            for link in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)',data.decode()):
                u=urlsplit(link)
                if not u.scheme and not u.netloc:
                    need((p.parent/unquote(u.path)).exists() if u.path else True,'Broken local link: '+str(rel))
    m=identity();h=(ROOT/'include/ispmv/ispmv.h').read_text()
    exports=set(re.findall(r'ISPMV_API\s+[^;]+?\b(ispmv_\w+)\s*\(',h))
    need(len(exports)==14 and exports==set(m['audit']['exports']),'Public export contract')
    need(m['version']=='0.1.0' and m['audit']['abi']==1,'Version contract')
    need(not re.search(r'prepare_pair|autotune_transpose|EXACT_COEFFICIENTS|reserved_extension|\baccuracy\b',h),'Removed public control')
    print(f'PUBLIC_AUDIT_PASS files={len(files)} exports={len(exports)}')
def check_headers():
    for compiler,lang,standard,header in [('cc','c','c11','ispmv.h'),('c++','c++','c++17','ispmv.hpp')]:
        run([compiler,'-x',lang,'-std='+standard,'-Wall','-Wextra','-Werror','-fsyntax-only','-I'+str(ROOT/'include'),'-'],input='#include <ispmv/'+header+'>\nint main(void){return ISPMV_ABI_VERSION!=1;}\n',text=True)
    print('PUBLIC_HEADERS_PASS')
def extract(archive,destination):
    checksum=archive.with_name(archive.name+'.sha256').read_text().split()
    need(checksum==[sha(archive),archive.name],'Archive checksum mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:t.extractall(destination,filter='data')
    print('ARCHIVE_VERIFIED_AND_EXTRACTED')
def check_sdk(sdk,runtime,toolkit):
    m=identity();need(json.loads((sdk/'manifest.json').read_text())==m,'SDK manifest mismatch')
    for line in (sdk/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);p=(sdk/name).resolve();need(p.is_relative_to(sdk.resolve()) and sha(p)==digest,'SDK checksum: '+name)
    lib=sdk/'lib/libispmv.so.0.1.0';need(sha(lib)==m['library_sha256'],'Library identity')
    need(b'ISPMV_DIAGNOSTIC_' not in lib.read_bytes(),'Diagnostic controls in production')
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
    need({l.split()[-1].split('@')[0] for l in symbols.splitlines()}==set(m['audit']['exports'])|{'ISPMV_SDK_1'},'ELF exports')
    dynamic=subprocess.check_output(['readelf','-dW',str(lib)],text=True)
    need('[libispmv.so.1]' in dynamic and 'RPATH' not in dynamic and 'RUNPATH' not in dynamic,'ELF SONAME/search path')
    need(set(re.findall(r'Shared library: \[(.*?)\]',dynamic))==set(m['audit']['dependencies']),'ELF dependencies')
    for folder in ['include','examples','benchmarks','docs','licenses']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():need(p.read_bytes()==(sdk/p.relative_to(ROOT)).read_bytes(),'Public/package mismatch: '+str(p.relative_to(ROOT)))
    for name in ['LICENSE','LICENSE-BINARY.txt','README.md']:
        need((ROOT/name).read_bytes()==(sdk/name).read_bytes(),'Public/package mismatch: '+name)
    for p in (ROOT/'cmake').glob('*.cmake'):need(p.read_bytes()==(sdk/'lib/cmake/ISpMV'/p.name).read_bytes(),'CMake mismatch')
    cudart=next((p for sub in ['lib','lib64'] for p in (runtime/sub).glob('libcudart.so.13') if p.exists()),None)
    need(cudart is not None,'CUDA 13 runtime link library missing')
    with tempfile.TemporaryDirectory(prefix='ispmv-public-') as tmp:
        b=Path(tmp)
        for source in (ROOT/'examples').glob('*'):
            if source.suffix not in ('.c','.cpp'):continue
            compiler,standard=('cc','c11') if source.suffix=='.c' else ('c++','c++17')
            run([compiler,'-std='+standard,'-Wall','-Wextra','-Werror','-I'+str(ROOT/'include'),'-I'+str(runtime/'include'),source,'-L'+str(sdk/'lib'),'-Wl,--allow-shlib-undefined','-lispmv',cudart,'-lm','-o',b/source.stem])
        # Runtime wheels cover direct consumer builds above; CMake clients need
        # the full CUDA development toolkit for their device CSR allocation.
        directories=['examples','benchmarks'] if toolkit else []
        for directory in directories:
            cmd=['cmake','-S',sdk/directory,'-B',b/directory,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(sdk),'-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined']
            if toolkit:cmd+=['-DCUDAToolkit_ROOT='+str(toolkit),'-DCUDA_CUDART='+str(cudart),'-DCUDA_cudart_LIBRARY='+str(cudart)]
            if directory=='examples' and toolkit:cmd+=['-DISPMV_EXAMPLE_DEVICE=ON']
            run(cmd);run(['cmake','--build',b/directory,'--parallel','2'])
    print('PUBLIC_SDK_HASH_ELF_AND_CLIENTS_PASS no_GPU_execution=1')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--headers',action='store_true');p.add_argument('--sdk',type=Path);p.add_argument('--cuda-runtime',type=Path);p.add_argument('--cuda-toolkit',type=Path);p.add_argument('--extract',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args()
    check_repository()
    if a.headers:check_headers()
    if a.extract:need(a.destination is not None,'--destination required');extract(a.extract,a.destination)
    if a.sdk:need(a.cuda_runtime is not None,'--cuda-runtime required');check_sdk(a.sdk.resolve(),a.cuda_runtime.resolve(),a.cuda_toolkit)
