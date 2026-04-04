from langage import tokenize, parse
from pathlib import Path
code = Path('test.obs').read_text(encoding='utf-8')
toks = [t for t in tokenize(code) if t[0] not in ('SKIP', 'NEWLINE')]
print('n tokens', len(toks))
while toks:
    try:
        ast = parse(toks)
        print('ast', ast)
    except Exception as e:
        print('error', e)
        break
print('remaining', toks[:10])
