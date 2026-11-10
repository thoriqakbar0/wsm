"""Teacher code receives only disclosed coefficients in a no-network probe."""
import json
import math

from .binary64 import WeightlessError
from .broker import _loads


def execute_code(runner, source, values, *, timeout, on_dispatch):
    program = '''import contextlib,io,json,sys
data=json.load(sys.stdin)
weights={}
for item in data['values']:
    weights.setdefault(item['tensor'],{})[item['index']]=item['value']
class Output(io.StringIO):
    truncated=False
    def write(self,text):
        remaining=16384-self.tell()
        if len(text)>remaining:
            self.truncated=True
        super().write(text[:remaining])
        return len(text)
output=Output()
try:
    with contextlib.redirect_stdout(output):
        exec(compile(data['source'],'<teacher>','exec'),{'weights':weights})
    result=json.dumps({'values':[{'tensor':name,'index':index,'value':value}
                                 for name,items in weights.items() for index,value in items.items()],
                       'stdout':output.getvalue(),'stdout_truncated':output.truncated},allow_nan=False)
except BaseException:
    result='null'
print(result)
'''
    raw = runner.probe(program, input_data=json.dumps({'source': source, 'values': values}, allow_nan=False).encode(),
                       timeout=timeout, on_dispatch=on_dispatch)
    try:
        response = _loads(raw)
        if (type(response) is not dict or set(response) != {'values', 'stdout', 'stdout_truncated'}
                or type(response['stdout']) is not str or len(response['stdout']) > 16384
                or type(response['stdout_truncated']) is not bool):
            raise ValueError()
        result = response['values']
        if type(result) is not list or len(result) != len({(item['tensor'], item['index']) for item in values}):
            raise ValueError()
        original = {(item['tensor'], item['index']): item['value'] for item in values}
        edits, seen = [], set()
        for item in result:
            if type(item) is not dict or set(item) != {'tensor', 'index', 'value'}:
                raise ValueError()
            if type(item['tensor']) is not str or type(item['index']) is not int:
                raise ValueError()
            address = item['tensor'], item['index']
            value = item['value']
            if address not in original or address in seen or type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError()
            seen.add(address)
            if value != original[address]:
                edits.append(item)
        return {'edits': edits, 'stdout': response['stdout'], 'stdout_truncated': response['stdout_truncated']}
    except (ValueError, TypeError, KeyError, OverflowError, WeightlessError):
        raise WeightlessError('invalid_code', 'Teacher code must preserve disclosed addresses and finite values') from None
