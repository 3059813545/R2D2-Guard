import re
from decimal import Decimal
NUMBER=re.compile(r'-?[0-9]+(?:\.[0-9]+)?\Z',re.ASCII)
def exact_number(s):
    if not isinstance(s,str) or len(s)>128 or not NUMBER.fullmatch(s):return None
    n=Decimal(s)
    if not n.is_finite() or n.copy_abs()>Decimal('1e100'):return None
    # Decimal.normalize obeys ambient precision; format directly to retain every digit.
    if n==0:return '0'
    value=format(n,'f')
    if '.' in value:value=value.rstrip('0').rstrip('.')
    return value

def exact_parse(text,truncated=False):
    p={'version':'decimal_contract_experimental_v2','routed':True,'terminal_field':None,'calculation_field':None}
    if truncated:p['reason']='truncated';return p
    if text.count('####')!=1:p['reason']='missing_or_multiple_terminal_markers';return p
    before,tail=text.split('####')
    final=tail.strip()
    if final.startswith('$'):final=final[1:].strip()
    answer=exact_number(final)
    p['terminal_field']=answer
    if answer is None:p['reason']='unsupported_complete_terminal_token';return p
    matches=re.findall(r'<<([^<>]*)>>',before)
    field=matches[-1].rsplit('=',1)[-1].strip() if matches and '=' in matches[-1] else None
    if field and field.startswith('$'):field=field[1:].strip()
    calc=exact_number(field);p['calculation_field']=calc
    if calc is None:p['reason']='missing_or_unsupported_calculation_token';return p
    p.update(routed=calc!=answer,reason='mismatch' if calc!=answer else 'equal');return p

