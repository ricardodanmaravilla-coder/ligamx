import os, requests
API_KEY=os.environ.get('API_SPORTS_KEY')
BASE_URL='https://v3.football.api-sports.io'
HEADERS={'x-apisports-key':API_KEY}

def obtener_cuotas_partido(fixture_id):
    """Solo devuelve mercados/lineas realmente observados. Nunca inventa 2.5/9.5/4.5."""
    if not fixture_id or not API_KEY: return {}
    url=f'{BASE_URL}/odds'; out={}
    for bookmaker_id in [8,6,11,1]:
        try:
            r=requests.get(url,headers=HEADERS,params={'fixture':fixture_id,'bookmaker':bookmaker_id},timeout=10)
            if r.status_code!=200: continue
            data=r.json().get('response',[])
            if not data: continue
            books=data[0].get('bookmakers',[])
            if not books: continue
            for m in books[0].get('bets',[]):
                name=m.get('name',''); vals=m.get('values',[])
                if name=='Match Winner':
                    for v in vals:
                        mp={'Home':'1','Draw':'X','Away':'2'}
                        if v.get('value') in mp: out[mp[v['value']]]=float(v['odd'])
                elif name=='Goals Over/Under':
                    _extract_total(vals,out,'goles','Goles')
                elif name in ['Corners Over Under','Corners','Total Corners']:
                    _extract_total(vals,out,'corners','Corners')
                elif name in ['Cards Over/Under','Cards','Total Cards']:
                    _extract_total(vals,out,'tarjetas','Tarjetas')
            if any(k in out for k in ('1','X','2')): break
        except Exception:
            continue
    return out

def _extract_total(vals,out,kind,label):
    pairs={}
    for v in vals:
        s=str(v.get('value',''))
        if s.startswith('Over ') or s.startswith('Under '):
            side,line=s.split(' ',1)
            try: odd=float(v.get('odd'))
            except: continue
            pairs.setdefault(line,{})[side]=odd
    complete=[(line,p) for line,p in pairs.items() if 'Over' in p and 'Under' in p]
    if not complete: return
    # choose the most balanced real market line, not an invented default
    line,p=min(complete,key=lambda x:abs(x[1]['Over']-x[1]['Under']))
    out[f'linea_{kind}_detectada']=line
    out[f'Over {line} {label}']=p['Over']; out[f'Under {line} {label}']=p['Under']
    if kind=='goles': out[f'Over {line}']=p['Over']; out[f'Under {line}']=p['Under']

def remove_vig_two_way(odd_a,odd_b):
    ia,ib=1/float(odd_a),1/float(odd_b); s=ia+ib
    return ia/s*100,ib/s*100

def evaluar_mercado(prob_pct,cuota,market_prob_pct=None):
    if not cuota or cuota<=1: return None
    p=prob_pct/100.0; ev=(p*cuota-1)*100
    edge=None if market_prob_pct is None else prob_pct-market_prob_pct
    b=cuota-1; kelly=max(0,((b*p)-(1-p))/b)*100 if b>0 else 0
    return {'cuota':cuota,'prob_modelo':prob_pct,'prob_mercado_no_vig':market_prob_pct,'edge_pp':edge,'ev_pct':ev,'kelly_pct':kelly}
