from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'

def test_metadata():
    m=json.loads((RES/'metadata.json').read_text())
    assert m['n_bus']==39 and m['n_branch']==46 and m['n_gen']==10
    assert abs(m['load_loss_MW']-2724.476)<1e-6

def test_h3_candidate_gap():
    with open(RES/'h0_h3.csv') as f: rows=list(csv.DictReader(f))
    r=next(x for x in rows if x['noise']=='homogeneous' and x['H']=='H3')
    assert abs(float(r['gap'])-(float(r['J'])-float(r['W'])))<1e-10
    assert 0 < float(r['gap']) < 0.1

def test_curvature_screen_positive():
    c=json.loads((RES/'curvature_screen.json').read_text())
    assert min(c['homogeneous'])>0
    assert min(c['activity_scaled'])>0

def test_terminal_layers_positive():
    t=json.loads((RES/'terminal_layer.json').read_text())
    assert t['homogeneous_ms']>1.0
    assert t['activity_scaled_ms']>1.0
