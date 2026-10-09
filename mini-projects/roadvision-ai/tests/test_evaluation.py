import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import yaml
from scripts.evaluate import evaluate

def test_evaluation_saves_measured_values_and_provenance(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    root.mkdir()
    records = [{'split':s,'sha256':s} for s in ('train','val','test')]
    (root/'split_manifest.json').write_text(json.dumps(records))
    data = root/'data.yaml'
    data.write_text(yaml.safe_dump({'path':str(root),'names':{0:'D00',1:'D10',2:'D20',3:'D40'}}))
    weights=tmp_path/'fixture.pt'
    weights.write_bytes(b'not real weights')
    output=tmp_path/'evaluation'
    calls=[]
    class Model:
        names={0:'D00',1:'D10',2:'D20',3:'D40'}
        def __init__(self,*args,**kwargs): pass
        def val(self,**kwargs):
            calls.append(kwargs)
            output.mkdir()
            self.validator=SimpleNamespace(save_dir=output)
            return SimpleNamespace(box=SimpleNamespace(ap_class_index=[3],class_result=lambda i:(.7,.6,.5,.4)),results_dict={'metrics/mAP50(B)':.5})
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Model,__version__='test-fixture'))
    assert evaluate(weights,data,output)==output
    saved=json.loads((output/'metrics.json').read_text())
    assert saved['metrics']['metrics/mAP50(B)']==.5
    assert len(saved['weights_sha256'])==64
    assert calls[0]['split']=='test' and calls[0]['conf']==.001
    assert 'D40' in (output/'per_class.csv').read_text()
    records[2]['sha256']='train'
    (root/'split_manifest.json').write_text(json.dumps(records))
    with pytest.raises(ValueError,match='leakage'):
        evaluate(weights,data,tmp_path/'other')
