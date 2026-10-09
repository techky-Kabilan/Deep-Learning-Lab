import json
from pathlib import Path
from PIL import Image
import pytest
from scripts.prepare_rdd2022 import prepare, convert_xml

def fixture(source, count=20):
    images = source / "India/train/images"
    annotations = source / "India/train/annotations/xmls"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (20,20), (i*10,30,40)).save(images / f"sample{i}.jpg")
        (annotations / f"sample{i}.xml").write_text('<annotation><object><name>D40</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object><object><name>D43</name></object></annotation>')

def test_normalization_and_unknown_categories(tmp_path):
    fixture(tmp_path)
    labels, ignored = convert_xml(tmp_path / "India/train/annotations/xmls/sample0.xml",20,20)
    assert labels[0].split() == ["3","0.50000000","0.50000000","1.00000000","1.00000000"]
    assert ignored["D43"] == 1

def test_split_reproducibility_dedup_and_no_overwrite(tmp_path):
    source = tmp_path / "raw"
    fixture(source)
    prepare(source, tmp_path / "a")
    prepare(source, tmp_path / "b")
    a = json.loads((tmp_path / "a/split_manifest.json").read_text())
    b = json.loads((tmp_path / "b/split_manifest.json").read_text())
    assert a == b and {r["split"] for r in a} == {"train","val","test"}
    assert len({r["sha256"] for r in a}) == len(a)
    with pytest.raises(ValueError): prepare(source,tmp_path / "a")

def test_sequence_groups_never_cross_splits(tmp_path):
    source = tmp_path / "raw"
    fixture(source)
    groups = {f"India/train/images/sample{i}.jpg":f"route{i//2}" for i in range(20)}
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(groups))
    prepare(source,tmp_path / "prepared",group_manifest=path)
    records = json.loads((tmp_path / "prepared/split_manifest.json").read_text())
    for group in set(groups.values()):
        assert len({r["split"] for r in records if r["group"] == group}) == 1
