"""The XML-like text-diagram renderer."""

import pytest

from realsim.tools.check_contract import REPO_ROOT
from realsim.tools.text_diagram import _load, _render


def test_xml_boxes_and_rows_are_rectangular(tmp_path):
    source = tmp_path / "diagram.xml"
    source.write_text("""<diagrams target="out.md">
  <diagram id="sample">
    <row gap="2">
      <box title="left" width="8"><line>a</line><line>b</line></box>
      <between><at row="1">--</at></between>
      <box title="right" width="9"><line>c</line></box>
    </row>
  </diagram>
</diagrams>
""")
    _target, drawings = _load(source)
    drawing = drawings["sample"]
    assert len({len(line) for line in drawing.lines}) == 1


def test_xml_box_rejects_overflow(tmp_path):
    source = tmp_path / "diagram.xml"
    source.write_text("""<diagrams target="out.md">
  <diagram id="sample">
    <box title="small" width="4"><line>does not fit</line></box>
  </diagram>
</diagrams>
""")
    with pytest.raises(ValueError, match="needs width"):
        _load(source)


def test_xml_groups_each_multiline_connector(tmp_path):
    source = tmp_path / "diagram.xml"
    source.write_text("""<diagrams target="out.md">
  <diagram id="sample">
    <place-lines width="12">
      <at column="1"><line>▲ left</line><line>│</line></at>
      <at column="9"><line>│</line><line>▼</line></at>
    </place-lines>
  </diagram>
</diagrams>
""")
    _target, drawings = _load(source)
    assert drawings["sample"].render() == " ▲ left  │\n │       ▼"


def test_sensor_view_document_is_rendered():
    source = REPO_ROOT / "docs" / "sensor_view_selector_flow.diagram.xml"
    target, drawings = _load(source)
    document = (REPO_ROOT / target).read_text()
    assert _render(document, drawings) == document
